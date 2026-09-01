"""
Feature engineering for milestone M3.

Aggregates raw access events into one row per user-day, then expresses each
feature relative to that user's own history rather than a hospital-wide
average.

Why per user-day: a single access event carries almost no signal. One chart
view at 3am is indistinguishable from any other view. The attack scenarios
only become visible in aggregate, and a daily window is the shortest window
that still captures the low-and-slow snooping scenario.

Why leave-one-day-out baselines: a user's "normal" must be computed from days
OTHER than the day being scored. Otherwise a multi-day attack quietly becomes
part of the baseline it is supposed to deviate from, and the workstation and
off-hours signals disappear entirely. This mirrors how a production system
would score against a trailing window.

Why robust statistics: baselines use median and MAD rather than mean and
standard deviation. In production there are no labels, so a baseline is
necessarily built from data that may already contain some attack activity.
Median and MAD tolerate that contamination; mean and standard deviation do not.

Usage:
    python features.py --in data/labeled_logs.csv --out data/features.csv
"""

import argparse
import numpy as np
import pandas as pd

# minimum share of a user's activity before an hour or workstation counts as
# "familiar" -- low enough to allow genuine variety, high enough that a single
# unusual day does not establish a new normal
FAMILIARITY_THRESHOLD = 0.05

# MAD floors keep z-scores finite when a user's behavior is almost perfectly
# constant. Without these, a nurse whose cross-department ratio never varies
# produces a division by ~0 and a meaningless five-figure z-score.
RATIO_FLOOR = 0.05
COUNT_FLOOR_FRAC = 0.15
COUNT_FLOOR_MIN = 1.0

Z_CLIP = 25.0


def mad(series):
    """Median absolute deviation, scaled to be comparable to a standard
    deviation for normally distributed data."""
    med = series.median()
    return 1.4826 * float((series - med).abs().median())


def build_daily_aggregates(df):
    """One row per user-day of raw, un-normalized measurements."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["is_cross_dept"] = (df["department"] != df["patient_department"]).astype(int)

    rows = []
    for (uid, date), g in df.groupby(["user_id", "date"]):
        per_hour = g.groupby("hour").size()
        peak = int(per_hour.max())
        rows.append({
            "user_id": uid,
            "date": date,
            "n_events": len(g),
            "cross_dept_ratio": float(g["is_cross_dept"].mean()),
            "n_distinct_patients": int(g["patient_token"].nunique()),
            "n_distinct_patient_depts": int(g["patient_department"].nunique()),
            "n_distinct_workstations": int(g["workstation_id"].nunique()),
            "records_touched_sum": int(g["records_touched"].sum()),
            "records_touched_max": int(g["records_touched"].max()),
            "export_ratio": float((g["action"] == "export").mean()),
            "search_ratio": float((g["action"] == "search").mean()),
            "peak_hour_events": peak,
            "burst_intensity": peak / len(g),
            # labels travel alongside the features for evaluation only and are
            # excluded from FEATURE_COLUMNS below
            "is_anomaly": bool(g["is_anomaly"].any()),
            "attack_type": (g.loc[g["is_anomaly"], "attack_type"].iloc[0]
                            if g["is_anomaly"].any() else None),
        })
    return pd.DataFrame(rows), df


def add_leave_one_out_profile_features(daily, events):
    """For each user-day, measure how much of that day's activity happened at
    hours or on workstations the user does not use on their OTHER days."""
    records = []

    for uid, user_events in events.groupby("user_id"):
        by_date = {d: g for d, g in user_events.groupby("date")}
        for date, target in by_date.items():
            other = user_events[user_events["date"] != date]
            if len(other) == 0:
                records.append((uid, date, 0.0, 0.0))
                continue

            hour_share = other["hour"].value_counts(normalize=True)
            familiar_hours = set(hour_share[hour_share >= FAMILIARITY_THRESHOLD].index)

            ws_share = other["workstation_id"].value_counts(normalize=True)
            familiar_ws = set(ws_share[ws_share >= FAMILIARITY_THRESHOLD].index)

            records.append((
                uid, date,
                float((~target["hour"].isin(familiar_hours)).mean()),
                float((~target["workstation_id"].isin(familiar_ws)).mean()),
            ))

    prof = pd.DataFrame(records, columns=[
        "user_id", "date", "off_hours_ratio", "new_workstation_ratio"])
    return daily.merge(prof, on=["user_id", "date"], how="left")


def add_leave_one_out_zscores(daily, count_cols, ratio_cols):
    """Express each measurement as a deviation from the user's median across
    their OTHER days. Leaving the day out matters most for short, intense
    attacks, which would otherwise drag their own median upward."""
    out = daily.reset_index(drop=True).copy()

    for col in list(count_cols) + list(ratio_cols):
        z = np.zeros(len(out))
        for uid, pos in out.groupby("user_id").groups.items():
            pos = list(pos)
            vals = out.loc[pos, col].astype(float)
            for i in pos:
                others = vals.drop(i)
                if len(others) < 2:
                    z[i] = 0.0
                    continue
                center = others.median()
                spread = mad(others)
                if col in ratio_cols:
                    spread = max(spread, RATIO_FLOOR)
                else:
                    spread = max(spread, COUNT_FLOOR_MIN, COUNT_FLOOR_FRAC * abs(center))
                z[i] = (vals.loc[i] - center) / spread
        out[f"z_{col}"] = np.clip(z, -Z_CLIP, Z_CLIP)
    return out


NORMALIZE_COUNTS = [
    "n_events", "n_distinct_patients", "n_distinct_patient_depts",
    "records_touched_sum", "peak_hour_events",
]
NORMALIZE_RATIOS = ["cross_dept_ratio"]

FEATURE_COLUMNS = [
    "z_n_events",
    "z_cross_dept_ratio",
    "z_n_distinct_patients",
    "z_n_distinct_patient_depts",
    "z_records_touched_sum",
    "z_peak_hour_events",
    "cross_dept_ratio",
    "off_hours_ratio",
    "new_workstation_ratio",
    "export_ratio",
    "search_ratio",
    "burst_intensity",
    "n_distinct_workstations",
]


def build_features(df):
    daily, events = build_daily_aggregates(df)
    daily = add_leave_one_out_profile_features(daily, events)
    daily = add_leave_one_out_zscores(daily, NORMALIZE_COUNTS, set(NORMALIZE_RATIOS))
    return daily.sort_values(["date", "user_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Aggregate access logs into user-day features.")
    ap.add_argument("--in", dest="infile", default="data/labeled_logs.csv")
    ap.add_argument("--out", dest="outfile", default="data/features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["timestamp"])
    feat = build_features(df)
    feat.to_csv(args.outfile, index=False)

    print(f"Built {len(feat):,} user-day rows from {len(df):,} events")
    print(f"  -> {args.outfile}")
    print()
    print(f"Anomalous user-days: {int(feat.is_anomaly.sum())} "
          f"({feat.is_anomaly.mean()*100:.2f}%)")
    print()
    print("Anomalous days by scenario:")
    print(feat[feat.is_anomaly].groupby("attack_type").size().to_string())
    print()
    print("Feature means, normal vs anomalous:")
    comp = feat.groupby("is_anomaly")[FEATURE_COLUMNS].mean().T
    comp.columns = ["normal", "anomalous"]
    print(comp.round(2).to_string())


if __name__ == "__main__":
    main()
