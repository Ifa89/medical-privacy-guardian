"""
Anomaly detection and evaluation for milestone M3.

Trains an Isolation Forest on the user-day feature table and scores every
user-day. Also trains a deliberately naive baseline -- a global threshold on
raw event volume -- so the value of per-user modeling can be measured rather
than assumed.

Isolation Forest is unsupervised: it never sees the labels. Labels are used
only to score the result afterward. The `contamination` parameter tells the
model roughly what fraction of data to treat as outlying, which in a real
deployment would be set by how many alerts the security team can handle per
day, not by the true attack rate (which nobody knows).

Usage:
    python detect.py --in data/features.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

SEED = 42


def evaluate(y_true, y_pred, label):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"--- {label} ---")
    print(f"  true positives  {tp:5d}   false positives {fp:5d}")
    print(f"  false negatives {fn:5d}   true negatives  {tn:5d}")
    print(f"  precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
    print(f"  alert volume: {tp + fp} of {len(y_true)} user-days "
          f"({(tp + fp) / len(y_true) * 100:.2f}%)")
    print()
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def per_scenario_recall(df):
    print("--- recall by scenario ---")
    for scenario, g in df[df.is_anomaly].groupby("attack_type"):
        caught = int(g["pred"].sum())
        print(f"  {scenario:24s} {caught:2d}/{len(g):2d} "
              f"({caught / len(g) * 100:5.1f}%)")
    print()


def detection_latency(df):
    """Days between the first anomalous day for a user and the first day the
    system flagged them. Zero means caught on day one."""
    print("--- detection latency by compromised account ---")
    rows = []
    for uid, g in df[df.is_anomaly].groupby("user_id"):
        g = g.sort_values("date")
        first_attack = g["date"].iloc[0]
        flagged = g[g["pred"] == 1]
        if len(flagged) == 0:
            rows.append((uid, g["attack_type"].iloc[0], None))
            print(f"  {uid}  {g['attack_type'].iloc[0]:24s} NOT DETECTED")
        else:
            delay = (flagged["date"].iloc[0] - first_attack).days
            rows.append((uid, g["attack_type"].iloc[0], delay))
            print(f"  {uid}  {g['attack_type'].iloc[0]:24s} {delay} day(s)")
    detected = [d for _, _, d in rows if d is not None]
    if detected:
        print(f"  mean latency across detected accounts: {np.mean(detected):.1f} days")
    print()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the detection model.")
    ap.add_argument("--in", dest="infile", default="data/features.csv")
    ap.add_argument("--out", dest="outfile", default="data/scored.csv")
    ap.add_argument("--contamination", type=float, default=0.01)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["date"])
    y = df["is_anomaly"].astype(int).values

    X = df[FEATURE_COLUMNS].fillna(0.0).values
    # Isolation Forest is not scale-sensitive in principle, but standardizing
    # keeps the split thresholds interpretable when features have wildly
    # different ranges (z-scores up to 25 next to ratios bounded at 1).
    X = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X)

    # sklearn returns -1 for outliers; higher score_samples means more normal,
    # so negate to get an intuitive "risk score"
    df["pred"] = (model.predict(X) == -1).astype(int)
    df["risk_score"] = -model.score_samples(X)

    print(f"Scored {len(df):,} user-days on {len(FEATURE_COLUMNS)} features")
    print(f"True anomalous user-days: {int(y.sum())} ({y.mean()*100:.2f}%)")
    print()

    evaluate(y, df["pred"].values, "Isolation Forest (per-user normalized features)")
    per_scenario_recall(df)
    detection_latency(df)

    # --- naive baseline -----------------------------------------------------
    # A global volume threshold, tuned to raise the same number of alerts, so
    # the comparison is fair. This is the model a reasonable person would build
    # first, and the number it produces is the argument for per-user baselines.
    n_alerts = int(df["pred"].sum())
    threshold = np.sort(df["n_events"].values)[-n_alerts]
    naive_pred = (df["n_events"].values >= threshold).astype(int)
    print(f"(naive baseline flags any user-day with >= {threshold} events)")
    evaluate(y, naive_pred, "Naive global volume threshold")

    print("--- top 15 by risk score ---")
    top = df.nlargest(15, "risk_score")[
        ["user_id", "date", "n_events", "cross_dept_ratio",
         "new_workstation_ratio", "risk_score", "is_anomaly", "attack_type"]
    ]
    print(top.to_string(index=False))

    df.to_csv(args.outfile, index=False)
    print()
    print(f"Scored data -> {args.outfile}")


if __name__ == "__main__":
    main()
