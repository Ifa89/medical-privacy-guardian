"""
Attack injection for milestone M2.

Takes the baseline logs produced by generate_logs.py and seeds three insider
threat scenarios into them, preserving ground-truth labels so detection
performance can be measured objectively.

The scenarios are deliberately graded in difficulty. If every attack were an
obvious outlier the model would score well and prove nothing.

Usage:
    python inject_attacks.py --in data/access_logs.csv --out data/labeled_logs.csv
"""

import argparse
import uuid
import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd

SEED = 1337

CLINICAL_UNITS = ["maternity", "icu", "er", "oncology", "radiology", "surgery"]


def token_for_patient(patient_number: int) -> str:
    digest = hashlib.sha256(f"patient-{patient_number}".encode()).hexdigest()
    return f"PT-{digest[:12]}"


def user_baselines(df):
    """Per-user statistics used to scale attacks relative to that person's own
    normal behavior. An attack that is 3x baseline for a quiet user may still
    be below the hospital-wide average, which is exactly the case a global
    threshold would miss."""
    per_day = df.groupby(["user_id", df["timestamp"].dt.date]).size()
    stats = per_day.groupby("user_id").agg(["mean", "std", "max"])
    stats.columns = ["daily_mean", "daily_std", "daily_max"]

    modal_hour = (
        df.groupby("user_id")["timestamp"]
        .apply(lambda s: int(s.dt.hour.mode().iloc[0]))
        .rename("modal_hour")
    )
    home_ws = (
        df.groupby("user_id")["workstation_id"]
        .apply(lambda s: s.mode().iloc[0])
        .rename("home_workstation")
    )
    return stats.join(modal_hour).join(home_ws)


def make_event(rng, user, ts, patient_unit, action, touched, session, workstation,
               attack_type, n_patients):
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": ts,
        "user_id": user["user_id"],
        "role": user["role"],
        "department": user["department"],
        "patient_token": token_for_patient(int(rng.integers(0, n_patients))),
        "patient_department": patient_unit,
        "action": action,
        "records_touched": touched,
        "session_id": session,
        "workstation_id": workstation,
        "is_anomaly": True,
        "attack_type": attack_type,
    }


# --- Scenario 1: after-hours bulk exfiltration -----------------------------
# Loud on every axis: wrong time, huge volume, indiscriminate across units.
# This is the sanity check -- if the model misses this, something is broken.

def after_hours_exfil(rng, user, base, days_range, n_patients):
    rows = []
    day = rng.choice(days_range)
    start = pd.Timestamp(day) + timedelta(hours=int(rng.integers(1, 4)))
    session = str(uuid.uuid4())[:8]

    n_events = int(rng.integers(180, 340))
    duration_h = float(rng.uniform(0.6, 1.8))

    for _ in range(n_events):
        ts = start + timedelta(hours=float(rng.uniform(0, duration_h)))
        action = rng.choice(["view", "export", "print"], p=[0.55, 0.32, 0.13])
        touched = int(rng.integers(6, 30)) if action == "export" else int(rng.integers(1, 5))
        rows.append(make_event(
            rng, user, ts, rng.choice(CLINICAL_UNITS), action, touched,
            session, base["home_workstation"], "after_hours_exfil", n_patients
        ))
    return rows


# --- Scenario 2: record snooping -------------------------------------------
# Normal shift, normal volume, normal workstation. ONLY the cross-department
# ratio moves. This scenario exists to test whether the feature set is doing
# real work or just detecting "lots of activity."

def record_snooping(rng, user, base, days_range, n_patients):
    rows = []
    n_days = int(rng.integers(4, 9))
    start_idx = int(rng.integers(0, max(1, len(days_range) - n_days)))
    own_unit = user["department"]
    targets = [u for u in CLINICAL_UNITS if u != own_unit]

    for d in range(n_days):
        day = days_range[start_idx + d]
        session = str(uuid.uuid4())[:8]
        # a handful of extra lookups per day -- well inside normal daily variance
        n_events = int(rng.integers(6, 16))
        for _ in range(n_events):
            hour = base["modal_hour"] + rng.normal(0, 1.6)
            hour = float(np.clip(hour, 0, 23.9))
            ts = pd.Timestamp(day) + timedelta(hours=hour)
            rows.append(make_event(
                rng, user, ts, rng.choice(targets), "view", 1,
                session, base["home_workstation"], "record_snooping", n_patients
            ))
    return rows


# --- Scenario 3: credential compromise -------------------------------------
# The hardest case. Plausible hours, moderate volume, but originating from a
# workstation the account has never used. Location is the primary tell.

def credential_compromise(rng, user, base, days_range, n_patients):
    rows = []
    n_days = int(rng.integers(1, 4))
    start_idx = int(rng.integers(0, max(1, len(days_range) - n_days)))

    foreign_unit = rng.choice([u for u in CLINICAL_UNITS if u != user["department"]])
    foreign_ws = f"WS-{foreign_unit.upper()[:4]}-{rng.integers(1, 9):02d}"

    for d in range(n_days):
        day = days_range[start_idx + d]
        session = str(uuid.uuid4())[:8]
        # elevated but not absurd: 1.6-2.4x this user's own daily mean
        n_events = int(np.clip(base["daily_mean"] * rng.uniform(1.6, 2.4), 10, None))
        for _ in range(n_events):
            # shifted a few hours off their normal pattern -- suspicious to a
            # per-user model, unremarkable hospital-wide
            hour = base["modal_hour"] + rng.choice([-1, 1]) * rng.uniform(2.5, 5.0)
            hour = float(np.clip(hour + rng.normal(0, 0.7), 0, 23.9))
            ts = pd.Timestamp(day) + timedelta(hours=hour)
            action = rng.choice(["view", "search", "export"], p=[0.62, 0.26, 0.12])
            touched = int(rng.integers(3, 16)) if action in ("search", "export") else 1
            rows.append(make_event(
                rng, user, ts, rng.choice(CLINICAL_UNITS), action, touched,
                session, foreign_ws, "credential_compromise", n_patients
            ))
    return rows


SCENARIOS = {
    "after_hours_exfil": after_hours_exfil,
    "record_snooping": record_snooping,
    "credential_compromise": credential_compromise,
}


def main():
    ap = argparse.ArgumentParser(description="Inject insider threat scenarios into baseline logs.")
    ap.add_argument("--in", dest="infile", default="data/access_logs.csv")
    ap.add_argument("--out", dest="outfile", default="data/labeled_logs.csv")
    ap.add_argument("--n-exfil", type=int, default=2)
    ap.add_argument("--n-snoop", type=int, default=3)
    ap.add_argument("--n-compromise", type=int, default=2)
    ap.add_argument("--patients", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.infile, parse_dates=["timestamp"])

    baselines = user_baselines(df)
    days_range = sorted(df["timestamp"].dt.normalize().unique())

    # attackers are drawn from clinical staff only -- billing and records_admin
    # legitimately touch every unit, so cross-department signals mean nothing
    # for them and the scenarios would not be meaningful
    eligible = (
        df[df["department"].isin(CLINICAL_UNITS)]
        .drop_duplicates("user_id")[["user_id", "role", "department"]]
    )

    plan = (
        [("after_hours_exfil", n) for n in [args.n_exfil]]
        + [("record_snooping", n) for n in [args.n_snoop]]
        + [("credential_compromise", n) for n in [args.n_compromise]]
    )

    chosen = rng.choice(eligible["user_id"].values,
                        size=sum(n for _, n in plan), replace=False)
    cursor = 0
    attack_rows = []
    assignments = []

    for scenario, count in plan:
        for _ in range(count):
            uid = chosen[cursor]; cursor += 1
            user = eligible[eligible["user_id"] == uid].iloc[0]
            base = baselines.loc[uid]
            rows = SCENARIOS[scenario](rng, user, base, days_range, args.patients)
            attack_rows.extend(rows)
            assignments.append({
                "user_id": uid, "role": user["role"],
                "department": user["department"],
                "scenario": scenario, "events": len(rows),
            })

    labeled = pd.concat([df, pd.DataFrame(attack_rows)], ignore_index=True)
    labeled = labeled.sort_values("timestamp").reset_index(drop=True)
    labeled.to_csv(args.outfile, index=False)

    key_path = args.outfile.replace(".csv", "_attack_key.csv")
    pd.DataFrame(assignments).to_csv(key_path, index=False)

    n_atk = int(labeled["is_anomaly"].sum())
    print(f"Wrote {len(labeled):,} events ({n_atk:,} anomalous, "
          f"{n_atk/len(labeled)*100:.2f}% base rate)")
    print(f"  labeled data -> {args.outfile}")
    print(f"  attack key   -> {key_path}")
    print()
    print("Attacks by scenario:")
    print(labeled[labeled.is_anomaly].groupby("attack_type").size().to_string())
    print()
    print("Compromised accounts:")
    print(pd.DataFrame(assignments).to_string(index=False))


if __name__ == "__main__":
    main()
