"""
Synthetic EHR access log generator.

Produces realistic baseline activity for a fictional hospital: staff work
assigned shifts, touch patients mostly from their own unit, and use
workstations in their own department. No attack behavior is injected here --
that is milestone M2.

Usage:
    python generate_logs.py --users 150 --days 30 --out ../../data/access_logs.csv
"""

import argparse
import uuid
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42

# --- Hospital structure -----------------------------------------------------

CLINICAL_UNITS = ["maternity", "icu", "er", "oncology", "radiology", "surgery"]
BACK_OFFICE = ["billing", "health_information"]
ALL_UNITS = CLINICAL_UNITS + BACK_OFFICE

# role -> (units it can belong to, share of staff, records touched per shift)
ROLES = {
    "nurse":      {"units": CLINICAL_UNITS, "share": 0.44, "volume": (35, 12)},
    "physician":  {"units": CLINICAL_UNITS, "share": 0.20, "volume": (18, 7)},
    "lab_tech":   {"units": ["radiology", "oncology", "er"], "share": 0.12, "volume": (26, 9)},
    "billing":    {"units": ["billing"], "share": 0.14, "volume": (48, 14)},
    "records_admin": {"units": ["health_information"], "share": 0.10, "volume": (30, 10)},
}

# shift -> (start hour, length in hours)
SHIFTS = {
    "day":     (7, 12),
    "night":   (19, 12),
    "business": (8, 9),
}

# how often each role picks a patient from outside its own unit
CROSS_UNIT_RATE = {
    "nurse": 0.04,
    "physician": 0.14,     # consulting physicians legitimately roam
    "lab_tech": 0.30,      # labs serve the whole hospital
    "billing": 0.85,       # billing touches every unit by design
    "records_admin": 0.70,
}

# action mix per role
ACTION_MIX = {
    "nurse":         {"view": 0.62, "edit": 0.30, "print": 0.05, "search": 0.03, "export": 0.00},
    "physician":     {"view": 0.55, "edit": 0.33, "print": 0.06, "search": 0.06, "export": 0.00},
    "lab_tech":      {"view": 0.70, "edit": 0.20, "print": 0.06, "search": 0.04, "export": 0.00},
    "billing":       {"view": 0.60, "edit": 0.10, "print": 0.10, "search": 0.14, "export": 0.06},
    "records_admin": {"view": 0.50, "edit": 0.12, "print": 0.12, "search": 0.20, "export": 0.06},
}


def token_for_patient(patient_number: int) -> str:
    """Patient identifiers are hashed at generation time. Nothing downstream
    ever sees a raw identifier -- this mirrors the tokenization stage of the
    real pipeline."""
    digest = hashlib.sha256(f"patient-{patient_number}".encode()).hexdigest()
    return f"PT-{digest[:12]}"


def build_staff(rng, n_users):
    """Assign every staff member a role, unit, shift, workstation pool, and a
    personal busyness multiplier so no two users look identical."""
    roles, weights = zip(*[(r, ROLES[r]["share"]) for r in ROLES])
    weights = np.array(weights) / sum(weights)

    staff = []
    for i in range(n_users):
        role = rng.choice(roles, p=weights)
        unit = rng.choice(ROLES[role]["units"])

        if role in ("billing", "records_admin"):
            shift = "business"
        else:
            shift = rng.choice(["day", "night"], p=[0.65, 0.35])

        staff.append({
            "user_id": f"U{i:04d}",
            "role": role,
            "department": unit,
            "shift": shift,
            # each user favors a couple of terminals in their own unit
            "workstations": [
                f"WS-{unit.upper()[:4]}-{rng.integers(1, 9):02d}" for _ in range(2)
            ],
            # some people are simply busier than others; stays stable over time
            "intensity": float(np.clip(rng.normal(1.0, 0.22), 0.5, 1.9)),
            # probability of working any given weekend day
            "weekend_rate": 0.85 if shift != "business" else 0.06,
        })
    return pd.DataFrame(staff)


def pick_patient_unit(rng, user_unit, role):
    """Most access is to patients in the user's own unit. The rest is spread
    across the hospital, which is normal and must not look anomalous."""
    if rng.random() < CROSS_UNIT_RATE[role]:
        others = [u for u in CLINICAL_UNITS if u != user_unit]
        return rng.choice(others)
    return user_unit if user_unit in CLINICAL_UNITS else rng.choice(CLINICAL_UNITS)


def generate_events(rng, staff, start_date, days, n_patients):
    rows = []

    for _, user in staff.iterrows():
        mix_actions = list(ACTION_MIX[user["role"]].keys())
        mix_probs = np.array(list(ACTION_MIX[user["role"]].values()))
        mix_probs = mix_probs / mix_probs.sum()

        vol_mean, vol_sd = ROLES[user["role"]]["volume"]
        shift_start_hour, shift_len = SHIFTS[user["shift"]]

        for day in range(days):
            date = start_date + timedelta(days=day)
            is_weekend = date.weekday() >= 5

            # not everyone works every day
            if is_weekend:
                if rng.random() > user["weekend_rate"]:
                    continue
            else:
                if rng.random() < 0.14:      # days off, PTO, sick
                    continue

            n_events = int(np.clip(
                rng.normal(vol_mean * user["intensity"], vol_sd), 4, None
            ))
            if is_weekend:
                n_events = int(n_events * 0.75)   # lighter weekend census

            session_id = str(uuid.uuid4())[:8]
            workstation = rng.choice(user["workstations"])
            shift_begin = date.replace(hour=0, minute=0, second=0) + timedelta(hours=int(shift_start_hour))

            # Charting is bursty: activity clusters into rounds rather than
            # spreading evenly across the shift.
            n_bursts = max(2, int(n_events / rng.integers(4, 9)))
            burst_offsets = np.sort(rng.uniform(0.05, 0.95, n_bursts)) * shift_len

            for e in range(n_events):
                burst = burst_offsets[rng.integers(0, n_bursts)]
                jitter = rng.normal(0, 0.28)          # minutes-scale spread
                hours_in = float(np.clip(burst + jitter, 0, shift_len - 0.01))
                ts = shift_begin + timedelta(hours=hours_in)

                patient_unit = pick_patient_unit(rng, user["department"], user["role"])
                patient_no = int(rng.integers(0, n_patients))
                action = rng.choice(mix_actions, p=mix_probs)

                # bulk actions touch more than one record
                if action in ("search", "export"):
                    touched = int(rng.integers(2, 14))
                elif action == "print":
                    touched = int(rng.integers(1, 4))
                else:
                    touched = 1

                rows.append({
                    "event_id": str(uuid.uuid4()),
                    "timestamp": ts,
                    "user_id": user["user_id"],
                    "role": user["role"],
                    "department": user["department"],
                    "patient_token": token_for_patient(patient_no),
                    "patient_department": patient_unit,
                    "action": action,
                    "records_touched": touched,
                    "session_id": session_id,
                    "workstation_id": workstation,
                    "is_anomaly": False,
                    "attack_type": None,
                })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic EHR access logs.")
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--patients", type=int, default=4000)
    ap.add_argument("--start", type=str, default="2026-08-01")
    ap.add_argument("--out", type=str, default="data/access_logs.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    staff = build_staff(rng, args.users)
    events = generate_events(rng, staff, start_date, args.days, args.patients)

    events.to_csv(args.out, index=False)

    staff_out = args.out.replace(".csv", "_staff.csv")
    staff.drop(columns=["workstations"]).to_csv(staff_out, index=False)

    print(f"Wrote {len(events):,} events for {len(staff)} users over {args.days} days")
    print(f"  events -> {args.out}")
    print(f"  roster -> {staff_out}")
    print()
    print("Events per role:")
    print(events.groupby("role").size().to_string())
    print()
    print("Hour-of-day distribution (top 6):")
    print(events["timestamp"].dt.hour.value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
