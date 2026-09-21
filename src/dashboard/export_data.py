"""
Export scored alerts and their explanations into a single JSON bundle the
dashboard reads.

The dashboard is a static React app with no backend, so everything it needs is
flattened into one file at build time. Patient tokens never appear here -- only
the aggregate counts already used by the detection and explanation layers.

Usage:
    python src/dashboard/export_data.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Tier rules live in the explanation layer and must not be duplicated here --
# two copies of a security control's logic will drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "explain"))
from explain import assign_tier  # noqa: E402


def build_daily_series(logs, user_id):
    """That user's access count for every day, so the dashboard can draw their
    own baseline rather than a hospital-wide one."""
    u = logs[logs["user_id"] == user_id]
    counts = u.groupby(u["timestamp"].dt.normalize()).size()
    return [{"date": str(d.date()), "n": int(n)} for d, n in counts.items()]


def main():
    ap = argparse.ArgumentParser(description="Export dashboard data bundle.")
    ap.add_argument("--scored", default="data/scored.csv")
    ap.add_argument("--logs", default="data/labeled_logs.csv")
    ap.add_argument("--explanations", default="data/explanations")
    ap.add_argument("--out", default="dashboard/public/data.json")
    args = ap.parse_args()

    scored = pd.read_csv(args.scored, parse_dates=["date"])
    logs = pd.read_csv(args.logs, parse_dates=["timestamp"])
    exp_dir = Path(args.explanations)

    flagged = scored[scored["pred"] == 1].sort_values("risk_score", ascending=False)

    alerts = []
    for _, r in flagged.iterrows():
        key = f"{r['user_id']}_{r['date'].date()}"
        exp_file = exp_dir / f"{key}.json"
        explanation = json.loads(exp_file.read_text()) if exp_file.exists() else None

        tier, tier_reasons = assign_tier(r)

        alerts.append({
            "id": key,
            "user_id": r["user_id"],
            "role": r["role"],
            "department": r["department"],
            "date": str(r["date"].date()),
            "risk_score": round(float(r["risk_score"]), 3),
            "tier": tier,
            "tier_reasons": tier_reasons,
            "is_anomaly": bool(r["is_anomaly"]),
            "attack_type": r["attack_type"] if pd.notna(r["attack_type"]) else None,
            "measures": {
                "n_events": int(r["n_events"]),
                "distinct_patients": int(r["n_distinct_patients"]),
                "units": int(r["n_distinct_patient_depts"]),
                "cross_dept_ratio": round(float(r["cross_dept_ratio"]), 3),
                "off_hours_ratio": round(float(r["off_hours_ratio"]), 3),
                "new_workstation_ratio": round(float(r["new_workstation_ratio"]), 3),
                "rare_dept_ratio": round(float(r["rare_dept_ratio"]), 3),
                "export_ratio": round(float(r["export_ratio"]), 3),
                "peak_hour_events": int(r["peak_hour_events"]),
            },
            "deviation": {
                "events": round(float(r["z_n_events"]), 1),
                "cross_dept": round(float(r["z_cross_dept_ratio"]), 1),
                "distinct_patients": round(float(r["z_n_distinct_patients"]), 1),
                "records_touched": round(float(r["z_records_touched_sum"]), 1),
            },
            "series": build_daily_series(logs, r["user_id"]),
            "explanation": explanation,
        })

    y = scored["is_anomaly"].astype(int)
    p = scored["pred"].astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    accounts_total = int(scored[scored["is_anomaly"]]["user_id"].nunique())
    caught = scored[(scored["is_anomaly"]) & (scored["pred"] == 1)]["user_id"].nunique()

    bundle = {
        "generated_from": {
            "user_days": int(len(scored)),
            "users": int(scored["user_id"].nunique()),
            "days": int(scored["date"].nunique()),
        },
        "metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3)
                  if (precision + recall) else 0,
            "alerts": tp + fp,
            "accounts_caught": int(caught),
            "accounts_total": accounts_total,
        },
        "alerts": alerts,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {len(alerts)} alerts -> {out}")
    print(f"  {sum(1 for a in alerts if a['explanation'])} have explanations")


if __name__ == "__main__":
    main()
