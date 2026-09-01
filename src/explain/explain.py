"""
Alert explanation layer for milestone M4.

Takes a flagged user-day, assembles the context an analyst would need, and
asks Claude to turn it into a readable explanation with a HIPAA assessment
and recommended actions.

Two design decisions worth stating plainly:

1. No protected health information is sent. Patient identifiers were hashed
   at generation time and only aggregate counts leave this process. The model
   sees "37 distinct patient records across 5 units", never a patient.

2. The model explains and recommends; it does not decide the response tier.
   Tier assignment is deterministic, computed from the risk score and feature
   signature in assign_tier() below. An LLM choosing whether to disable a
   clinician's account would put a non-deterministic component in the control
   path of a patient-safety decision. Explanation is advisory; containment is
   not.

Usage:
    python explain.py --dry-run                 # print the prompt, call nothing
    python explain.py --top 5                   # explain the 5 highest-risk days
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"
CACHE_DIR = Path("data/explanations")

SYSTEM_PROMPT = """You are a security analyst assistant for a hospital SOC. \
You receive statistical summaries of anomalous electronic health record access \
and explain them to an on-call analyst.

You never see patient data. All identifiers are hashed. Do not speculate about \
patient identity or invent details not present in the summary.

Respond with ONLY a JSON object, no markdown fences and no preamble, with these keys:

  "summary": 2-3 sentences stating what happened and why it deviates from this
             user's normal pattern. Reference the specific numbers given.
  "assessment": one of "likely_benign", "needs_review", "likely_incident".
  "reasoning": 1-2 sentences on what most supports your assessment, including
               the innocent explanation if one is plausible.
  "hipaa_concern": the relevant HIPAA Security Rule provision if one applies,
                   or null. Be specific (e.g. "164.312(b) Audit Controls").
  "recommended_actions": array of 2-4 short imperative strings for the analyst.
  "confidence": "low", "medium", or "high" -- how much the data supports your
                assessment. Say low when the signal is ambiguous."""


def load_context(scored_path, logs_path):
    scored = pd.read_csv(scored_path, parse_dates=["date"])
    logs = pd.read_csv(logs_path, parse_dates=["timestamp"])
    logs["date"] = logs["timestamp"].dt.normalize()
    return scored, logs


def build_context(row, logs):
    """Assemble everything the model needs, as aggregates only."""
    uid, date = row["user_id"], row["date"]
    user_all = logs[logs["user_id"] == uid]
    day = user_all[user_all["date"] == date]
    other = user_all[user_all["date"] != date]

    other_daily = other.groupby("date").size()

    hours = sorted(day["timestamp"].dt.hour.unique().tolist())
    normal_hours = sorted(other["timestamp"].dt.hour.value_counts(
        normalize=True).pipe(lambda s: s[s >= 0.05]).index.tolist())

    day_ws = set(day["workstation_id"].unique())
    known_ws = set(other["workstation_id"].value_counts(
        normalize=True).pipe(lambda s: s[s >= 0.05]).index)

    return {
        "account": {
            "user_id": uid,
            "role": row["role"],
            "assigned_department": row["department"],
        },
        "date": str(date.date()),
        "this_day": {
            "total_accesses": int(row["n_events"]),
            "distinct_patient_records": int(row["n_distinct_patients"]),
            "units_accessed": int(row["n_distinct_patient_depts"]),
            "cross_department_share": round(float(row["cross_dept_ratio"]), 3),
            "share_at_unusual_hours": round(float(row["off_hours_ratio"]), 3),
            "share_from_unfamiliar_workstations": round(float(row["new_workstation_ratio"]), 3),
            "share_to_rarely_visited_units": round(float(row["rare_dept_ratio"]), 3),
            "export_share": round(float(row["export_ratio"]), 3),
            "busiest_hour_accesses": int(row["peak_hour_events"]),
            "active_hours": hours,
            "workstations_used": sorted(day_ws),
        },
        "this_users_baseline": {
            "median_daily_accesses": float(other_daily.median()),
            "max_daily_accesses_observed": int(other_daily.max()),
            "days_of_history": int(len(other_daily)),
            "typical_active_hours": normal_hours,
            "usual_workstations": sorted(known_ws),
        },
        "deviation_scores": {
            "access_volume": round(float(row["z_n_events"]), 1),
            "cross_department_rate": round(float(row["z_cross_dept_ratio"]), 1),
            "distinct_records": round(float(row["z_n_distinct_patients"]), 1),
            "records_touched": round(float(row["z_records_touched_sum"]), 1),
            "_note": "robust z-scores against this user's own history; "
                     "roughly, 3+ is unusual and 10+ is extreme",
        },
        "model_risk_score": round(float(row["risk_score"]), 3),
    }


def assign_tier(row):
    """Deterministic response tiering. Deliberately NOT delegated to the model.

    Tier 1 fires automatically (monitor, alert).
    Tier 2 fires automatically with rollback (revoke session, force re-auth).
    Tier 3 requires analyst approval (disable account, quarantine host).
    """
    reasons = []
    tier = 1

    # unfamiliar workstation is the strongest single indicator of a stolen
    # credential, so it escalates on its own
    if row["new_workstation_ratio"] >= 0.5:
        tier = max(tier, 2)
        reasons.append("majority of activity from unfamiliar workstations")

    if row["z_n_events"] >= 10 and row["off_hours_ratio"] >= 0.3:
        tier = max(tier, 3)
        reasons.append("extreme volume outside normal hours")

    if row["export_ratio"] >= 0.2 and row["z_records_touched_sum"] >= 10:
        tier = max(tier, 3)
        reasons.append("bulk export well above baseline")

    if row["rare_dept_ratio"] >= 0.3 and row["z_cross_dept_ratio"] >= 3:
        tier = max(tier, 2)
        reasons.append("sustained access to units this user rarely visits")

    if not reasons:
        reasons.append("anomaly score above threshold, no single strong indicator")

    return tier, reasons


def explain(client, model, context, max_tokens=1000):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(context, indent=2)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": text, "assessment": "parse_error",
                "recommended_actions": [], "confidence": "low",
                "hipaa_concern": None, "reasoning": ""}


TIER_ACTIONS = {
    1: "AUTOMATIC: enrich alert, snapshot session, raise monitoring, notify SOC",
    2: "AUTOMATIC (reversible): revoke session token, require step-up re-authentication",
    3: "STAGED - ANALYST APPROVAL REQUIRED: disable account, quarantine workstation",
}


def render(ctx, tier, tier_reasons, result, truth=None):
    a = ctx["account"]
    print("=" * 78)
    print(f"ALERT  {a['user_id']}  {a['role']} / {a['assigned_department']}  {ctx['date']}")
    print(f"       risk score {ctx['model_risk_score']}  |  response tier {tier}")
    print("=" * 78)
    print()
    print(f"  ASSESSMENT: {result.get('assessment', '?')}  "
          f"(confidence: {result.get('confidence', '?')})")
    print()
    print("  " + result.get("summary", "").replace("\n", "\n  "))
    print()
    if result.get("reasoning"):
        print("  Reasoning: " + result["reasoning"].replace("\n", "\n  "))
        print()
    if result.get("hipaa_concern"):
        print(f"  HIPAA: {result['hipaa_concern']}")
        print()
    print("  Recommended actions:")
    for act in result.get("recommended_actions", []):
        print(f"    - {act}")
    print()
    print(f"  Tier {tier} -> {TIER_ACTIONS[tier]}")
    for r in tier_reasons:
        print(f"    triggered by: {r}")
    if truth is not None:
        print()
        print(f"  [ground truth: {truth}]")
    print()


def main():
    ap = argparse.ArgumentParser(description="Generate analyst explanations for flagged days.")
    ap.add_argument("--scored", default="data/scored.csv")
    ap.add_argument("--logs", default="data/labeled_logs.csv")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the assembled context and exit without calling the API")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    scored, logs = load_context(args.scored, args.logs)
    flagged = scored[scored["pred"] == 1].nlargest(args.top, "risk_score")

    if len(flagged) == 0:
        print("No flagged user-days found.")
        return

    if args.dry_run:
        row = flagged.iloc[0]
        ctx = build_context(row, logs)
        tier, reasons = assign_tier(row)
        print("SYSTEM PROMPT:")
        print(SYSTEM_PROMPT)
        print()
        print("USER MESSAGE:")
        print(json.dumps(ctx, indent=2))
        print()
        print(f"Deterministic tier: {tier}  ({'; '.join(reasons)})")
        print()
        print(f"[dry run: no API call made. {len(flagged)} day(s) would be explained]")
        return

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found. Put it in .env")

    from anthropic import Anthropic
    client = Anthropic()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in flagged.iterrows():
        ctx = build_context(row, logs)
        tier, reasons = assign_tier(row)

        cache_file = CACHE_DIR / f"{row['user_id']}_{row['date'].date()}.json"
        if cache_file.exists() and not args.no_cache:
            result = json.loads(cache_file.read_text())
        else:
            result = explain(client, args.model, ctx)
            cache_file.write_text(json.dumps(result, indent=2))

        truth = row["attack_type"] if row["is_anomaly"] else "not an attack (false positive)"
        render(ctx, tier, reasons, result, truth)


if __name__ == "__main__":
    main()
