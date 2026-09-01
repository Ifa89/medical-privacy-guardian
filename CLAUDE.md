# Medical Data Privacy Guardian

## What this is

An ML-based insider threat detection system for healthcare environments. It
learns a per-user behavioral baseline from EHR access logs, flags deviations
with an Isolation Forest model, uses the Anthropic API to generate
analyst-readable explanations of each flag, and executes a tiered automated
response.

This is a personal portfolio project. The goal is a working system with
measured detection performance, not a prototype.

## Environment

- Windows, PowerShell
- Python 3.12, venv at `.venv` (activate with `.\.venv\Scripts\Activate.ps1`)
- pandas, numpy, scikit-learn, matplotlib, seaborn, faker, python-dotenv, anthropic
- No GPU, no cloud, no Docker. Everything runs locally.
- API key lives in `.env` as `ANTHROPIC_API_KEY`. Never hardcode it.

## Layout

```
src/generator/   M1-M2  synthetic log generation and attack injection
src/detection/   M3     feature engineering and Isolation Forest models
src/explain/     M4     context aggregation and Anthropic API integration
src/response/    M5     tiered response engine and audit logging
dashboard/       M6     React interface
data/            generated CSVs (gitignored)
notebooks/       exploration and plots
```

## Milestones

| ID | Scope | Exit criteria |
|----|-------|---------------|
| M1 | Synthetic EHR access log generator | Data plausible enough that anomalies aren't visually obvious |
| M2 | Attack injection with ground-truth labels | Labeled dataset exists; detection is measurable |
| M3 | Isolation Forest on per-user baselines | Documented precision, recall, detection latency |
| M4 | Anthropic API explanation layer | CLI demo producing analyst-grade summaries |
| M5 | Tiered automated response | End-to-end run with no manual intervention |
| M6 | React dashboard | Demonstrable to a non-technical viewer |

M1 is done. Currently working on M2.

## Design decisions already made

- **Per-user baselines, not global.** A user is compared against their own
  history. Global thresholds can't separate a busy nurse from a compromised
  account.
- **Patient identifiers are SHA-256 tokenized at generation.** No raw
  identifier ever reaches the model or API layer.
- **Response autonomy scales with reversibility.** Tier 1 (alerting,
  monitoring) fires automatically. Tier 2 (session revocation, step-up MFA)
  fires automatically with rollback. Tier 3 (account disable, host
  quarantine) requires analyst approval, because a false positive that locks
  out a clinician is a patient-safety event.
- **Every action is written to an immutable audit log** with the triggering
  anomaly, model confidence, and generated explanation.

## Known constraints

- Cross-department access carries no signal for `billing` and
  `records_admin` roles, since those roles legitimately touch every unit.
  Detection for them must rely on volume, timing, and export actions.
- Isolation Forest assumes anomalies are rare and separable. Slow, low-volume
  exfiltration staying within baseline may evade it.
- New users have no baseline and will generate false positives.

## How I want you to work with me

- Explain the reasoning behind ML and detection choices; don't just write the
  code. I need to be able to defend every design decision in an interview.
- Ask before adding dependencies.
- Prefer clear code over clever code.
- When something won't work well, say so directly rather than building it
  anyway.
