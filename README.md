# Medical Data Privacy Guardian

Insider threat detection for healthcare EHR access logs. Learns a behavioral
baseline for each individual user, flags deviations with an Isolation Forest
model, and uses the Anthropic API to turn each flag into a plain-language
explanation for a security analyst.

Healthcare breaches take an average of 279 days to detect. This system is built
to compress that to hours, and to explain what it found rather than emitting a
score.

**Status:** M1–M3 complete, M4 in progress. This is an active personal project.

---

## Results

Evaluated on 3,718 user-days containing 7 compromised accounts across three
attack scenarios (0.59% anomalous user-days).

| Metric | Per-user model | Naive global threshold |
|---|---|---|
| Precision | 0.294 | 0.077 |
| Recall (user-day) | 0.455 | 0.136 |
| **F1** | **0.357** | 0.098 |
| Accounts detected | **7 / 7** | — |
| Mean detection latency | **0.9 days** | — |

The naive baseline is a global threshold on raw event volume, tuned to raise
the same number of alerts so the comparison is fair. Per-user normalization
measures about 3.6× better by F1.

Recall by scenario:

| Scenario | Caught |
|---|---|
| After-hours bulk exfiltration | 2 / 2 (100%) |
| Credential compromise | 2 / 2 (100%) |
| Low-and-slow record snooping | 6 / 18 (33%) |

**Account-level recall is the number that matters operationally.** An analyst
investigates a person, not a calendar square — catching a snooping nurse on day
2 of 6 ends the breach on day 2. All 7 accounts were caught, with a mean delay
of under a day.

---

## Pipeline

```
generate_logs.py  ->  inject_attacks.py  ->  features.py  ->  detect.py  ->  explain.py
   synthetic            labeled attack        user-day        Isolation       analyst
   baseline             scenarios             features        Forest          explanation
```

---

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/generator/generate_logs.py   --users 150 --days 30 --out data/access_logs.csv
python src/generator/inject_attacks.py  --in data/access_logs.csv --out data/labeled_logs.csv
python src/detection/features.py        --in data/labeled_logs.csv --out data/features.csv
python src/detection/detect.py          --in data/features.csv
```

All random seeds are fixed, so a fresh clone reproduces the numbers above
exactly. Generated data is gitignored — it is derived, not source.

For the explanation layer, add an Anthropic API key to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python src/explain/explain.py --dry-run     # prints the prompt, calls nothing
python src/explain/explain.py --top 5
```

---

## Design decisions

**Per-user baselines, not global thresholds.** A busy nurse and a compromised
account look identical to a global threshold. One compromised account in the
test set generated 28 accesses on its attack day — more than double that user's
own median, but *below* the hospital-wide average. Only a per-user model sees
it.

**Robust statistics (median and MAD), not mean and standard deviation.** In
production there are no labels, so a user's baseline is necessarily computed
from data that may already contain their attack. Median and MAD tolerate that
contamination.

**Leave-one-day-out baselines.** A user's normal is computed from days *other*
than the day being scored. Without this, a multi-day attack quietly becomes part
of the baseline it is supposed to deviate from — in an earlier iteration this
erased the workstation signal entirely.

**Separate models per role group.** Billing and records staff legitimately touch
every unit, so their normal cross-department behavior was defining the outlier
boundary for clinical staff. Splitting the models was a large part of the jump
from F1 0.167 to 0.357.

**No PHI reaches the model.** Patient identifiers are SHA-256 tokenized at
generation. The explanation layer sends only aggregates — "240 distinct records
across 6 units", never a patient.

**The LLM explains; it does not decide.** Response tiering is deterministic
rule-based logic. An LLM choosing whether to disable a clinician's account would
place a non-deterministic component in the control path of a patient-safety
decision. Tier 1 (monitor, alert) and Tier 2 (revoke session, force re-auth)
fire automatically; Tier 3 (disable account, quarantine host) requires analyst
approval.

---

## Known limitations

- **Precision is 0.294.** Roughly two in three alerts are false positives. At
  150 users that is about one nuisance alert per day; at 5,000 users it would be
  30+, which is how alert fatigue starts. Not yet solved.
- **Low-and-slow snooping is caught on only a third of its active days.** The
  scenario adds fewer events than the user's normal daily volume, at normal
  hours, from a normal workstation. Detected at the account level, but not
  reliably per-day.
- **Synthetic data only.** No real patient data is used. This is a legal
  necessity and a genuine constraint on generalization.
- **Isolation Forest assumes anomalies are rare and separable.** A slow
  exfiltration campaign deliberately staying within baseline may evade it.
- **New users have no baseline** and will generate false positives during their
  first weeks.
- **Cross-department signal is meaningless for back-office roles** by
  construction, so detection there relies on volume, timing, and export
  behavior.

---

## Roadmap

- [x] M1 — Synthetic EHR access log generator
- [x] M2 — Attack injection with ground-truth labels
- [x] M3 — Isolation Forest on per-user baselines
- [ ] M4 — Anthropic API explanation layer
- [ ] M5 — Tiered automated response with audit logging
- [ ] M6 — React analyst dashboard

---

## Stack

Python 3.12 · scikit-learn · pandas · NumPy · Anthropic API · React (planned)

Runs entirely locally. No cloud dependency, no GPU required.
