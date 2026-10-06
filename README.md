# Rule-Based Detection of Insider Threats in EHR Systems Using OpenEMR Audit Logs

Final Year Project, BSc Computer Science (2:1), Brunel University London, 2026
Supervisor: Dr. Cigdem Sengul | Module: CS3072

---

## Engineering Highlights

- **Python detection engine** with 10 configurable rules, an explanation-based filter that dismisses alerts with a legitimate clinical reason, and composite risk scoring (MEDIUM, HIGH, CRITICAL).
- **Containerised test environment**: a three-service Docker Compose stack (OpenEMR, MariaDB, phpMyAdmin) started with a single command.
- **Data pipeline**: imports Synthea patient records into OpenEMR, then generates 1,250 labelled audit events with ground truth embedded, extracted for evaluation with SQL.
- **Iterative development**: 12 versions, improving F1 from 69.58% (V1) to 96.54% (V12).
- **Debugging**: found and fixed a simulation bug in V8 where cross-department events were generated at night and double-triggered the after-hours rule (see `cross_department_v2.py`).
- **Evaluation**: benchmarked against five scikit-learn models with cross-validation, plus McNemar's significance test.
- **Reproducibility**: a seeded copy of the pipeline (`reproducible_demo_copy/`) produces identical results on every run.

**Tech stack:** Python, SQL, Docker Compose, MariaDB, OpenEMR, SQLAlchemy, pandas, scikit-learn

---

## What This Project Does

This is my final year project. I built a rule-based insider threat detection system on top of OpenEMR, an open-source EHR platform. The idea is to detect suspicious staff behaviour in hospital audit logs - things like accessing patient records outside working hours, browsing through patients alphabetically, or a receptionist pulling up clinical data they shouldn't need.

I also ran a comparison against 5 ML models (Random Forest, Gradient Boosting, Logistic Regression, SVM, Isolation Forest) to see how the rule-based approach holds up.

The final version (V12) achieved **98.56% accuracy** and **96.54% F1-score** on 1,250 labelled synthetic audit entries.

---

## Files

| File | What it does |
|------|-------------|
| `src/detection_engine_v12.py` | The main detection engine - runs 10 rules, filters out flags that have a clinical justification, assigns risk levels |
| `src/ml_baseline_comparison_v4.py` | Trains and evaluates 5 ML models on the same dataset for comparison |
| `src/simulate_access_patterns.py` | Generates the 1,250 labelled audit entries used for testing (all 10 threat patterns) |
| `src/mcnemars_test_v3.py` | McNemar's significance test comparing rule-based vs Gradient Boosting |
| `src/cross_department_v2.py` | Fix script - regenerates cross-department patterns with corrected timestamps (see note below) |
| `src/import_synthea_patients.py` | Imports Synthea C-CDA patient records into OpenEMR |
| `outputs/` | All generated CSVs, PNGs and JSON output |
| `docker-compose.yml` | Spins up OpenEMR + MariaDB + phpMyAdmin |
| `requirements.txt` | Python package dependencies |
| `README.md` | Setup and run instructions |

---

## How to Run

### Step 1 - Start OpenEMR

```bash
docker compose up -d
```

Give it about 60 seconds to fully start up on first run.

- OpenEMR: http://localhost:8080 (admin / pass)
- phpMyAdmin: http://localhost:8081 (root / root)
- MySQL: localhost:3306 (openemr / openemrpass)

These are default credentials for a local, throwaway development stack holding only synthetic data. Do not reuse them anywhere real.

### Step 2 - Install dependencies

```bash
pip3 install -r requirements.txt
```

### Step 3 - Generate Synthea patient data

The patient records come from [Synthea](https://github.com/synthetichealth/synthea) - you need to generate them yourself before running the import script.

```bash
git clone https://github.com/synthetichealth/synthea
cd synthea
./run_synthea -p 100
```

This creates C-CDA XML files in `synthea/output/ccda/`. Then open `src/import_synthea_patients.py` and update the `ccda_folder` path at the bottom to point to that folder.

### Step 4 - Import patients into OpenEMR

```bash
python3 src/import_synthea_patients.py
```

This pulls the patient demographics (names, postcodes, addresses) into OpenEMR's database - needed for the surname and geographic detection rules.

### Step 5 - Generate the audit data

```bash
python3 src/simulate_access_patterns.py
```

Populates the audit log with 1,250 labelled entries:

- 993 normal (legitimate clinical access)
- 257 malicious events across all 10 threat patterns

Each entry has the ground truth embedded as JSON in the comments field so the detection engine can be evaluated against it.

### Step 6 - Run the detection engine

```bash
python3 src/detection_engine_v12.py
```

Runs all 10 rules, applies the explanation-based filter, scores composite risk (CRITICAL/HIGH/MEDIUM), and prints accuracy/precision/recall/F1 against ground truth. Also saves results to `outputs/rule_based_results.json` - the ML script needs this.

### Step 7 - Run the ML comparison

```bash
python3 src/ml_baseline_comparison_v4.py
```

Run Step 6 first - this loads the rule-based results from `outputs/rule_based_results.json` automatically.

Outputs:
- `outputs/ml_comparison_results_v4.csv`
- `outputs/cross_validation_results.csv`
- `outputs/roc_curves_comparison.png`
- `outputs/feature_importance_comparison.png`

### Step 8 - Run McNemar's test

```bash
python3 src/mcnemars_test_v3.py
```

Compares rule-based vs Gradient Boosting on the 250-entry holdout test set using McNemar's test. Result: chi-squared = 2.7692, p = 0.0923 - not statistically significant.

---

## Detection Rules

All 10 rules are exercised in the final synthetic workflow.

| Rule | What it flags | Where it comes from |
|------|--------------|---------------------|
| After-Hours | Access before 6am or after 10pm | Hedda et al. (2017) |
| High-Volume | >10 distinct patients in one hour | Above clinical workflow norms |
| Sequential Browsing | Patients accessed A-Z alphabetically | Known snooping pattern |
| Same-Surname | 2+ same-surname patients within 45 min | Hedda et al. (1.54x expected rate) |
| Geographic Proximity | 5+ patients from same postcode in 2h | Hedda et al. (2.52x expected rate) |
| Street Match | 3+ patients from same street in 4h | Hedda et al. (4.11x expected rate) |
| Cross-Department | Receptionist accessing clinical records | RBAC violation |
| Extended Viewing | 5+ accesses to the same patient in 2h | Data harvesting indicator |
| Failed Access | 3+ failed/denied attempts | Possible privilege escalation |
| VIP Access | High-profile patient accessed without justification | Celebrity/executive snooping |

---

## Key Design Decisions in V12

**Explanation-based filtering** (from Hedda et al., 2017): before raising an alert, the engine checks whether the access had a legitimate clinical reason - an appointment within ±48h, an active order within ±72h, or clinical notes written within ±24h. If any of those exist, the flag is dismissed. Hedda et al. showed this reduces false positives by 16-44% on real data.

**Composite risk scoring**: if the same log entry is flagged by 2+ rules it gets bumped to HIGH risk, 3+ rules = CRITICAL. This helped cut noise from single-rule false positives.

**Role-specific thresholds**: the idea was to set different patient-per-hour limits per role (a nurse seeing 15/hr is fine, a receptionist doing the same isn't). I implemented it but had to disable it - the synthetic dataset is too small to calculate reliable baselines per role.

It took 12 versions to get here. V1 was F1 = 69.58%, V12 is 96.54%.

---

## Results

| Model | Type | Accuracy | Precision | Recall | F1 |
|-------|------|----------|-----------|--------|----|
| Rule-Based V12 | Rule-Based | 98.56% | 95.44% | 97.67% | 96.54% |
| Random Forest | Supervised | 99.20% | 100.00% | 96.08% | 98.00% |
| Gradient Boosting | Supervised | 98.80% | 98.00% | 96.08% | 97.03% |
| SVM (RBF) | Supervised | 98.00% | 97.92% | 92.16% | 94.95% |
| Logistic Regression | Supervised | 98.00% | 100.00% | 90.20% | 94.85% |
| Isolation Forest | Unsupervised | 82.00% | 55.77% | 56.86% | 56.31% |

Although Gradient Boosting achieved slightly higher performance on the paired 20% holdout set, McNemar's test found the difference not to be statistically significant (χ² = 2.7692, p = 0.0923). This suggests that, on the final synthetic dataset, the rule-based system performed comparably to the best supervised baseline while requiring no labelled training data and producing directly interpretable alerts.

---

## Dataset

- 1,250 labelled entries generated by `simulate_access_patterns.py`
- 257 malicious (20.6%) / 993 legitimate (79.4%)
- 80/20 train/test split: 1,000 training entries / 250 test entries
- Ground truth format: `SIMULATED|{"is_malicious": true, "threat_pattern": "after_hours", ...}`
- Patient data from Synthea (MITRE)

---

## Note on cross_department_v2.py

This script exists because of a bug I found in V8. The original simulation was generating cross-department events at night, which meant those entries were also triggering the after-hours rule. That inflated the false positive count and pulled accuracy down from ~97% to ~91%. This script deletes those entries and regenerates them with timestamps between 9am to 4pm so only the cross-department rule fires.

---

## Stopping the environment

```bash
docker compose down
```

To also wipe the data volumes:

```bash
docker compose down -v
```

---

## Submission Version Note

The results reported in this dissertation - 98.56% accuracy, 96.54% F1-score, 1,250 entries - came from the locked final run of the scripts in `src/`. Those outputs are stored in `outputs/` and should be treated as the authoritative results.

The simulation in `src/simulate_access_patterns.py` is stochastic (no fixed seed, timestamps relative to the current date), so re-running it will produce slightly different event counts. This is by design - the evaluation was done on a fresh dataset rather than one tuned to a particular seed.

A supplementary `reproducible_demo_copy/` folder is included for demonstration purposes. It contains the same scripts but with `random.seed(42)` and a fixed anchor date (`datetime(2026, 1, 1)`) in the simulation, so it produces the same output on every run. See `reproducible_demo_copy/README_reproducible_demo.md` for details.

See also `REPRODUCIBILITY_NOTE.txt` in the project root.
