# Reproducible Demo Copy

This folder is a supplementary copy of the project designed for deterministic, repeatable demonstrations. It is separate from the main `src/` folder, which contains the version used to produce the locked dissertation results.

---

## What is different here

The only file that differs from the main `src/` version is `simulate_access_patterns.py`. Two changes were made:

1. `random.seed(42)` added at the top - fixes the random number generator so every run produces the same sequence of events
2. `BASE_TIME = datetime(2026, 1, 1)` replaces all `datetime.now()` calls - anchors timestamps to a fixed date rather than the current system time

Everything else (detection engine, ML comparison, McNemar's test, import script) is an exact copy of the files in `src/`.

---

## Why the main version is stochastic

The simulation in `src/simulate_access_patterns.py` does not have a fixed seed. Each run generates slightly different events (different patient selections, different timestamps, small variation in counts). This was intentional - the dissertation evaluated the detection engine on a genuinely fresh dataset rather than one that was tuned to a fixed seed. The final locked run produced 1,250 entries (993 normal / 257 malicious) and those results are stored in `outputs/`.

Running the stochastic version again will produce slightly different numbers. That is expected and does not invalidate the dissertation results.

---

## How to run the demo

Follow the same steps as the main project. The only difference is you run the scripts from this folder instead of `src/`:

```bash
# Step 1 - start OpenEMR (same docker-compose.yml as the main project)
docker compose up -d

# Step 2 - install dependencies
pip3 install -r ../requirements.txt

# Step 3 - import Synthea patients (if not already done)
python3 src/import_synthea_patients.py

# Step 4 - generate audit data (deterministic version)
python3 src/simulate_access_patterns.py

# Step 5 - run the detection engine
python3 src/detection_engine_v12.py

# Step 6 - run ML comparison (needs detection engine output first)
python3 src/ml_baseline_comparison_v4.py

# Step 7 - run McNemar's test
python3 src/mcnemars_test_v3.py
```

Because the simulation is seeded, the event counts and timestamps will be identical on every machine and every run.

---

## Dissertation results vs demo results

The metrics in the dissertation (`outputs/`) came from the **main `src/` version** (stochastic, no fixed seed). Running this demo version will produce similar but not identical numbers - the patterns and rules are the same, just the random draws are fixed differently. The demo results should not be used to reproduce or challenge the reported dissertation figures.

| | Dissertation version | Demo version |
|---|---|---|
| Script | `src/simulate_access_patterns.py` | `reproducible_demo_copy/src/simulate_access_patterns.py` |
| Random seed | None (stochastic) | `random.seed(42)` |
| Time anchor | `datetime.now()` | `datetime(2026, 1, 1)` |
| Results | Locked in `outputs/` | Will vary slightly from dissertation |
