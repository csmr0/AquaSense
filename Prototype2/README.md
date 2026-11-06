# AquaSense — Milestone 5: Repair prioritization

This milestone adds a **water-saving investigation priority list** on top of the EPANET-based leak localization you have already built.

**Keep using the original leak-free `network.inp`.** The script generates EPANET leak scenarios in Python, compares them with unseen sensor observations, estimates emitter leakage and produces a ranked shortlist.

## Folder layout

```text
D:\VIT Hackathon\Prototype2\
  network.inp                         <-- your original leak-free network
  train_anomaly_detector.py           <-- existing milestone 2
  localize_leak.py                     <-- existing milestone 3
  optimize_sensors.py                 <-- existing milestone 4
  prioritize_repairs.py               <-- NEW (copy from this ZIP)
  ml_results\                         <-- output created automatically
```

## Install and run in PowerShell

```powershell
python -m pip install wntr numpy pandas joblib scikit-learn
python prioritize_repairs.py --hidden-junction 4
```

Test another location or severity (hidden test scenarios use out-of-grid defaults, local demand changes and simulated sensor noise):

```powershell
python prioritize_repairs.py --hidden-junction 5 --hidden-emitter 0.45
python prioritize_repairs.py --hidden-junction 4 --no-noise
```

Test with a leak manually inserted in EPANET:

1. Open your network in the EPANET desktop app.
2. Change a junction **Emitter Coeff.** from 0 to, say, 0.5.
3. Choose **File → Export → Network** and save the leaking scenario as `observed_leak.inp` **without overwriting** the leak-free `network.inp`.
4. Run:

```powershell
python prioritize_repairs.py --observed-inp observed_leak.inp
```

The observed `.inp` is **only used to obtain pressure/flow measurements**. The fitting algorithm never receives the emitter settings or the junction where you placed the leak.

Or reuse measurements from a sensor CSV with the following headers:

```csv
pressure_4_m,pressure_6_m,flow_1_lps
47.0,49.8,20.0
```

Then:

```powershell
python prioritize_repairs.py --observed-csv my_sensor_readings.csv
```

**Do not use invented sensor readings as experimental evidence.** The row above is only an example of CSV formatting.

## Sensor selection

By default, the pressure sensors are junctions `4` and `6` and the flow sensor is pipe `1`. This is your provisional placement based on the earlier 50-scenario holdout.

To test the alternate configuration with a flow sensor on pipe 2:

```powershell
python prioritize_repairs.py --pressure 4 6 --flow 2 --hidden-junction 4
```

If the unsupervised detector was trained with different sensor IDs, the script **skips it** rather than silently using incompatible features. To include Isolation Forest detection here, set `PRESSURE_SENSORS = ["4", "6"]` and `FLOW_SENSORS = ["1"]` in `train_anomaly_detector.py`, retrain it, then run this module. Localization and repair ranking continue to work without a saved detector.

## Files generated

- `ml_results/m5_hydraulic_fit.csv` — candidates ordered by hydraulic mismatch, including a **NO LEAK** hypothesis.
- `ml_results/m5_repair_priority.csv` — alternative junction locations ranked by a **heuristic** combining relative match strength and water potentially saved if repaired.
- `ml_results/m5_sensor_readings.csv` — sensor readings used for inference.
- `ml_results/m5_report.html` — standalone report; double-click to open in the browser.

## How it works

For each possible leaking junction and a grid of emitter coefficients and demand levels, the script runs EPANET through WNTR and calculates a sensor fingerprint. It then finds the lowest normalized mismatch for each site.

- `Estimated leak (L/s)`: additional EPANET emitter outflow, taken from the difference in nodal demand relative to the matched no-leak scenario (demand-driven only).
- `Potential recovery (m³/day)`: estimated leak (L/s) × 86.4 × assumed recovery fraction; this **assumes 24-hour constant flow**. Change the fraction using `--recovery-fraction 0.9` if you want to examine a hypothetical 90% repair effectiveness.
- `Relative fit weight (%)`: **uncalibrated** softmax weighting of mismatch scores; **not a probability** that a location truly leaks.
- `Triage index`: relative fit weight (0..1) × potential recovery. It is only a prioritization heuristic, **not a validated expected savings estimate** and does not account for repair cost.

### Important interpretation

**Only ONE leak is assumed.** Every candidate is a different *possible* location of that one leak. Never add up the estimated losses from all candidates and call the total water saved. A field operator would investigate the highest-priority candidate and use verification to narrow the search.

A NO LEAK best fit or a NORMAL Isolation Forest status should prevent an automatic repair decision.

### Limitations

The script requires **LPS flow units**, **emitter exponent 0.5**, and **DD/DDA demand-driven mode** in your EPANET network.

This is a single-snapshot, single-leak, junction-only model. Real pipe leaks may occur anywhere along pipes. Demand variations, sensor noise, uncertain pipe roughness, intermittent supply, pressure-dependent consumption, and multiple simultaneous leaks require more complex models. Current outputs are **synthetic**, not operational findings about Mauritius. Use `--help` for all options.

A complete hackathon evaluation should test this repair ranking on held-out scenarios with different leak sizes, sensor noise, multiple random seeds, and ideally a known cost/repair-time assumption per location. The previous sensor-placement accuracy figures should not be presented as the accuracy of this new repair module.
