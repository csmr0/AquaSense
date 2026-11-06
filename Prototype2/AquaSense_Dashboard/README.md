# AquaSense — Interactive Hackathon Dashboard (Milestone 6)

A local Windows-friendly **EPANET 2.2 / WNTR + Streamlit** web dashboard for water-leak **anomaly detection**, **junction-level localization**, and **repair investigation prioritization**. This package builds on your earlier `prioritize_repairs.py` implementation.

> **Research prototype, not a CWA live system.** It uses either a simulated network or a locally exported EPANET `.inp` network. It does not connect to the open EPANET desktop window or to live water-utility telemetry.

## 1 — Installation (Windows PowerShell)

Extract the ZIP to a folder, e.g. `D:\VIT Hackathon\AquaSense_Dashboard`.

If you have a **leak-free** `network.inp` in your existing `Prototype2` folder, **copy** it into this new dashboard folder. **Never overwrite it with the leaking network.** The dashboard also includes `sample_network.inp` for trying the interface without your file.

From the new folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Alternatively, double-click `run_dashboard.bat` after installing dependencies. Streamlit should open `http://localhost:8501` in a browser (the port may be different if in use).

**Python:** use a version supported by your installed WNTR package, preferably Python 3.11–3.13. Your earlier WNTR environment may already work.

## 2 — First demonstration with the five-junction network

1. In **01 / Network**, choose `network.inp` if you copied your working leak-free file, or **Included demonstration network**.
2. Under **02 / Virtual sensors**, use pressure junctions **4** and **6**, plus flow pipe **1**.
3. Keep all five candidate junctions selected.
4. Under **Observation source**, choose **Simulate a hidden leak**. Select junction **4**, emitter coefficient **0.46**, and optionally add sensor noise.
5. Click **Train Isolation Forest**. This trains the anomaly detector using *only leak-free* simulation outputs. (You can skip this step and still perform hydraulic localization.)
6. Click **Run leak analysis**. The first run builds the hydraulic signature library; it is cached for subsequent cases using the same network, sensors, and candidate junctions.
7. See the network visualization, best hydraulic-fit candidate, anomaly alert (if trained), potential water savings, and investigation priority table. Expand **Reveal hidden test location** only after inspecting the prediction.
8. Download the HTML report or CSV files from **Export report**.

**Note:** The run of 5 candidate junctions generates approximately 287 EPANET cases; training Isolation Forest adds another 240 independent simulations. Larger networks take longer.

## 3 — Use your leak created in EPANET desktop

1. Open your normal EPANET network and add an emitter coefficient at one junction.
2. In EPANET, choose **File → Export → Network** and save it as **`observed_leak.inp`**, separately from the normal `network.inp`.
3. Put `observed_leak.inp` in this dashboard folder, or choose **Import EPANET leak scenario (.inp)** and upload the file there.
4. Click **Run leak analysis**. The dashboard simulates the imported scenario to obtain **only selected sensor pressure and flow measurements**. The localization algorithm uses the **leak-free reference** and its own signature library. It never uses emitter ID or leak location metadata as an input feature.

The observed scenario should contain the same pipes, junctions, and supply setup as the baseline, apart from the leak and any deliberately tested demand changes. This prototype checks node/link IDs but cannot detect every other hydraulic model difference.

## 4 — Import readings without an EPANET leak file

Choose **Import sensor readings (.csv)**. The CSV header must match your chosen pressure/flow sensor IDs, e.g.:

```csv
pressure_4_m,pressure_6_m,flow_1_lps
47.424,49.648,20.354
```

This CSV contains **only example formatting**: replace the readings with real simulation outputs from your own test, or measured data if you have authorized, properly calibrated sensors.

## 5 — Detector, localization and repair priorities

- **Unsupervised anomaly detector:** Isolation Forest trained on leak-free EPANET runs with varied network-wide and local demand. It flags unusual *sensor patterns*. It does **not** directly predict a location.
- **Hydraulic localization:** compares the observation against simulated leak fingerprints across candidate junctions, emitter coefficients, and demand multipliers. Ranks candidates by normalized mismatch. This is **physics-based matching**, not unsupervised localization.
- **Repair priorities:** estimates a potential daily recovery using fitted emitter flow × 86.4 m³/day per L/s × user-assumed repair recovery fraction. Combines the estimate with an **uncalibrated** fit weight into a heuristic triage index.

**Important:** The candidate sites are competing explanations for **one hypothetical leak**. Do **not** add candidate recovery values. Triage indexes are not calibrated expected savings, nor verified repair recommendations.

The app includes a **NO LEAK** hypothesis. When that is the best hydraulic match, or when Isolation Forest reports **NORMAL**, investigate normal demand and sensor behavior rather than automatically dispatching a repair.

## 6 — Exported files

On successful run, these are written to `ml_results/`:

- `dashboard_hydraulic_fit.csv`
- `dashboard_repair_priority.csv`
- `dashboard_sensor_readings.csv`
- `dashboard_report.html`
- `dashboard_detector.joblib` after training within the app (not auto-loaded from uploads)

CSV and HTML files can also be downloaded directly from the application.

## 7 — Optional smoke test

```powershell
python smoke_test.py
```

This tests that the **included sample** can run through the EPANET engine, a simulated leak increases inlet flow, the exact J4 leak fingerprint identifies J4, and the reference case ranks `NO LEAK` first. It does *not* validate performance on unseen scenarios.

## Known limitations

- Supports **LPS** EPANET files, **demand-driven** hydraulics (`DDA`/`DD`), and emitter exponent **0.5** for volume estimates. The dashboard validates these assumptions.
- Junction emitters approximate leaks; **not arbitrary real pipe-break positions**. One leak and one steady-state hydraulic snapshot only.
- Network diagrams reflect topology and any imported node coordinates, not a verified map of Mauritius infrastructure.
- This version does not infer pipe-roughness uncertainty, sensor bias, repair costs, intermittent supply, multiple leaks, calibration with real pressure logger data, or live SCADA integration.
- For large files, shortlist candidate junctions (20 maximum in this dashboard) or partition the study area.
- Approximate false-alarm / detection percentages are based on synthetic holds from the *same* hydraulic model and must not be described as field accuracy.
- The app was syntax-checked in the packaging environment. It could not be executed end-to-end there because WNTR and Streamlit packages were not installed in that environment. **Use `smoke_test.py` on your Windows WNTR installation** and share any errors for corrections.

## Technical architecture

```text
EPANET .inp (reference)                  EPANET .inp or CSV (observations)
          |                                         |
          v                                         v
      WNTR / EPANET                         virtual sensors
          |                                         |
          +--> normal simulations --> Isolation Forest --> anomaly flag
          |
          +--> single-leak signatures --------+------> hydraulic matching
                                                       |
                                                       v
                                              alternative leak sites
                                                       |
                                                       v
                                              recovery / triage report
```

**Core code:** `app.py` (Streamlit UI + training and integration), `prioritize_repairs.py` (tested Milestone 5 hydraulic core), `sample_network.inp` (educational sample).
