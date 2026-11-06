# 💧 AquaSense

**Smart water-leak detection, localization, and repair prioritization using EPANET simulation and machine learning.**

**VIT Hackathon · Track 3: Water Loss Detection**  
**Primary users:** Water-network operators

AquaSense is a research prototype that helps operators investigate **where water might be leaking** in a distribution network and **which location to inspect first**. It combines an unsupervised anomaly detector with physics-based hydraulic simulations to work with a limited number of pressure and flow sensors.

> **Important:** This is a **simulation-based hackathon prototype**, not a live monitoring system for Mauritius's Central Water Authority (CWA). The current performance figures come from synthetic EPANET scenarios, not real-world deployment.

## The problem

Water lost before it reaches customers affects supply reliability, treatment resources, and utility costs. Detecting abnormal pressure or flow is only the beginning: an operator also needs to **localize the possible leak**, despite noisy readings and limited sensor coverage, then prioritize investigation by the potential water that could be recovered.

Our solution addresses the hackathon task: **use flow, pressure, or consumption data to locate likely leaks and rank repairs by water saved.**

## What AquaSense does

- **Simulates a distribution network** from an EPANET `.inp` file using Python and WNTR.
- **Detects anomalies** with an Isolation Forest trained on leak-free pressure and flow readings.
- **Localizes suspected leaks** by comparing sensor readings with simulated hydraulic fingerprints for candidate junctions.
- **Optimizes virtual sensor placement** by comparing localization performance under different pressure/flow monitoring locations.
- **Ranks investigation candidates** using hydraulic fit and an estimate of potential recoverable water.
- **Visualizes results** in a Streamlit dashboard with a network diagram, measurements, rankings, and downloadable reports.

## How it works

```text
Leak-free EPANET network (.inp)
             |
             v
       WNTR / EPANET
             |
      +------+------+------------------+
      |             |                  |
      v             v                  v
Normal operating   Candidate leak   Observed scenario
simulations        simulations      (.inp or sensor CSV)
      |             |                  |
      v             v                  |
Isolation Forest   Hydraulic         |
(anomaly flag)     fingerprints      |
      |             |                  |
      +-------------+------------------+
                    |
                    v
           Hydraulic fit ranking
                    |
                    v
          Potential recovery estimate
                    |
                    v
       Investigation priority + report
```

**The two prediction stages are different:** Isolation Forest identifies unusual measurements **without leak labels**. Leak localization is **physics-based fingerprint matching**, not unsupervised learning. The localization algorithm sees selected sensor measurements, not the hidden emitter location used to create a test leak.

## Technology stack

| Component | Technology |
| --- | --- |
| Hydraulic modeling | [EPANET](https://www.epa.gov/water-research/epanet) |
| Python integration | [WNTR](https://usepa.github.io/WNTR/) |
| Anomaly detection | Scikit-learn Isolation Forest |
| Data processing | NumPy, Pandas |
| Network visualization | NetworkX, Plotly |
| User interface | Streamlit |
| Exports | CSV and HTML reports |

## Results so far

The following values were obtained in our **small, simulated five-junction network**. They do not represent field performance in Mauritius.

| Experiment | Result | Evaluation scope |
| --- | --- | --- |
| Anomaly detection, sensors J4/J6 + pipe 1 | **95% detection** | 38 of 40 simulated J4 leaks flagged |
| False alarms, same sensor setup | **5%** | 2 of 40 normal scenarios flagged |
| Leak localization, sensors J4/J6 + pipe 1 | **76% Top-1 accuracy** | 38 of 50 independent holdout leak scenarios |
| Leak localization, same setup | **98% Top-3 accuracy** | 49 of 50 independent holdout leak scenarios |
| Original sensor configuration J2/J6 + pipe 1 | **54% Top-1 accuracy** | 27 of the same 50 holdout scenarios |

In one simulated leak at junction 4, AquaSense ranked **J4 first** and estimated **297.4 m³/day** of potential recovery under a constant-leak assumption. This estimate is conditional on the model and should **not** be interpreted as confirmed real-world savings.

**Key finding:** Sensor placement can matter as much as the prediction algorithm. In our holdout experiment, moving one pressure sensor increased Top-1 localization from 54% to 76%, without increasing the number of sensors.

## Repository layout

```text
AquaSense/
├── README.md                           # Project overview (this file)
└── Prototype2/
    ├── AquaSense_Dashboard/            # Integrated Streamlit application
    │   ├── app.py
    │   ├── prioritize_repairs.py
    │   ├── requirements.txt
    │   ├── run_dashboard.bat
    │   ├── sample_network.inp
    │   ├── smoke_test.py
    │   └── README.md                    # Detailed dashboard instructions
    ├── network.inp                      # Leak-free experimental network
    ├── compare_leaks.py                # Baseline vs. leak simulation
    ├── train_anomaly_detector.py       # Isolation Forest experiments
    ├── localize_leak.py                # Junction localization experiments
    ├── optimize_sensors.py             # Sensor placement search
    ├── prioritize_repairs.py           # Repair-priority experiments
    └── ml_results/                     # Generated test output (if retained)
```

*The listing shows the intended core layout; EPANET may also generate `.bin`, `.rpt`, and temporary `.inp` files.*

## Quick start (Windows)

**Requirements:** Python 3.11–3.13 is recommended; an installed version of EPANET desktop is useful for editing networks but **not required** to launch the Python dashboard.

Open PowerShell in your cloned repository:

```powershell
cd Prototype2\AquaSense_Dashboard
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

The dashboard includes `sample_network.inp`. For your own model, copy a **leak-free** `network.inp` into `Prototype2/AquaSense_Dashboard/`, or select/upload a supported model using the app's network controls.

Optionally check the bundled demonstration network:

```powershell
python smoke_test.py
```

## Demo walkthrough

1. Choose the included sample network or your leak-free `network.inp`.
2. Select virtual pressure sensors at junctions **4** and **6**, and a flow sensor on pipe **1** (for our five-junction example).
3. Select **Simulate a hidden leak** and choose a candidate junction.
4. Select **Train Isolation Forest** for the current sensor configuration.
5. Select **Run leak analysis** and review the anomaly flag and ranked suspected locations.
6. Inspect estimated potential water recovery and download the report. Reveal the simulated leak location **after** inspecting the prediction.

**Alternatively, use an EPANET desktop scenario:** Add an emitter to a junction, then choose **File → Export → Network** in EPANET. Save the leaking scenario as `observed_leak.inp` without overwriting the leak-free model. Import that file into the dashboard using **Import EPANET leak scenario (.inp)**.

The application can also accept a sensor CSV with columns matching the selected sensors, for example:

```csv
pressure_4_m,pressure_6_m,flow_1_lps
47.424,49.648,20.354
```

*The row above illustrates the required format; it is not live telemetry.*

## Understanding repair priorities

The application evaluates possible locations as **alternative explanations for one suspected leak**. It estimates recoverable water based on a fitted emitter flow and a user-assumed recovery fraction. For a constant 24-hour leak:

```text
Potential recovery (m³/day) = estimated leak (L/s) × 86.4 × recovery fraction
```

A heuristic investigation index combines this estimate with a relative hydraulic-fit weight. **Fit weights are not calibrated leak probabilities**, and the index is **not** a validated economic benefit or an instruction to dispatch a repair crew. Do not add recovery estimates from competing candidate locations.

## Scope, limitations, and responsible use

- **Simulation only:** No live CWA telemetry or authoritative nationwide pipeline dataset is connected.
- **Current model:** Single leaks represented by junction emitters, one hydraulic observation snapshot, and a small demonstration network.
- **Sensor constraints:** Noise, demand uncertainty, and limited sensor coverage can make different leak locations difficult to distinguish.
- **Estimates, not guarantees:** Water-savings estimates depend on modeled leak rate, continuous operation, and assumed repair effectiveness; repair costs and feasibility are not yet modeled.
- **Human oversight:** Findings are leads for field verification, not automatic orders to excavate, shut valves, or interrupt supply.
- **Data stewardship:** Real utility network data and customer information should be used only with authorization and appropriate security controls.
- **Fairness:** Repair priorities should also consider service disruption, vulnerable communities, and equitable access—not only the highest modeled water volume.

## Roadmap

- [x] EPANET-to-Python hydraulic simulation
- [x] Unsupervised anomaly detection
- [x] Junction-level hydraulic localization
- [x] Sensor placement experiments
- [x] Investigation-priority estimates
- [x] Initial interactive Streamlit dashboard
- [ ] Broader testing on unseen leak sizes, multiple random seeds, and additional network topologies
- [ ] Pipe-segment localization and multiple simultaneous leaks
- [ ] Better uncertainty estimates and repair-cost considerations
- [ ] Validation against authorized measured data, if available

## References and acknowledgments

- [U.S. EPA — EPANET](https://www.epa.gov/water-research/epanet)
- [U.S. EPA — WNTR documentation](https://usepa.github.io/WNTR/)
- [Scikit-learn — Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

Developed for **VIT Hackathon — Track 3: Water Loss Detection** by a **two-member team**. The project is intended as an exploratory decision-support prototype for reducing water losses in Mauritius.
