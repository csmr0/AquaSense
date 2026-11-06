"""AquaSense: interactive EPANET + unsupervised detection + hydraulic localization.

Start from this folder: streamlit run app.py
This is a simulation-backed hackathon research prototype, NOT field leak detection.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import wntr
from sklearn.ensemble import IsolationForest

from prioritize_repairs import (
    DEFAULT_DEMANDS, DEFAULT_EMITTERS,
    build_library, infer, make_report, read_network, simulate,
)

ROOT = Path(__file__).resolve().parent
LOCAL_NETWORK = ROOT / "network.inp"
SAMPLE_NETWORK = ROOT / "sample_network.inp"
RESULTS_DIR = ROOT / "ml_results"

st.set_page_config(page_title="AquaSense | Water Loss Intelligence", page_icon="💧", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1450px}
[data-testid="stMetric"]{background:rgba(120,155,198,.06);border:1px solid rgba(120,155,198,.19);border-radius:13px;padding:14px}
.hero{background:linear-gradient(110deg,#0d253a,#153b55 70%,#126c79);color:white;padding:25px 28px;border-radius:18px;margin-bottom:15px}
.hero h1{color:white;margin:0 0 3px;font-size:32px}
.hero p{color:#d2e4ef;margin:0}
.small-note{opacity:.83;font-size:.87rem}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>💧 AquaSense</h1><p>Water loss intelligence · EPANET simulation · anomaly detection · leak localization</p></div>', unsafe_allow_html=True)
st.caption("HACKATHON PROTOTYPE · Results are simulated or derived from imported EPANET scenarios; not live readings from Mauritius CWA.")


def network_bytes_source():
    choices = ["Included demonstration network"]
    if LOCAL_NETWORK.exists():
        choices.insert(0, "Existing network.inp in this folder")
    choices.append("Upload an EPANET .inp")
    mode = st.sidebar.radio("Leak-free reference network", choices)
    if mode == "Upload an EPANET .inp":
        item = st.sidebar.file_uploader("Upload leak-free .inp", type=["inp"], key="baseline")
        if item is None:
            return None, "Upload your leak-free EPANET network."
        return item.getvalue(), f"Uploaded: {item.name}"
    path = LOCAL_NETWORK if mode.startswith("Existing") else SAMPLE_NETWORK
    return path.read_bytes(), path.name


def parse_network(raw: bytes):
    with tempfile.TemporaryDirectory(prefix="aqua_read_") as td:
        path = Path(td) / "baseline.inp"
        path.write_bytes(raw)
        return read_network(path)


def validate_model(model):
    units = str(model.options.hydraulic.inpfile_units).upper()
    demand = str(model.options.hydraulic.demand_model).upper()
    exponent = float(model.options.hydraulic.emitter_exponent)
    problems = []
    if units != "LPS":
        problems.append(f"Expected EPANET flow units LPS, found {units}.")
    if demand not in ("DDA", "DD"):
        problems.append(f"Expected demand-driven hydraulics (DDA), found {demand}.")
    if not math.isclose(exponent, 0.5, abs_tol=1e-8):
        problems.append(f"Expected emitter exponent 0.5, found {exponent}.")
    if not model.junction_name_list or not model.pipe_name_list:
        problems.append("The network must have at least one junction and one pipe.")
    return problems


def features_for(pressure, flow):
    return ([f"pressure_{n}_m" for n in pressure]
            + [f"flow_{p}_lps" for p in flow])


@st.cache_data(show_spinner=False, max_entries=6)
def signature_library(raw: bytes, pressure: tuple, flow: tuple, candidates: tuple):
    # Cached by network content, sensor placement and candidate list.
    with tempfile.TemporaryDirectory(prefix="aqua_sign_") as td:
        td = Path(td)
        input_path = td / "baseline.inp"
        input_path.write_bytes(raw)
        return build_library(input_path, td / "hydraulic", list(candidates), list(pressure), list(flow))


def measured_scenario(raw, pressure, flow, *, simulated_node=None,
                      emitter=0.46, demand_factor=1.013, seed=2026,
                      add_noise=True, pressure_sigma=.08, flow_sigma=.08,
                      preserve_emitters=False):
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(prefix="aqua_obs_") as td:
        td = Path(td)
        path = td / "observation.inp"
        path.write_bytes(raw)
        local = None
        if not preserve_emitters:
            original = read_network(path)
            local = {j: float(rng.uniform(.98, 1.02)) for j in original.junction_name_list}
        readings, _ = simulate(
            path, td / "hydraulic", list(pressure), list(flow),
            clear_emitters=not preserve_emitters,
            leak_node=simulated_node,
            emitter_epanet=float(emitter if simulated_node else 0.0),
            demand_factor=float(demand_factor if not preserve_emitters else 1.0),
            local_demand_factors=local,
        )
    readings = readings.astype(float)
    if add_noise and not preserve_emitters:
        sigmas = np.array([pressure_sigma] * len(pressure) + [flow_sigma] * len(flow))
        readings += rng.normal(0.0, sigmas)
    return readings


def train_unsupervised(raw, pressure, flow, candidates, n_train=160, n_test=40, seed=42,
                       pressure_sigma=.08, flow_sigma=.08):
    # Generate ONLY normal training data, with varied overall/local demand.
    rng = np.random.default_rng(seed)
    columns = features_for(pressure, flow)
    def one_run(path, prefix, leak=None):
        wn = read_network(path)
        wn.options.time.duration = 0
        overall = rng.uniform(.90, 1.10)
        for name in wn.junction_name_list:
            obj = wn.get_node(name)
            obj.emitter_coefficient = 0.0
            factor = overall * rng.uniform(.95, 1.05)
            for demand in obj.demand_timeseries_list:
                demand.base_value *= factor
        if leak is not None:
            wn.get_node(leak).emitter_coefficient = float(rng.uniform(.25, .7)) / 1000.0
        result = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(prefix), convergence_error=True)
        values = [float(result.node["pressure"].loc[0, p]) for p in pressure]
        values += [float(result.link["flowrate"].loc[0, f]) * 1000 for f in flow]
        values = np.array(values, dtype=float)
        values += rng.normal(0, np.array([pressure_sigma] * len(pressure)
                                           + [flow_sigma] * len(flow)))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite sensor readings during training")
        return values
    with tempfile.TemporaryDirectory(prefix="aqua_train_") as td:
        td = Path(td)
        path = td / "baseline.inp"
        path.write_bytes(raw)
        normal = np.array([one_run(path, td / "engine") for _ in range(n_train)])
        model = IsolationForest(n_estimators=200, contamination=.03, random_state=seed)
        model.fit(pd.DataFrame(normal, columns=columns))
        holdout_normal = np.array([one_run(path, td / "engine") for _ in range(n_test)])
        holdout_leak = np.array([
            one_run(path, td / "engine", leak=candidates[int(rng.integers(len(candidates)))])
            for _ in range(n_test)
        ])
    p_normal = model.predict(pd.DataFrame(holdout_normal, columns=columns))
    p_leak = model.predict(pd.DataFrame(holdout_leak, columns=columns))
    return {"model": model, "features": columns,
            "false_alarm": float(np.mean(p_normal == -1)),
            "detection": float(np.mean(p_leak == -1)),
            "n_normal": n_test, "n_leak": n_test}


def network_positions(model):
    graph = nx.Graph()
    for name in model.node_name_list:
        graph.add_node(name)
    for name in model.link_name_list:
        edge = model.get_link(name)
        graph.add_edge(edge.start_node_name, edge.end_node_name)
    coords = {}
    for name in model.node_name_list:
        point = getattr(model.get_node(name), "coordinates", None)
        if point is not None and len(point) == 2:
            coords[name] = np.asarray(point, dtype=float)
    enough = len(coords) == len(model.node_name_list) and len(set(map(tuple, coords.values()))) > 1
    if not enough:
        coords = nx.spring_layout(graph, seed=21)
    return coords


def plot_network(model, *, pressure, best_candidate=None, alternative=None,
                 observed=None, features=None):
    positions = network_positions(model)
    fig = go.Figure()
    for link_name in model.link_name_list:
        link = model.get_link(link_name)
        a = positions[link.start_node_name]
        b = positions[link.end_node_name]
        waypoints = [a]
        for v in (getattr(link, "vertices", None) or []):
            waypoints.append(np.asarray(v, dtype=float))
        waypoints.append(b)
        fig.add_trace(go.Scatter(
            x=[float(pt[0]) for pt in waypoints], y=[float(pt[1]) for pt in waypoints],
            mode="lines", line=dict(color="#7993aa", width=4),
            name=f"Pipe {link_name}", showlegend=False,
            hovertemplate=f"Pipe {link_name}<br>{link.start_node_name} ↔ {link.end_node_name}<extra></extra>",
        ))
        center = (np.asarray(a) + np.asarray(b)) / 2
        fig.add_annotation(x=float(center[0]), y=float(center[1]), text=str(link_name),
                           font=dict(size=10, color="#8ba4bb"), showarrow=False, yshift=9)
    obs_lookup = dict(zip(features or [], observed if observed is not None else []))
    for name in model.node_name_list:
        obj = model.get_node(name)
        junction = name in model.junction_name_list
        active = name == best_candidate
        second = name == alternative
        sensor = name in pressure
        color = "#eb5757" if active else "#f0b44b" if second else ("#33c9d0" if sensor else "#6e95bf")
        symbol = "circle" if junction else "square"
        label = f"J{name}" if junction else f"Source {name}"
        pressure_val = obs_lookup.get(f"pressure_{name}_m")
        hover = label + (f"<br>Measured pressure: {pressure_val:.3f} m" if pressure_val is not None else "")
        if active:
            hover += "<br>Best hydraulic-fit candidate"
        fig.add_trace(go.Scatter(
            x=[float(positions[name][0])], y=[float(positions[name][1])],
            mode="markers+text", text=[label], textposition="top center",
            textfont=dict(size=12, color="#dcecf8"),
            marker=dict(size=22 if active else 16, color=color, symbol=symbol,
                        line=dict(color="#eaf7ff" if sensor else "#183448", width=2)),
            name=label, showlegend=False, hovertemplate=hover + "<extra></extra>",
        ))
    fig.update_layout(
        height=415, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0c1d2d",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="#17324a", font_color="white"),
    )
    return fig


with st.sidebar:
    st.header("01 / Network")
    baseline_raw, source_label = network_bytes_source()
if baseline_raw is None:
    st.info(source_label)
    st.stop()
try:
    reference = parse_network(baseline_raw)
except Exception as exc:
    st.error(f"Could not read the selected EPANET .inp: {exc}")
    st.stop()
problems = validate_model(reference)
if problems:
    st.error("This version expects a demand-driven LPS network with emitter exponent 0.5: " + " ".join(problems))
    st.stop()
all_junctions = list(reference.junction_name_list)
all_pipes = list(reference.pipe_name_list)
original_emitters = [j for j in all_junctions if getattr(reference.get_node(j), "emitter_coefficient", None)]
if original_emitters:
    st.warning("The selected reference file already has emitter(s) at " + ", ".join(original_emitters)
               + ". The signature generator removes them, but use a leak-free reference export for a clean experiment.")

with st.sidebar:
    st.caption(f"{len(all_junctions)} junctions · {len(all_pipes)} pipes · {len(reference.reservoir_name_list)} reservoirs")
    st.divider()
    st.header("02 / Virtual sensors")
    preferred_pressure = [p for p in ("4", "6") if p in all_junctions]
    pressure = st.multiselect("Junction pressure sensors", all_junctions,
                              default=preferred_pressure or all_junctions[:2])
    flow = st.multiselect("Pipe flow sensors", all_pipes,
                          default=["1"] if "1" in all_pipes else all_pipes[:1])
    if not pressure or not flow:
        st.warning("Select at least one pressure and one flow sensor.")
        st.stop()
    st.divider()
    st.header("03 / Leak candidates")
    candidates = st.multiselect("Candidate junctions to search", all_junctions,
                                default=all_junctions[:min(len(all_junctions), 15)])
    if not candidates:
        st.info("Choose at least one candidate location.")
        st.stop()
    n_sims = len(DEFAULT_DEMANDS) * (1 + len(DEFAULT_EMITTERS) * len(candidates))
    st.caption(f"Signature library: {n_sims:,} EPANET runs · cached after first build")
    if len(candidates) > 20:
        st.error("Select at most 20 candidate junctions for this hackathon prototype. Use a smaller study area on larger networks.")
        st.stop()

st.subheader("Simulation & analysis")
mode_col, param_col = st.columns([1.5, 1])
with mode_col:
    scenario_mode = st.radio("Observation source", [
        "Simulate a hidden leak", "Simulate normal operation",
        "Import EPANET leak scenario (.inp)", "Import sensor readings (.csv)"
    ], horizontal=False)
    scenario_raw = None
    scenario_csv = None
    hidden = None
    if scenario_mode == "Simulate a hidden leak":
        hidden = st.selectbox("Hidden leak junction (known only to simulator)", candidates,
                              index=candidates.index("4") if "4" in candidates else 0)
    elif scenario_mode == "Import EPANET leak scenario (.inp)":
        local_obs = ROOT / "observed_leak.inp"
        options = ["Upload an EPANET scenario"]
        if local_obs.exists():
            options.insert(0, "Use observed_leak.inp in project folder")
        selected = st.radio("Observation file", options, horizontal=True)
        if selected.startswith("Use "):
            scenario_raw = local_obs.read_bytes()
        else:
            f = st.file_uploader("Leaking EPANET network exported from desktop", type=["inp"], key="observation")
            if f is not None:
                scenario_raw = f.getvalue()
    elif scenario_mode == "Import sensor readings (.csv)":
        f = st.file_uploader("CSV with sensor feature headers", type=["csv"], key="sensor_csv")
        if f is not None:
            scenario_csv = f.getvalue()
    st.caption("In imported .inp mode, EPANET generates observations; only pressure and flow measurements reach the predictor.")

with param_col:
    emitter = 0.46
    demand_factor = 1.013
    add_noise = True
    if scenario_mode == "Simulate a hidden leak":
        emitter = st.slider("Hidden emitter coefficient (L/s/√m)", 0.10, 1.00, 0.46, 0.01)
        demand_factor = st.slider("Demand multiplier (scenario only)", 0.90, 1.10, 1.013, 0.001)
        add_noise = st.checkbox("Add sensor noise", value=True)
    elif scenario_mode == "Simulate normal operation":
        demand_factor = st.slider("Demand multiplier (scenario only)", 0.90, 1.10, 1.013, 0.001)
        add_noise = st.checkbox("Add sensor noise", value=True)
    with st.expander("Assumptions & settings", expanded=False):
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026, step=1)
        p_sigma = st.number_input("Pressure noise SD (m)", min_value=.005, max_value=2., value=.08, step=.01, format="%.3f")
        f_sigma = st.number_input("Flow noise SD (L/s)", min_value=.005, max_value=2., value=.08, step=.01, format="%.3f")
        recovery_fraction = st.slider("Assumed fraction recoverable after repair", 0., 1., 1., .05)

pressure = tuple(pressure)
flow = tuple(flow)
candidates = tuple(candidates)
features = features_for(pressure, flow)
network_hash = hashlib.sha256(baseline_raw).hexdigest()
model_key = json.dumps([network_hash, list(pressure), list(flow), list(candidates)], sort_keys=True)

st.divider()
st.subheader("Unsupervised detector")
left, right = st.columns([1, 2])
with left:
    do_train = st.button("Train Isolation Forest", type="secondary", use_container_width=True,
                         help="Trains on leak-free EPANET runs, then tests unseen normal and leak scenarios.")
with right:
    st.caption("Optional but recommended. Training generates 160 leak-free simulations and evaluates on 80 additional simulations. This detector only flags anomalies; hydraulic matching locates them.")
if do_train:
    try:
        with st.spinner("Training and evaluating on EPANET-generated data …"):
            training = train_unsupervised(baseline_raw, pressure, flow, candidates,
                                         seed=int(seed), pressure_sigma=p_sigma, flow_sigma=f_sigma)
        st.session_state["detector"] = training
        st.session_state["detector_key"] = model_key
        # Re-run the analysis after training so its anomaly status is up to date.
        st.session_state.pop("analysis", None)
        RESULTS_DIR.mkdir(exist_ok=True)
        # This is a locally generated model; the dashboard NEVER loads uploaded pickle files.
        import joblib
        joblib.dump({"model": training["model"], "features": features,
                     "network_sha256": network_hash}, RESULTS_DIR / "dashboard_detector.joblib")
    except Exception as exc:
        st.error(f"Detector training failed: {exc}")

trained = st.session_state.get("detector") if st.session_state.get("detector_key") == model_key else None
if trained is not None:
    a, b, c = st.columns(3)
    a.metric("Normal holdout false alarms", f"{trained['false_alarm']:.1%}")
    b.metric("Leak holdout detection", f"{trained['detection']:.1%}")
    c.metric("Sensors used", len(features))
    st.caption("Holdout evaluation generated from the same simulated network; not an estimate of detection accuracy on real Mauritius measurements.")
else:
    st.info("No compatible detector trained in this dashboard session. Hydraulic localization still works without it. For combined detection + localization, click **Train Isolation Forest**.")

config_fingerprint = hashlib.sha256(repr((network_hash, pressure, flow, candidates, scenario_mode,
    hidden, emitter, demand_factor, add_noise, seed, p_sigma, f_sigma, recovery_fraction)).encode()
    + (scenario_raw or b"") + (scenario_csv or b"")).hexdigest()

run = st.button("▶ Run leak analysis", type="primary", use_container_width=True)
if run:
    if scenario_mode == "Import EPANET leak scenario (.inp)" and scenario_raw is None:
        st.error("Choose or upload the observed EPANET leak scenario before running analysis.")
        st.stop()
    if scenario_mode == "Import sensor readings (.csv)" and scenario_csv is None:
        st.error("Upload a sensor CSV before running analysis.")
        st.stop()
    try:
        with st.spinner(f"Building / loading {n_sims:,} EPANET signatures and analyzing the observation …"):
            lib = signature_library(baseline_raw, pressure, flow, candidates)
            if scenario_mode == "Import sensor readings (.csv)":
                csv = pd.read_csv(io.BytesIO(scenario_csv))
                if csv.empty or not set(features).issubset(csv.columns):
                    raise ValueError(f"CSV needs a row with these column names: {', '.join(features)}")
                obs = csv.loc[0, features].to_numpy(dtype=float)
            elif scenario_mode == "Import EPANET leak scenario (.inp)":
                # Observe data only; do NOT infer from hidden emitters or leak metadata.
                observed_network = parse_network(scenario_raw)
                if set(observed_network.node_name_list) != set(reference.node_name_list) or set(observed_network.link_name_list) != set(reference.link_name_list):
                    raise ValueError("Observed scenario must contain the same node/link IDs as the baseline network.")
                obs = measured_scenario(scenario_raw, pressure, flow, preserve_emitters=True)
            else:
                obs = measured_scenario(baseline_raw, pressure, flow, simulated_node=hidden,
                    emitter=emitter, demand_factor=demand_factor, seed=int(seed),
                    add_noise=add_noise, pressure_sigma=p_sigma, flow_sigma=f_sigma)
            sigma = np.array([p_sigma] * len(pressure) + [f_sigma] * len(flow))
            if not np.isfinite(obs).all():
                raise ValueError("Sensor readings must all be finite numbers.")
            fits, priority = infer(obs, lib, sigma, recovery_fraction)
            prediction, anomaly_score = None, None
            if trained is not None:
                frame = pd.DataFrame([obs], columns=features)
                prediction = ("ANOMALY" if int(trained["model"].predict(frame)[0]) == -1 else "NORMAL")
                anomaly_score = float(trained["model"].decision_function(frame)[0])
            RESULTS_DIR.mkdir(exist_ok=True)
            fits.to_csv(RESULTS_DIR / "dashboard_hydraulic_fit.csv", index=False)
            priority.to_csv(RESULTS_DIR / "dashboard_repair_priority.csv", index=False)
            sensor_df = pd.DataFrame([dict(zip(features, obs))])
            sensor_df.to_csv(RESULTS_DIR / "dashboard_sensor_readings.csv", index=False)
            report_file = RESULTS_DIR / "dashboard_report.html"
            make_report(report_file, features, obs, fits, priority, scenario_mode,
                        prediction or "NOT TRAINED", anomaly_score, recovery_fraction)
            st.session_state["analysis"] = {
                "fits": fits, "priority": priority, "features": features, "obs": obs,
                "prediction": prediction, "anomaly_score": anomaly_score,
                "source": scenario_mode, "report": report_file.read_bytes(),
                "sensor_csv": sensor_df.to_csv(index=False).encode(),
                "fit_csv": fits.to_csv(index=False).encode(),
                "priority_csv": priority.to_csv(index=False).encode(),
                "hidden_truth": hidden if scenario_mode == "Simulate a hidden leak" else None,
                "config": config_fingerprint,
            }
        st.success("EPANET analysis completed. Results are shown below.")
    except Exception as exc:
        st.error(f"Analysis could not complete: {exc}")

analysis = st.session_state.get("analysis")
if analysis is not None and analysis["config"] != config_fingerprint:
    st.warning("Inputs have changed since the previous analysis. Click **Run leak analysis** to update the results.")
    analysis = None

if analysis is None:
    st.subheader("Network preview")
    st.plotly_chart(plot_network(reference, pressure=pressure), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption("Blue nodes are junctions; cyan nodes are selected pressure sensors. A source is represented by a square.")
    st.stop()

fits = analysis["fits"]
priority = analysis["priority"]
best = str(fits.iloc[0]["candidate_junction"])
secondary = str(fits.iloc[1]["candidate_junction"]) if len(fits) > 1 else None
leaking_fit = fits[fits["candidate_junction"] != "NO LEAK"]
recovery = float(priority.iloc[0]["potential_recovery_m3_day"]) if len(priority) else 0.
if best == "NO LEAK":
    st.info("The no-leak hypothesis is the closest match. No repair should be dispatched on these readings alone.")
elif analysis["prediction"] == "NORMAL":
    st.warning("The anomaly detector considers this observation normal even though a leak hypothesis fits. Check demand and sensor uncertainties before initiating investigation.")

st.subheader("Analysis outcome")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best hydraulic-fit candidate", "NO LEAK" if best == "NO LEAK" else f"J{best}")
c2.metric("Anomaly detector", analysis["prediction"] or "Not trained")
c3.metric("Top candidate's potential recovery", f"{recovery:,.1f} m³/day")
c4.metric("Observed flow sensor", f"{float(analysis['obs'][-1]):,.2f} L/s")
st.caption("Potential recovery is conditional on the candidate being the real leak and a constant 24-hour flow. It is not verified savings.")

tab_map, tab_rank, tab_data, tab_export = st.tabs([
    "Network map", "Localization & investigation", "Sensor measurements", "Export report"
])
with tab_map:
    st.plotly_chart(plot_network(reference, pressure=pressure,
        best_candidate=None if best == "NO LEAK" else best,
        alternative=None if secondary == "NO LEAK" else secondary,
        observed=analysis["obs"], features=features),
        use_container_width=True, config={"displayModeBar": False})
    st.caption("Red = best hydraulic fit; amber = second fit; cyan = pressure sensor. This is network topology, not a geographic map of Mauritius.")
    if analysis["hidden_truth"] is not None:
        with st.expander("Reveal hidden test location (only after prediction)"):
            truth = analysis["hidden_truth"]
            rank = int(fits.loc[fits["candidate_junction"] == truth, "fit_rank"].iloc[0])
            st.write(f"Injected leak was at **J{truth}**. Its hydraulic-fit rank: **#{rank}**.")
with tab_rank:
    st.write("**Hydraulic hypotheses** — one possible leak, competing locations")
    st.dataframe(fits[["candidate_junction", "fit_rank", "mismatch",
        "relative_fit_weight_pct", "fitted_emitter_epanet", "potential_recovery_m3_day"]],
        use_container_width=True, hide_index=True)
    st.write("**Investigation order** — heuristic prioritization")
    st.dataframe(priority[["investigation_rank", "candidate_junction", "fit_rank",
        "potential_recovery_m3_day", "triage_index"]],
        use_container_width=True, hide_index=True)
    if len(priority):
        fig = go.Figure(go.Bar(
            x=priority["candidate_junction"].map(lambda s: "J" + str(s)),
            y=priority["triage_index"], marker_color="#1c9eb6"))
        fig.update_layout(title="Investigation priority index (relative units)",
                          xaxis_title="Candidate junction", yaxis_title="Heuristic triage index",
                          margin=dict(l=5, r=5, t=45, b=5), height=280)
        st.plotly_chart(fig, use_container_width=True)
    st.info("Fit weights are **not calibrated leak probabilities**. Candidate volumes are **alternatives for one leak**, so never sum them. Repair cost and field verification are not included.")
with tab_data:
    st.dataframe(pd.DataFrame([dict(zip(analysis["features"], analysis["obs"]))]),
                 use_container_width=True, hide_index=True)
    st.caption("Pressure in metres of water head; flow in litres per second. Only selected sensor features were supplied to the inference algorithm.")
    st.code(",".join(analysis["features"]) + "\n" +
            ",".join(f"{v:.4f}" for v in analysis["obs"]), language="text")
with tab_export:
    st.write("**Download analysis artifacts**")
    d1, d2 = st.columns(2)
    d1.download_button("Download investigation report (.html)",
                       data=analysis["report"], file_name="aquasense_report.html",
                       mime="text/html", use_container_width=True)
    d2.download_button("Download repair priorities (.csv)",
                       data=analysis["priority_csv"], file_name="aquasense_repair_priorities.csv",
                       mime="text/csv", use_container_width=True)
    d3, d4 = st.columns(2)
    d3.download_button("Download hydraulic fits (.csv)", data=analysis["fit_csv"],
                       file_name="aquasense_hydraulic_fits.csv", mime="text/csv", use_container_width=True)
    d4.download_button("Download sensor readings (.csv)", data=analysis["sensor_csv"],
                       file_name="aquasense_sensor_readings.csv", mime="text/csv", use_container_width=True)
    st.caption("Reports are also saved locally in the `ml_results/` folder. On an imported network, no emitter location is read by the predictor.")

st.divider()
st.caption("Assumptions: demand-driven (DDA), LPS, emitter exponent 0.5, one leak at a junction, one steady-state hydraulic snapshot. No support yet for arbitrary pipe cracks, multiple leaks, live SCADA connections, or national network-wide validation.")
