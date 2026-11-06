"""AquaSense Milestone 5: EPANET-backed, single-leak repair triage.

Run beside your LEAK-FREE network.inp:
    python prioritize_repairs.py --hidden-junction 4
    python prioritize_repairs.py --hidden-junction 5 --hidden-emitter 0.45
    python prioritize_repairs.py --observed-inp observed_leak.inp
    python prioritize_repairs.py --observed-csv sensor_readings.csv
    python prioritize_repairs.py --pressure 4 6 --flow 2 --hidden-junction 5

Dependencies: wntr numpy pandas; joblib optional for existing Isolation Forest.

Assumptions: ONE leak, at a JUNCTION, with fixed supply conditions at t=0.
All suspected sites refer to competing explanations of the SAME event; do not
sum estimated savings across suspects. Results are simulations, not field claims.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd
import wntr

DEFAULT_EMITTERS = (0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
DEFAULT_DEMANDS = (0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075)


def read_network(path):
    network = wntr.network.WaterNetworkModel(str(path))
    if not network.junction_name_list:
        raise ValueError("Network has no junctions")
    return network


def simulate(network_path, prefix, pressure_sensors, flow_sensors, *,
             clear_emitters, leak_node=None, emitter_epanet=0.0,
             demand_factor=1.0, local_demand_factors=None):
    """Return the selected hydraulic readings plus full nodal demands.

    Leak generator and predictor call the same hydraulic engine. The inference
    routine receives ONLY readings, never emitter locations/settings from the
    hidden or observed INP case.
    """
    model = read_network(network_path)
    model.options.time.duration = 0  # one steady hydraulic snapshot
    for node_name in model.junction_name_list:
        node = model.get_node(node_name)
        if clear_emitters:
            node.emitter_coefficient = 0.0
        for demand in node.demand_timeseries_list:
            demand.base_value *= demand_factor * (local_demand_factors or {}).get(node_name, 1.0)
    if leak_node is not None:
        if not clear_emitters:
            raise ValueError("Do not inject an emitter while retaining source emitters")
        # EPANET with LPS, exponent 0.5 -> WNTR SI m³/s/sqrt(m)
        model.get_node(leak_node).emitter_coefficient = emitter_epanet / 1000.0

    result = wntr.sim.EpanetSimulator(model).run_sim(
        file_prefix=str(prefix), convergence_error=True,
    )
    if getattr(result, "error_code", 0) not in (0, None):
        raise RuntimeError(f"EPANET returned error_code={result.error_code}")

    selected = []
    for node_name in pressure_sensors:
        selected.append(float(result.node["pressure"].loc[0, node_name]))
    for pipe_name in flow_sensors:
        selected.append(float(result.link["flowrate"].loc[0, pipe_name]) * 1000.0)
    observations = np.asarray(selected, dtype=float)
    if not np.all(np.isfinite(observations)):
        raise RuntimeError("Simulation produced non-finite sensor readings")

    demands = {node: float(result.node["demand"].loc[0, node]) * 1000.0
               for node in model.junction_name_list}
    return observations, demands


def build_library(network_path, prefix, junctions, pressure_sensors, flow_sensors):
    """Fit library includes a no-leak hypothesis and candidate leak severities."""
    library = []
    for factor in DEFAULT_DEMANDS:
        baseline_readings, baseline_demands = simulate(
            network_path, prefix, pressure_sensors, flow_sensors,
            clear_emitters=True, demand_factor=factor,
        )
        library.append({"junction": "NO LEAK", "emitter": 0.0,
                        "demand_factor": factor, "readings": baseline_readings,
                        "loss_lps": 0.0})
        for junction in junctions:
            for emitter in DEFAULT_EMITTERS:
                readings, demands = simulate(
                    network_path, prefix, pressure_sensors, flow_sensors,
                    clear_emitters=True, leak_node=junction,
                    emitter_epanet=emitter, demand_factor=factor,
                )
                # In a DD/DDA EPANET snapshot, emitter flow is included in
                # nodal demand. Subtract the same-node baseline at SAME factor.
                # For PDA/PDD, consumer demand may itself change with pressure;
                # this simple subtraction is not a valid isolated leak volume.
                leak_lps = max(0.0, demands[junction] - baseline_demands[junction])
                library.append({"junction": junction, "emitter": emitter,
                                "demand_factor": factor, "readings": readings,
                                "loss_lps": leak_lps})
    return library


def infer(readings, library, sigma, recovery_fraction):
    """Fit best signature per location; compute UNCAlibrated relative evidence.

    The 'priority index' is a triage heuristic: relative model fit weight
    multiplied by estimated potential recovery. It is NOT a calibrated
    probability-weighted savings forecast or an economic net benefit.
    """
    best = {}
    for s in library:
        residuals = (readings - s["readings"]) / sigma
        mismatch = float(np.sqrt(np.mean(residuals ** 2)))
        site = s["junction"]
        if site not in best or mismatch < best[site]["mismatch"]:
            best[site] = {**s, "mismatch": mismatch}

    candidates = sorted(best.values(), key=lambda s: s["mismatch"])
    # Gaussian-like illustrative support across candidate hypotheses;
    # nuisance-parameter fitting and uncertain demand mean NOT probabilities.
    scores = np.array([r["mismatch"] for r in candidates], dtype=float)
    logw = -0.5 * len(readings) * scores ** 2
    w = np.exp(logw - np.max(logw))
    w /= np.sum(w)
    rows = []
    for rank, (r, weight) in enumerate(zip(candidates, w), 1):
        recovered = r["loss_lps"] * 86.4 * recovery_fraction
        rows.append({
            "candidate_junction": r["junction"],
            "fit_rank": rank,
            "mismatch": round(r["mismatch"], 4),
            "relative_fit_weight_pct": round(float(weight) * 100, 2),
            "fitted_emitter_epanet": r["emitter"],
            "fitted_demand_multiplier": r["demand_factor"],
            "estimated_leak_lps": round(r["loss_lps"], 3),
            "potential_recovery_m3_day": round(recovered, 1),
            "triage_index": round(float(weight) * recovered, 2),
        })
    frame = pd.DataFrame(rows)
    leaking = (frame[frame["candidate_junction"] != "NO LEAK"]
               .sort_values(["triage_index", "mismatch"], ascending=[False, True])
               .copy())
    leaking.insert(0, "investigation_rank", range(1, len(leaking) + 1))
    return frame, leaking


def detector_status(observation, features, detector_path):
    if not detector_path.is_file():
        return "Not available (optional)", None
    try:
        import joblib
        payload = joblib.load(detector_path)  # ONLY trusted local model files
        model_features = list(payload["features"])
        if model_features != features:
            return "Skipped: detector trained on different sensor configuration", None
        data = pd.DataFrame([observation], columns=features)
        flag = int(payload["model"].predict(data)[0])
        score = float(payload["model"].decision_function(data)[0])
        return ("ANOMALY" if flag == -1 else "NORMAL"), score
    except Exception as exc:
        return f"Unavailable: {exc}", None


def make_report(target, features, observed, all_candidates, priority, source,
                detector_text, detector_score, recovery_fraction):
    def esc(v):
        return html.escape(str(v), quote=True)

    top_fit = all_candidates.iloc[0]
    top_priority = priority.iloc[0] if len(priority) else None
    row_html = []
    for _, rec in priority.iterrows():
        weight = float(rec["relative_fit_weight_pct"])
        row_html.append(
            "<tr>"
            f"<td>{int(rec['investigation_rank'])}</td>"
            f"<td>J{esc(rec['candidate_junction'])}</td>"
            f"<td>{int(rec['fit_rank'])}</td>"
            f"<td>{rec['mismatch']:.3f}</td>"
            f"<td><div class='bar'><span style='width:{max(0,min(100,weight)):.2f}%'></span></div>{weight:.1f}%</td>"
            f"<td>{rec['potential_recovery_m3_day']:.1f}</td>"
            f"<td>{rec['triage_index']:.1f}</td>"
            "</tr>"
        )
    measurements = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:.3f}</td></tr>" for k, v in zip(features, observed)
    )
    top_label = "No leak" if top_fit["candidate_junction"] == "NO LEAK" else f"J{top_fit['candidate_junction']}"
    suspect_label = ("J" + str(top_priority["candidate_junction"])) if top_priority is not None else "—"
    warning = ("No-leak simulation is the best hydraulic fit; DO NOT dispatch a repair based on this ranking alone."
               if top_fit["candidate_junction"] == "NO LEAK" else
               "These sites are ALTERNATIVE explanations for ONE possible leak. Do not add their water-saving estimates.")
    score_text = "" if detector_score is None else f" (score {detector_score:+.3f})"
    html_doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AquaSense — Milestone 5 report</title>
<style>
:root{{--ink:#15283a;--muted:#596b7a;--blue:#1865ae;--line:#d9e2eb;--bg:#f3f6fa}}
*{{box-sizing:border-box}}body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink);margin:0}}
main{{max-width:1070px;margin:auto;padding:36px 22px 75px}}
header{{display:flex;justify-content:space-between;align-items:end;gap:20px;flex-wrap:wrap}}
h1{{font-size:34px;letter-spacing:-1.2px;margin:2px 0}}h2{{font-size:21px;margin:30px 0 13px}}
p{{color:var(--muted)}}small{{color:var(--muted)}}
.panel{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;overflow:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:20px}}
.value{{font-size:29px;font-weight:750;letter-spacing:-.5px}}.metric-label{{font-size:12px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:12px}}
tr:last-child td{{border-bottom:0}}.bar{{width:80px;height:6px;background:#e7f0f8;border-radius:8px;display:inline-block;margin-right:9px;vertical-align:middle}}
.bar span{{display:block;height:100%;background:#3184cc;border-radius:8px}}.warn{{padding:14px 18px;background:#fff4d9;border:1px solid #f0d69a;border-radius:12px;color:#69511b}}
.tag{{color:#1865ae;background:#e6f2fd;border-radius:999px;padding:7px 13px;font-size:12px;font-weight:700}}
</style></head><body><main>
<header><div><span class='tag'>AquaSense · Hackathon prototype</span><h1>Repair investigation priorities</h1><small>{esc(source)}</small></div><small>One simulated network · one possible leak</small></header>
<div class='cards'>
<div class='panel'><div class='metric-label'>Best hydraulic-fit hypothesis</div><div class='value'>{esc(top_label)}</div></div>
<div class='panel'><div class='metric-label'>Top investigation-priority site</div><div class='value'>{esc(suspect_label)}</div></div>
<div class='panel'><div class='metric-label'>Anomaly detector</div><div class='value' style='font-size:18px'>{esc(detector_text)}</div><small>{esc(score_text)}</small></div>
<div class='panel'><div class='metric-label'>Assumed recovery fraction if repaired</div><div class='value'>{100*recovery_fraction:.0f}%</div></div>
</div>
<h2>Investigation queue</h2><div class='panel'><table><thead><tr><th>Priority</th><th>Candidate</th><th>Hydraulic fit rank</th><th>Mismatch</th><th>Relative fit weight*</th><th>Potential m³/day</th><th>Triage index*</th></tr></thead>
<tbody>{''.join(row_html)}</tbody></table></div>
<p class='warn'><b>Interpretation:</b> {esc(warning)} If the anomaly detector reports NORMAL, investigate the no-leak explanation first. Leakage values assume a steady 24-hour event and are not verified field savings.</p>
<h2>Measurements used for inference</h2><div class='panel'><table><thead><tr><th>Sensor</th><th>Observed measurement</th></tr></thead><tbody>{measurements}</tbody></table></div>
<h2>Method and limitations</h2><div class='panel'>
<p>EPANET emitter scenarios are matched against sparse pressure and pipe-flow readings while varying overall consumer demand. Potential recovery = fitted emitter outflow × 86.4 m³/day per L/s × recovery fraction.</p>
<p>*Relative fit weights are a Gaussian-like softmax of hydraulic mismatch scores. They are <b>not calibrated probabilities</b>. The triage index is a <b>heuristic</b> (weight × possible water recovery), not expected actual savings, repair cost, or economic return.</p>
<p>Assumes one leak at a junction in a demand-driven, steady-state model. No uncertainty for pipe roughness, unmodelled demands, multiple leaks, repairs, or temporal pressure variation. Candidate sites are not confirmed leaks. Not suitable for directing real-world repair work without verification.</p>
</div></main></body></html>"""
    target.write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", type=Path, default=Path("network.inp"))
    parser.add_argument("--pressure", nargs="+", default=["4", "6"], help="Pressure sensor junction IDs")
    parser.add_argument("--flow", nargs="+", default=["1"], help="Flow sensor pipe IDs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hidden-junction", help="TEST ONLY: inject a hidden synthetic junction leak")
    source.add_argument("--observed-inp", type=Path, help="INP with manually configured leak (measurements only read)")
    source.add_argument("--observed-csv", type=Path, help="CSV containing one row with the sensor feature columns")
    source.add_argument("--no-leak", action="store_true", help="TEST ONLY: generate a normal scenario")
    parser.add_argument("--hidden-emitter", type=float, default=0.46, help="Hidden leak EPANET emitter coefficient (LPS convention)")
    parser.add_argument("--demand-factor", type=float, default=1.013, help="Hidden test demand scaling")
    parser.add_argument("--no-noise", action="store_true", help="Disable the simulated sensor noise on test observations")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pressure-sigma", type=float, default=0.08, help="Pressure measurement SD (m); assumed for fitting")
    parser.add_argument("--flow-sigma", type=float, default=0.08, help="Flow measurement SD (L/s); assumed for fitting")
    parser.add_argument("--recovery-fraction", type=float, default=1.0,
                        help="Fraction of simulated loss recovered if fixed, 0 to 1")
    parser.add_argument("--out", type=Path, default=Path("ml_results"))
    args = parser.parse_args()

    if not args.network.is_file():
        parser.error(f"Missing baseline network: {args.network}")
    if args.observed_inp is not None and not args.observed_inp.is_file():
        parser.error(f"Missing observed scenario: {args.observed_inp}")
    if args.observed_csv is not None and not args.observed_csv.is_file():
        parser.error(f"Missing sensor CSV: {args.observed_csv}")
    if not (0.0 <= args.recovery_fraction <= 1.0):
        parser.error("--recovery-fraction must be between 0 and 1")
    if args.pressure_sigma <= 0 or args.flow_sigma <= 0 or args.hidden_emitter <= 0 or args.demand_factor <= 0:
        parser.error("Noise sigma, emitter coefficient and demand factor must be positive")
    if len(set(args.pressure)) != len(args.pressure) or len(set(args.flow)) != len(args.flow):
        parser.error("Sensor IDs must not be duplicated")

    network = read_network(args.network)
    junctions = list(network.junction_name_list)
    pipes = list(network.pipe_name_list)
    if not set(args.pressure).issubset(junctions):
        parser.error(f"Invalid pressure sensor; valid junctions: {junctions}")
    if not set(args.flow).issubset(pipes):
        parser.error(f"Invalid flow sensor; valid pipes: {pipes}")
    hidden_node = args.hidden_junction or "4"
    if args.observed_inp is None and args.observed_csv is None and not args.no_leak and hidden_node not in junctions:
        parser.error(f"Unknown hidden junction {hidden_node}; choose from {junctions}")

    # Demand-driven assumptions are needed for (leaking demand - normal demand)
    demand_model = str(network.options.hydraulic.demand_model).upper()
    if demand_model not in ("DD", "DDA"):
        parser.error(f"Water-saving calculation currently supports demand-driven networks only (found {demand_model}).")
    file_units = str(network.options.hydraulic.inpfile_units).upper()
    if file_units != "LPS":
        parser.error(f"This prototype assumes LPS input-flow units (found {file_units}). Set EPANET Flow Units to LPS first.")
    emitter_exponent = float(network.options.hydraulic.emitter_exponent)
    if not math.isclose(emitter_exponent, 0.5, abs_tol=1e-8):
        parser.error(f"This prototype expects emitter exponent 0.5; found {emitter_exponent}.")

    args.out.mkdir(exist_ok=True, parents=True)
    prefix = args.out / "priority_epanet_run"
    features = ([f"pressure_{n}_m" for n in args.pressure]
                + [f"flow_{p}_lps" for p in args.flow])
    sigma = np.array([args.pressure_sigma] * len(args.pressure)
                     + [args.flow_sigma] * len(args.flow), dtype=float)

    print(f"Sensors: {', '.join(features)}")
    print(f"Building signature library ({len(junctions)} candidate junctions)...")
    library = build_library(args.network, prefix, junctions, args.pressure, args.flow)
    print(f"Generated {len(library)} EPANET scenarios")

    if args.observed_csv is not None:
        observations = pd.read_csv(args.observed_csv)
        if observations.empty or not set(features).issubset(observations.columns):
            parser.error(f"CSV needs columns: {features}")
        readings = observations.loc[0, features].to_numpy(dtype=float)
        label = f"Observed measurements from {args.observed_csv} (first row)"
    elif args.observed_inp is not None:
        # The source INP may have an emitter. Only its numeric readings cross
        # into infer() — the leak metadata is never inspected there.
        readings, _ = simulate(args.observed_inp, prefix, args.pressure, args.flow,
                               clear_emitters=False)
        label = f"Observed EPANET readings from {args.observed_inp}"
    else:
        rng = np.random.default_rng(args.seed)
        local = {j: float(rng.uniform(0.98, 1.02)) for j in junctions}
        readings, _ = simulate(
            args.network, prefix, args.pressure, args.flow,
            clear_emitters=True,
            leak_node=None if args.no_leak else hidden_node,
            emitter_epanet=0.0 if args.no_leak else args.hidden_emitter,
            demand_factor=args.demand_factor,
            local_demand_factors=local,
        )
        label = "Generated hidden synthetic scenario with local demand variation"
        if not args.no_noise:
            readings += rng.normal(0.0, sigma)

    if not np.all(np.isfinite(readings)):
        raise ValueError("Observation contains invalid/non-finite measurements")
    all_candidates, priority = infer(readings, library, sigma, args.recovery_fraction)
    detector_text, detector_score = detector_status(
        readings, features, args.out / "detector.joblib",
    )
    print("\n--- OBSERVED MEASUREMENTS ---")
    for name, value in zip(features, readings):
        print(f"{name:>18}: {value:9.3f}")
    print("\nIsolation Forest:", detector_text,
          "" if detector_score is None else f"(score {detector_score:+.4f})")

    print("\n--- HYDRAULIC FIT (lower mismatch = better) ---")
    print(all_candidates[["candidate_junction", "fit_rank", "mismatch",
                          "relative_fit_weight_pct", "potential_recovery_m3_day"]]
          .to_string(index=False))
    print("\n--- REPAIR INVESTIGATION PRIORITY (not confirmed leaks) ---")
    print(priority[["investigation_rank", "candidate_junction", "fit_rank",
                    "potential_recovery_m3_day", "triage_index"]]
          .to_string(index=False))
    best = str(all_candidates.iloc[0]["candidate_junction"])
    if best == "NO LEAK":
        print("\nWARNING: NO LEAK fits best. Do not dispatch repairs from this queue.")
    elif detector_text == "NORMAL":
        print("\nWARNING: Anomaly detector says NORMAL. Investigate demand/sensor error before repairs.")
    print("CAUTION: Candidate locations are alternatives for ONE possible leak.")
    print("Potential recovery assumes a constant leak for 24h; DO NOT sum candidate volumes.")
    print("Relative fit weights are NOT calibrated probabilities; triage index is heuristic.")

    if args.observed_csv is None and args.observed_inp is None:
        print("\nTEST ONLY — hidden truth revealed AFTER inference:",
              "NO LEAK" if args.no_leak else f"junction {hidden_node}")
        if not args.no_leak:
            print("True leak fit rank:",
                  int(all_candidates.loc[all_candidates["candidate_junction"] == hidden_node,
                                         "fit_rank"].iloc[0]))

    all_candidates.to_csv(args.out / "m5_hydraulic_fit.csv", index=False)
    priority.to_csv(args.out / "m5_repair_priority.csv", index=False)
    pd.DataFrame([dict(zip(features, readings))]).to_csv(
        args.out / "m5_sensor_readings.csv", index=False)
    make_report(args.out / "m5_report.html", features, readings,
                all_candidates, priority, label, detector_text, detector_score,
                args.recovery_fraction)
    print("\nSaved:")
    for f in ["m5_hydraulic_fit.csv", "m5_repair_priority.csv",
              "m5_sensor_readings.csv", "m5_report.html"]:
        print(" -", (args.out / f).resolve())


if __name__ == "__main__":
    main()
