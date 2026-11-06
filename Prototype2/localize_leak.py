"""AquaSense milestone 3: EPANET-based leak localization from sparse sensors.

Run from the directory containing the leak-free `network.inp`:

    python localize_leak.py
    python localize_leak.py --hidden-junction 5
    python localize_leak.py --observed-inp observed_leak.inp
    python localize_leak.py --evaluate

The simulation that PRODUCES an observation and the code that INFERS a leak
are separate. The inference routine receives ONLY the numeric sensor readings.
It never inspects emitter settings or the true leak location.

Method: fit pressure/flow measurements against hydraulic signatures simulated
for each candidate node, emitter coefficient, and overall demand multiplier.
This is a PHYSICS-BASED localization method that complements, but is not part
of, the unsupervised Isolation Forest anomaly detector.

Limitations: single leak, junction candidates only (not yet pipe-localization),
steady-state t=0, constant reservoir conditions, limited demand variability,
no sensor drift or pipe parameter uncertainty beyond measurement noise. A
small minimum mismatch is NOT a calibrated probability of correctness.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wntr

NETWORK_FILE = Path("network.inp")
RESULTS_DIR = Path("ml_results")
DETECTOR_FILE = RESULTS_DIR / "detector.joblib"

# Keep these identical to those in train_anomaly_detector.py.
PRESSURE_SENSORS = ["3", "5"]
FLOW_SENSORS = ["1"]
FEATURES = ([f"pressure_{n}_m" for n in PRESSURE_SENSORS]
            + [f"flow_{n}_lps" for n in FLOW_SENSORS])

# Actual expected measurement noise (standard deviation), in display units.
SIGMAS = np.array([0.08] * len(PRESSURE_SENSORS)
                  + [0.08] * len(FLOW_SENSORS), dtype=float)

# Emitter values in EPANET's LPS convention, converted to WNTR SI below.
# Expand these ranges if your leaks may be smaller/larger.
EMITTER_GRID_EPANET = (0.20, 0.35, 0.50, 0.65, 0.80)
DEMAND_GRID = (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10)

rng = np.random.default_rng(2026)


def run_hydraulics(path, *, clear_emitters, leak_node=None,
                   emitter_si=0.0, demand_multiplier=1.0,
                   local_demand_multipliers=None):
    """Return only three numeric sensor readings after an EPANET simulation.

    IMPORTANT: for --observed-inp, clear_emitters=False preserves the unknown
    scenario's emitter(s), but the inference function never receives that wn.
    """
    wn = wntr.network.WaterNetworkModel(str(path))
    wn.options.time.duration = 0

    for name in wn.junction_name_list:
        junction = wn.get_node(name)
        if clear_emitters:
            junction.emitter_coefficient = 0.0
        local = (local_demand_multipliers or {}).get(name, 1.0)
        for demand in junction.demand_timeseries_list:
            demand.base_value *= demand_multiplier * local

    if leak_node is not None:
        if not clear_emitters:
            raise ValueError("Cannot inject a leak when source emitters are kept")
        wn.get_node(leak_node).emitter_coefficient = emitter_si

    results = wntr.sim.EpanetSimulator(wn).run_sim(
        file_prefix=str(RESULTS_DIR / "localization_run"),
        convergence_error=True,
    )
    values = []
    for name in PRESSURE_SENSORS:
        values.append(float(results.node["pressure"].loc[0, name]))
    for name in FLOW_SENSORS:
        values.append(float(results.link["flowrate"].loc[0, name]) * 1000.0)
    return np.asarray(values, dtype=float)


def make_hidden_observation(node="4", coefficient_epanet=0.5,
                            demand_multiplier=None, noisy=True):
    """TEST GENERATOR ONLY: a hidden scenario; no labels are given to fitting."""
    if demand_multiplier is None:
        demand_multiplier = float(rng.uniform(0.95, 1.05))
    # Small, independent demand fluctuations not included in the signature grid.
    local = {n: float(rng.uniform(0.98, 1.02)) for n in JUNCTIONS}
    obs = run_hydraulics(
        NETWORK_FILE, clear_emitters=True, leak_node=node,
        emitter_si=coefficient_epanet / 1000.0,
        demand_multiplier=demand_multiplier,
        local_demand_multipliers=local,
    )
    return obs + (rng.normal(0, SIGMAS) if noisy else 0)


def build_signature_library():
    """Simulate each candidate leak, leak size and possible demand level."""
    signatures = []
    for demand_factor in DEMAND_GRID:
        baseline = run_hydraulics(
            NETWORK_FILE, clear_emitters=True,
            demand_multiplier=demand_factor,
        )
        signatures.append({
            "node": "NO LEAK", "coefficient": 0.0,
            "demand_factor": demand_factor, "readings": baseline,
            "estimated_loss_lps": 0.0,
        })
        for node in JUNCTIONS:
            for coeff_epanet in EMITTER_GRID_EPANET:
                leaking = run_hydraulics(
                    NETWORK_FILE, clear_emitters=True, leak_node=node,
                    emitter_si=coeff_epanet / 1000.0,
                    demand_multiplier=demand_factor,
                )
                # Source pipe 1 is the input pipe in the user's example network.
                input_flow_index = len(PRESSURE_SENSORS)
                extra_flow_lps = max(0.0, (
                    leaking[input_flow_index] - baseline[input_flow_index]
                ))
                signatures.append({
                    "node": node, "coefficient": coeff_epanet,
                    "demand_factor": demand_factor, "readings": leaking,
                    "estimated_loss_lps": extra_flow_lps,
                })
    return signatures


def rank_candidates(observation, library):
    """Use sensor values ONLY; do not access hidden case or emitter metadata."""
    best_by_location = {}
    for case in library:
        normalized_error = (observation - case["readings"]) / SIGMAS
        rmse = float(np.sqrt(np.mean(normalized_error ** 2)))
        node = case["node"]
        if node not in best_by_location or rmse < best_by_location[node]["score"]:
            best_by_location[node] = {
                "location": node,
                "score": rmse,
                "emitter_epanet": case["coefficient"],
                "demand_multiplier": case["demand_factor"],
                "estimated_loss_lps": case["estimated_loss_lps"],
                "estimated_loss_m3_day": case["estimated_loss_lps"] * 86.4,
            }
    ranked = sorted(best_by_location.values(), key=lambda x: x["score"])
    return ranked


def anomaly_status(observation):
    """Apply the previously TRAINED unsupervised detector, if available."""
    if not DETECTOR_FILE.exists():
        return "Not available: train_anomaly_detector.py first", None
    saved = joblib.load(DETECTOR_FILE)
    model, model_features = saved["model"], saved["features"]
    if model_features != FEATURES:
        return "Skipped: detector expects different sensors", None
    frame = pd.DataFrame([observation], columns=FEATURES)
    flag = int(model.predict(frame)[0])
    score = float(model.decision_function(frame)[0])
    return ("ANOMALY" if flag == -1 else "NORMAL"), score


def print_result(observation, ranking, reveal=None):
    print("\n--- Observed sensor readings (input to the predictor) ---")
    for feature, value in zip(FEATURES, observation):
        print(f"{feature:>20}: {value:8.3f}")

    status, anomaly_score = anomaly_status(observation)
    print(f"\nIsolation Forest: {status}")
    if anomaly_score is not None:
        print(f"Anomaly score: {anomaly_score:+.4f}")

    print("\n--- Hydraulic localization: lower mismatch = better fit ---")
    print(f"{'Rank':>4} {'Location':>10} {'Mismatch':>10} "
          f"{'Emitter':>9} {'Demand x':>9} {'Loss m3/day':>13}")
    for rank, item in enumerate(ranking, 1):
        print(f"{rank:>4} {item['location']:>10} "
              f"{item['score']:10.3f} {item['emitter_epanet']:9.2f} "
              f"{item['demand_multiplier']:9.3f} "
              f"{item['estimated_loss_m3_day']:13.1f}")

    leaks = [r for r in ranking if r["location"] != "NO LEAK"]
    if leaks:
        print("\nTop 3 suspected LEAK JUNCTIONS:",
              ", ".join(str(r["location"]) for r in leaks[:3]))
        if len(leaks) > 1 and leaks[1]["score"] <= leaks[0]["score"] * 1.15:
            print("CAUTION: Top locations have similar mismatch; localization is ambiguous.")
    print("These are fit scores, NOT probabilities. No exact pipe location is inferred.")
    if reveal is not None:
        print(f"\nTEST ONLY -- hidden leak revealed AFTER prediction: junction {reveal}")
        if leaks:
            print("Test result:", "CORRECT top-1" if leaks[0]["location"] == reveal
                  else "Not top-1 (check top-3 as well)")


def run_evaluation(library, cases_per_junction=10):
    """Quick simulation-only generalization test, not real-world validation."""
    records = []
    for actual in JUNCTIONS:
        for _ in range(cases_per_junction):
            severity = float(rng.uniform(0.27, 0.72))  # unseen leak sizes
            readings = make_hidden_observation(node=actual,
                      coefficient_epanet=severity, noisy=True)
            ranked = rank_candidates(readings, library)
            leaks = [r for r in ranked if r["location"] != "NO LEAK"]
            detector, _ = anomaly_status(readings)
            records.append({
                "actual": actual, "severity": severity,
                "predicted": leaks[0]["location"],
                "top1": leaks[0]["location"] == actual,
                "top3": actual in [r["location"] for r in leaks[:3]],
                "anomaly_flag": detector == "ANOMALY",
            })
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "localization_evaluation.csv", index=False)
    print("\n--- SIMULATED junction-localization evaluation ---")
    print(f"Tested {len(df)} hidden-leak scenarios across {len(JUNCTIONS)} junctions")
    print(f"Top-1 accuracy: {df['top1'].mean():.1%}")
    print(f"Top-3 accuracy: {df['top3'].mean():.1%}")
    if DETECTOR_FILE.exists():
        print(f"Isolation Forest flagged: {df['anomaly_flag'].mean():.1%}")
    print("These tests share the same network model as the signature library.")
    print("Do not interpret as field accuracy in Mauritius.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-junction", default="4",
                        help="TEST GENERATOR only (default: 4)")
    parser.add_argument("--observed-inp", type=Path,
                        help="EPANET .inp with leak added manually; sensor readings only are extracted")
    parser.add_argument("--evaluate", action="store_true",
                        help="Generate extra hidden-leak tests for top-1/top-3 accuracy")
    args = parser.parse_args()

    if not NETWORK_FILE.exists():
        parser.error("Place leak-free network.inp in your working directory")
    if args.observed_inp and not args.observed_inp.is_file():
        parser.error(f"File not found: {args.observed_inp}")
    RESULTS_DIR.mkdir(exist_ok=True)

    global JUNCTIONS
    network = wntr.network.WaterNetworkModel(str(NETWORK_FILE))
    JUNCTIONS = list(network.junction_name_list)
    if args.hidden_junction not in JUNCTIONS:
        parser.error(f"Unknown junction {args.hidden_junction}; choose from {JUNCTIONS}")
    for sensor in PRESSURE_SENSORS:
        if sensor not in JUNCTIONS:
            parser.error(f"Pressure sensor {sensor} not in {JUNCTIONS}")
    for sensor in FLOW_SENSORS:
        if sensor not in network.pipe_name_list:
            parser.error(f"Flow sensor pipe {sensor} not in network")

    preexisting = [j for j in JUNCTIONS
                   if network.get_node(j).emitter_coefficient]
    if preexisting:
        print(f"WARNING: baseline network has emitters at {preexisting}; "
              "they are cleared for building signatures.")

    print(f"Building EPANET signature library for {len(JUNCTIONS)} "
          "possible leak junctions...")
    print(f"Sensors: {', '.join(FEATURES)}")
    library = build_signature_library()
    print(f"Created {len(library)} hydraulic scenarios")

    if args.observed_inp:
        # This is the ONLY time the leakage-containing file is read.
        # The predictor below uses sensor numbers + the leak-free baseline ONLY.
        obs = run_hydraulics(args.observed_inp, clear_emitters=False)
        reveal = None
        print("Observation source: separate EPANET leak scenario file")
    else:
        obs = make_hidden_observation(node=args.hidden_junction,
                                      coefficient_epanet=0.50)
        reveal = args.hidden_junction
        print("Observation source: automatically simulated hidden leak")

    ranking = rank_candidates(obs, library)
    print_result(obs, ranking, reveal=reveal)
    pd.DataFrame([{k: v for k, v in r.items()} for r in ranking]).to_csv(
        RESULTS_DIR / "localization_ranking.csv", index=False)
    print(f"\nRanking saved: {(RESULTS_DIR / 'localization_ranking.csv').resolve()}")

    if args.evaluate:
        run_evaluation(library)


if __name__ == "__main__":
    main()
