"""AquaSense Milestone 4 — EPANET pressure sensor placement comparison.

Compare all 2-pressure-sensor placements fairly on IDENTICAL simulated leak
scenarios, while keeping one flow sensor. Uses the unchanged leak-free network.

Run:
    python optimize_sensors.py
    python optimize_sensors.py --all-flow-pipes
    python optimize_sensors.py --cases-per-node 25

Requirements: pip install wntr numpy pandas

The ranking method is hydraulic-signature matching, not a supervised ML model.
Detection must be evaluated separately; this script assumes there is one leak.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import wntr


EMITTERS_EPANET = (0.20, 0.35, 0.50, 0.65, 0.80)
DEMAND_FACTORS = (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10)
PRESSURE_NOISE_M = 0.08   # assumed sensor standard deviation, metres
FLOW_NOISE_LPS = 0.08    # assumed sensor standard deviation, L/s


def simulate(network_path, junctions, pipes, prefix, *, leak=None,
             emitter_lps=0.0, demand_multiplier=1.0, local=None):
    """A fresh model for each scenario; vector = all pressures and all pipe flows."""
    wn = wntr.network.WaterNetworkModel(str(network_path))
    wn.options.time.duration = 0
    local = local or {}
    for name in junctions:
        junction = wn.get_node(name)
        junction.emitter_coefficient = 0.0  # leak-free reference
        for demand in junction.demand_timeseries_list:
            demand.base_value *= demand_multiplier * local.get(name, 1.0)
    if leak is not None:
        # For exponent 0.5 and LPS in the EPANET editor: convert L/s to m^3/s.
        wn.get_node(leak).emitter_coefficient = emitter_lps / 1000.0
    result = wntr.sim.EpanetSimulator(wn).run_sim(
        file_prefix=str(prefix), convergence_error=True,
    )
    pressures = result.node["pressure"].loc[0, junctions].to_numpy(dtype=float)
    flows = (result.link["flowrate"].loc[0, pipes].to_numpy(dtype=float)
             * 1000.0)
    return np.concatenate((pressures, flows))


def scenario_set(network_path, junctions, pipes, prefix, n_per_node, seed):
    """Generate unseen leak sizes and individual demand variations.

    Every sensor configuration will use the EXACT SAME observations, including
    the same measurement noise, so comparisons are paired and reproducible.
    """
    rng = np.random.default_rng(seed)
    observations = []
    truth = []
    noise_sd = np.array([PRESSURE_NOISE_M] * len(junctions)
                        + [FLOW_NOISE_LPS] * len(pipes))
    for node in junctions:
        for _ in range(n_per_node):
            emitter = float(rng.uniform(0.27, 0.72))
            demand = float(rng.uniform(0.95, 1.05))
            local = {j: float(rng.uniform(0.98, 1.02)) for j in junctions}
            v = simulate(network_path, junctions, pipes, prefix,
                         leak=node, emitter_lps=emitter,
                         demand_multiplier=demand, local=local)
            observations.append(v + rng.normal(0.0, noise_sd))
            truth.append(node)
    return np.vstack(observations), np.array(truth)


def metrics(observations, truth, signatures, signature_nodes, selected_indices,
            junctions):
    """Fit each candidate junction's best matching hydraulic scenario.

    Returns top-1/top-3 fraction and predicted rank for each observation.
    Does NOT inspect hidden junction during inference; truth is used for scoring.
    """
    selected_indices = np.asarray(selected_indices, dtype=int)
    noise = np.array([PRESSURE_NOISE_M if idx < len(junctions)
                      else FLOW_NOISE_LPS for idx in selected_indices])
    sig = signatures[:, selected_indices]
    candidate_rows = [np.flatnonzero(signature_nodes == j) for j in junctions]
    top1 = []
    top3 = []
    rows = []
    for observation, actual in zip(observations[:, selected_indices], truth):
        errors = np.sqrt(np.mean(((sig - observation) / noise) ** 2, axis=1))
        best_scores = [float(errors[indices].min()) for indices in candidate_rows]
        order = np.argsort(best_scores, kind="stable")
        candidates = [junctions[i] for i in order]
        hit1 = candidates[0] == actual
        hit3 = actual in candidates[:3]
        top1.append(hit1)
        top3.append(hit3)
        rows.append({"actual": actual, "predicted": candidates[0],
                     "top3": ",".join(candidates[:3]),
                     "top1_correct": int(hit1), "top3_correct": int(hit3)})
    return {"top1": float(np.mean(top1)), "top3": float(np.mean(top3)),
            "cases": pd.DataFrame(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="network.inp", help="Leak-free EPANET INP")
    ap.add_argument("--flow-pipe", default="1", help="Fixed flow sensor pipe (default: 1)")
    ap.add_argument("--all-flow-pipes", action="store_true",
                    help="Also find the best flow pipe for two pressure sensors")
    ap.add_argument("--cases-per-node", type=int, default=10,
                    help="Number of leak tests PER junction PER validation/holdout split (default: 10)")
    args = ap.parse_args()
    if args.cases_per_node < 1:
        ap.error("--cases-per-node must be at least 1")
    network_path = Path(args.network)
    if not network_path.is_file():
        ap.error(f"Cannot find {network_path}; put script next to network.inp")

    wn = wntr.network.WaterNetworkModel(str(network_path))
    junctions = list(wn.junction_name_list)
    pipes = list(wn.pipe_name_list)
    if len(junctions) < 3:
        ap.error("Need at least 3 junctions to compare two-sensor placements")
    if args.flow_pipe not in pipes:
        ap.error(f"Pipe {args.flow_pipe!r} not found; choose from {pipes}")

    out = Path("ml_results")
    out.mkdir(exist_ok=True)
    prefix = out / "sensor_search_epanet"
    suspicious_emitters = [j for j in junctions
                           if wn.get_node(j).emitter_coefficient]
    if suspicious_emitters:
        print("WARNING: baseline has emitters; they will be cleared for all scenarios:",
              suspicious_emitters)

    signatures = []
    node_labels = []
    print(f"Generating signature library for {len(junctions)} leak locations...")
    for demand_multiplier in DEMAND_FACTORS:
        for junction in junctions:
            for emitter in EMITTERS_EPANET:
                signatures.append(simulate(
                    network_path, junctions, pipes, prefix,
                    leak=junction, emitter_lps=emitter,
                    demand_multiplier=demand_multiplier,
                ))
                node_labels.append(junction)
    signatures = np.vstack(signatures)
    node_labels = np.asarray(node_labels)
    print(f"Created {len(signatures)} leak signatures.")

    print(f"Generating {len(junctions) * args.cases_per_node} common validation scenarios...")
    validation, validation_truth = scenario_set(
        network_path, junctions, pipes, prefix, args.cases_per_node, seed=2027)
    print(f"Generating {len(junctions) * args.cases_per_node} independent holdout scenarios...")
    holdout, holdout_truth = scenario_set(
        network_path, junctions, pipes, prefix, args.cases_per_node, seed=2028)

    flow_choices = pipes if args.all_flow_pipes else [args.flow_pipe]
    pairs = list(combinations(junctions, 2))
    results = []
    for pair in pairs:
        for pipe in flow_choices:
            columns = [junctions.index(pair[0]), junctions.index(pair[1]),
                       len(junctions) + pipes.index(pipe)]
            m = metrics(validation, validation_truth, signatures, node_labels,
                        columns, junctions)
            results.append({"sensor_1": pair[0], "sensor_2": pair[1],
                            "flow_pipe": pipe, "validation_top1_pct": 100 * m["top1"],
                            "validation_top3_pct": 100 * m["top3"],
                            "columns": columns})

    # Select on validation data ONLY. Evaluate one selected configuration on a
    # previously untouched holdout set to reduce selection bias.
    results.sort(key=lambda x: (-x["validation_top1_pct"],
                                -x["validation_top3_pct"],
                                x["sensor_1"], x["sensor_2"], x["flow_pipe"]))
    export_rows = [{k: v for k, v in r.items() if k != "columns"} for r in results]
    pd.DataFrame(export_rows).to_csv(out / "sensor_placement_validation.csv",
                                     index=False)

    print("\n--- SENSOR PLACEMENT SEARCH (validation results) ---")
    print(f"Evaluated {len(results)} configurations with the SAME {len(validation_truth)} hidden leaks.")
    for rank, r in enumerate(results[:min(10, len(results))], 1):
        print(f"{rank:2d}. Pressure {r['sensor_1']}, {r['sensor_2']} + "
              f"flow {r['flow_pipe']} | top-1 {r['validation_top1_pct']:5.1f}% "
              f"| top-3 {r['validation_top3_pct']:5.1f}%")

    best = results[0]
    best_holdout = metrics(holdout, holdout_truth, signatures, node_labels,
                           best["columns"], junctions)
    print("\n--- INDEPENDENT HOLDOUT RESULT (selected configuration) ---")
    print(f"Pressure {best['sensor_1']}, {best['sensor_2']} and flow {best['flow_pipe']}")
    print(f"Top-1: {best_holdout['top1']:.1%} | Top-3: {best_holdout['top3']:.1%} "
          f"({len(holdout_truth)} hidden leaks)")
    best_holdout["cases"].to_csv(out / "selected_sensor_holdout.csv", index=False)

    # Also benchmark the original sensor placement, if present.
    if "2" in junctions and "6" in junctions and "1" in pipes:
        original_columns = [junctions.index("2"), junctions.index("6"),
                            len(junctions) + pipes.index("1")]
        original_holdout = metrics(holdout, holdout_truth, signatures, node_labels,
                                   original_columns, junctions)
        print("\nOriginal pressure 2, 6 + flow 1 on the IDENTICAL holdout cases:")
        print(f"Top-1: {original_holdout['top1']:.1%} | Top-3: {original_holdout['top3']:.1%}")

    print(f"\nResults saved to {out.resolve()}")
    print("Important: synthetic single-leak junction localization only; not field accuracy.")
    print("Your Isolation Forest must still be tested separately for anomaly detection.")


if __name__ == "__main__":
    main()
