"""Optional EPANET smoke test: python smoke_test.py
Requires dependencies installed and checks the INCLUDED sample model, not the user's network.
"""
from pathlib import Path
import tempfile
import numpy as np
from prioritize_repairs import simulate, build_library, infer

root = Path(__file__).resolve().parent
network = root / "sample_network.inp"
pressure = ["4", "6"]
flow = ["1"]
with tempfile.TemporaryDirectory(prefix="aqua_smoke_") as td:
    td = Path(td)
    baseline, _ = simulate(network, td / "engine", pressure, flow,
                           clear_emitters=True)
    leak, _ = simulate(network, td / "engine", pressure, flow,
                       clear_emitters=True, leak_node="4", emitter_epanet=0.5)
    assert leak[-1] > baseline[-1], "Leak should raise inlet flow in the sample network"
    assert leak[0] < baseline[0], "Leak should lower pressure at J4"
    signatures = build_library(network, td / "engine", ["2", "3", "4", "5", "6"], pressure, flow)
    sigmas = np.array([.08, .08, .08])
    ranks, _ = infer(leak, signatures, sigmas, 1.0)
    assert str(ranks.iloc[0]["candidate_junction"]) == "4", ranks.to_string(index=False)
    no_leak, _ = infer(baseline, signatures, sigmas, 1.0)
    assert str(no_leak.iloc[0]["candidate_junction"]) == "NO LEAK"
    print("PASS — baseline flow:", round(float(baseline[-1]), 3), "L/s")
    print("PASS — leaking flow:", round(float(leak[-1]), 3), "L/s")
    print("PASS — identified J4 as exact simulated emitter hypothesis")
    print("PASS — identified normal network as NO LEAK")
