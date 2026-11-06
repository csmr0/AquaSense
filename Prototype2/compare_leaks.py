
import wntr

FILE = "network.inp"
LEAK_JUNCTION = "4"

# EPANET UI uses L/s/sqrt(m).
# WNTR internally uses m³/s/sqrt(m).
EMITTER_COEFFICIENT = 0.0005


def simulate(leak_at=None):
    network = wntr.network.WaterNetworkModel(FILE)

    # Start every experiment without leaks
    for name in network.junction_name_list:
        network.get_node(name).emitter_coefficient = 0.0

    # Add a simulated leak if requested
    if leak_at is not None:
        network.get_node(leak_at).emitter_coefficient = (
            EMITTER_COEFFICIENT
        )

    simulator = wntr.sim.EpanetSimulator(network)
    prefix = "leak_test" if leak_at else "baseline_test"

    results = simulator.run_sim(file_prefix=prefix)
    return network, results


# Inspect any emitter already saved in the input file
original = wntr.network.WaterNetworkModel(FILE)

print("Emitters in original network:")
for name in original.junction_name_list:
    coefficient = original.get_node(name).emitter_coefficient
    if coefficient:
        print(name, ":", coefficient * 1000)

# Run both experiments
network, baseline = simulate()
_, leaking = simulate(LEAK_JUNCTION)

print("\nPRESSURE COMPARISON (m)")
print("Node | Baseline | Leak | Difference")

for name in network.junction_name_list:
    p0 = baseline.node["pressure"].loc[0, name]
    p1 = leaking.node["pressure"].loc[0, name]

    print(
        f"{name:4} | {p0:8.2f} | "
        f"{p1:6.2f} | {p1-p0:+.2f}"
    )

# Pipe 1 is the supply pipe in our example network
f0 = baseline.link["flowrate"].loc[0, "1"] * 1000
f1 = leaking.link["flowrate"].loc[0, "1"] * 1000

print("\nSOURCE FLOW (L/s)")
print(f"Baseline: {f0:.3f}")
print(f"With leak: {f1:.3f}")

# Extra demand caused by the emitter
d0 = baseline.node["demand"].loc[0, LEAK_JUNCTION]
d1 = leaking.node["demand"].loc[0, LEAK_JUNCTION]

leak_lps = (d1 - d0) * 1000
print(f"\nEstimated leak: {leak_lps:.3f} L/s")
print(f"Daily loss: {leak_lps * 86.4:.2f} m³/day")
