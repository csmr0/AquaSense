
import wntr

# Import the network exported from EPANET
network = wntr.network.WaterNetworkModel("network.inp")

# Run the actual EPANET hydraulic engine
simulator = wntr.sim.EpanetSimulator(network)
results = simulator.run_sim(file_prefix="test_run")

# Display available junctions
print("Junctions:", network.junction_name_list)

# Pressure at the first simulated time step (metres)
pressures = results.node["pressure"].iloc[0]

# Pipe flow rates in L/s (WNTR returns m³/s)
flows = results.link["flowrate"].iloc[0] * 1000

print("\nJunction pressures:")
print(pressures)

print("\nPipe flow rates (L/s):")
print(flows)
