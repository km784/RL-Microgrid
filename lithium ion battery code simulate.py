import numpy as np
import pybamm
import matplotlib.pyplot as plt

# Create a lithium-ion battery model (Doyle-Fuller-Newman model)
model = pybamm.lithium_ion.DFN()

# Define a simple experiment: discharge at 1C for 1 hour
experiment = pybamm.Experiment(["Discharge at 1C for 1 hour"])

# Create a simulation
sim = pybamm.Simulation(model, experiment=experiment)

# Solve the model
solution = sim.solve()

# Plot the results
solution.plot()
plt.show()

# Print some key results
print(f"Final voltage: {solution['Terminal voltage [V]'].entries[-1]:.3f} V")
print(f"Discharge capacity: {solution['Discharge capacity [A.h]'].entries[-1]:.3f} Ah")

line_power_to_house = 5000  # Watts, constant power needs by the house

# Variations in battery power (in Watts)
battery_power_variations = np.arange(0, 6000, 100)  # from 0 to 6000 Watts

# Calculate the power drawn from the grid for each battery power variation
power_grid = line_power_to_house - battery_power_variations

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(battery_power_variations, power_grid, label='Power to the Grid (W)', color='blue')
plt.axhline(0, color='red', linestyle='--', label='Zero Power to Grid')
plt.title('Impact of Battery Power on Power to the Grid')
plt.xlabel('Battery Power (W)')
plt.ylabel('Power to Grid (W)')
plt.grid()
plt.legend()
plt.xlim(0, 6000)
plt.ylim(-6000, 5000)
plt.show()