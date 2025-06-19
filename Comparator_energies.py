"""
Code for plotting the energy distributions and calculating the mean
Date: 04 April 2025
@author: Zarah Aigner

The following code is used to caluclate the energy configurations of the real datapoints, which first got generated from the PIMC code and the
energy configurations of the generated data points, which were produced by the VAE
"""
import numpy as np
import matplotlib.pyplot as plt
import re

"""
Defining the potential (for a HO)----------------------------------------------------------------------------------------------------------------------------------------------------
"""
def potential(r):
    return 0.5 * m * omega**2 * r**2

"""
Defining the parameters------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
P = 20          # Number of beads
beta = 2        # for T = 0.5 -> beta = 1/k_B*T
hbar = 1.0      # atomic units
m = 1.0         # atomic units
omega = 1.0     # atomic units

"""
Loading the input data-------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def parse_line(line):
    matches = re.findall(r'\[([^\]]+)\]', line)
    coords = [list(map(float, triplet.split(','))) for triplet in matches]
    return np.array(coords)  # Shape: (P, 3)

def parse_line(line):
    matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\]", line)
    coords = [list(map(float, m)) for m in matches[:20]]  
    return np.array(coords)  


configs = []
with open("input_data.csv", "r") as f:
    for line in f:
        config = parse_line(line)
        if config.shape == (P, 3):
            configs.append(config)

configs = np.array(configs)  # Shape: (N_samples, P, 3)

"""
Calculating the energy -> E = K + V----------------------------------------------------------------------------------------------------------------------
"""
energies = []

for beads in configs:

    # kinetic energy
    diffs = beads - np.roll(beads, shift=-1, axis=0)
    kinetic = ((m * P) / (2 * beta**2 * hbar**2 )) * np.mean(np.sum(diffs**2, axis=1))

    # potential energy
    r_vals = np.linalg.norm(beads, axis=1)  
    potential_in = np.mean([potential(r) for r in r_vals])

    # total energy
    total_energy = kinetic + potential_in
    energies.append(total_energy)

"""
Calculating the mean and the stanrad derivation--------------------------------------------------------------------------------------------------------
"""
energies = np.array(energies)

mean_energy = np.mean(energies)
std_energy = np.std(energies)

print(f"⟨E⟩ = {mean_energy:.5f} ± {std_energy:.5f}")

"""
Plotting the result----------------------------------------------------------------------------------------------------------------------------------------
"""
plt.hist(energies, bins=50, density=True, alpha=0.8, color="skyblue", edgecolor="k")
plt.xlabel("Energies")
plt.ylabel("Probability")
plt.title("Energy distribution of input data")
plt.grid(True)
plt.savefig("energy_input.pdf")
plt.show()


"""
Loading the output data-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def load_bead_data(filepath):
    def parse_line(line):
        matches = re.findall(r"\[\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\]", line)
        beads = [list(map(float, m)) for m in matches[:20]]
        return np.array(beads)

    bead_configurations = []
    with open(filepath, 'r') as file:
        for line in file:
            beads = parse_line(line)
            if beads.shape == (20, 3):
                bead_configurations.append(beads)

    return np.array(bead_configurations)

generated_data_flat = np.loadtxt('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations.csv', delimiter=",")
generated_data = generated_data_flat.reshape(-1, 20, 3)

"""
Calculating the energy of output data
"""
energies = []
for beads in generated_data:
    diffs = beads - np.roll(beads, shift=-1, axis=0) 
    kinetic = ((m * P) / (2 * beta**2 * hbar**2 )) * np.mean(np.sum(diffs**2, axis=1))

    r_vals = np.linalg.norm(beads, axis=1)
    potential_values_out = np.mean(potential(r_vals))

    total_energy = kinetic + potential_values_out
    energies.append(total_energy)

"""
Mean value and standard derivation
"""
energies = np.array(energies)
mean_energy = np.mean(energies)
std_energy = np.std(energies)

print(f"⟨E⟩ = {mean_energy:.5f} ± {std_energy:.5f}")

"""
Plotting the output data
"""
plt.hist(energies, bins=50, density=True, alpha=0.8, color="skyblue", edgecolor="k")
plt.xlabel("Energies")
plt.ylabel("Probability")
plt.title("Energy distribution of the output data")
plt.grid(True)
plt.savefig("Energy_well_output.pdf")
plt.show()
