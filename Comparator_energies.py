"""
Code for plotting the position distributions and fitting
Date: 04 April 2025
@author: Zarah Aigner

The following code is used to caluclate the energy configurations of the real datapoints, which first got generated from the PIMC code and the
energy configurations of the generated data points, which were produced by the VAE

The functions getV, potEnergy, kinetic estimator, total_energy, compute_energy were take from the main code from the PIMC simulation, from Michael Hütter
"""
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
Defining a function to load the data----------------------------------------------------------------------------------------------------------------------------------------------------
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


"""
Defining the functions to calculate the energy distributions, taken from PIMC simulation----------------------------------------------------------------------------------------------------
"""
def getV(R: np.ndarray, state: int) -> float:
    return 0.5 * np.sum(R**2)

def potEnergy(beads: np.ndarray, numTimeSlices: int, eState: np.ndarray) -> float:
    PE = 0.0
    for j in range(numTimeSlices):
        R = beads[j, :]  
        PE += getV(R, eState[j])
    return PE / numTimeSlices

def kinetic_estimator(beads: np.ndarray, tau: float, lam: np.ndarray, numTimeSlices: int, numParticles: int, simulation_dim: int) -> float:
    tot = 0.0
    for tslice in range(numTimeSlices):
        tslicep1 = (tslice + 1) % numTimeSlices
        for ptcl in range(numParticles):
            norm = 1.0 / (4.0 * lam[ptcl] * tau * tau)
            delR = beads[tslicep1, ptcl] - beads[tslice, ptcl]
            tot -= norm * np.dot(delR, delR)
    return (simulation_dim / 2) * numParticles / tau + tot / numTimeSlices

def total_energy(beads: np.ndarray, tau: float, lam: np.ndarray, simulation_dim: int, eState: np.ndarray) -> float:
    numTimeSlices, simulation_dim_check = beads.shape
    beads = beads.reshape(numTimeSlices, 1, simulation_dim_check)  # (M, 1, 3)
    numParticles = 1
    KE = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles, simulation_dim)
    PE = potEnergy(beads, numTimeSlices, eState)
    return KE + PE

def compute_energies(all_beads: np.ndarray, tau: float, lam: np.ndarray, simulation_dim: int) -> np.ndarray:
    N, numTimeSlices, _ = all_beads.shape
    energies = np.zeros(N)
    for i in range(N):
        beads = all_beads[i]
        eState = np.zeros(numTimeSlices, dtype=int)  
        energies[i] = total_energy(beads, tau, lam, simulation_dim, eState)
    return energies

"""
Loading the input and output data----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
    filepath = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv" # can be adjusted if needed
    real_data = load_bead_data(filepath) 

    tau = 0.1 #imaginary time step, must be small enough for trotter
    lam = np.array([0.5]) #because of arbitrary unit lam = hbar^2/2m
    simulation_dim = 3 #3D simulation

    energies = compute_energies(real_data, tau, lam, simulation_dim) # input data

    
    generated_data_flat = np.loadtxt('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations_denorm.csv', delimiter=",") # can be adjusted if needed
    generated_data = generated_data_flat.reshape(-1, 20, 3)

    energies_generatd = compute_energies(generated_data, tau, lam, simulation_dim) # output data

"""
Plotting the energy distribution of input data-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# Parameters for the fit may needed to be adjusted
kT = 5
shift = 15 - kT / 2
A2 = 0.12 * np.exp(0.5) * np.sqrt(2 / kT)

# fit
def maxwell_boltzmann_shifted_fixed(E, A, kT, shift):
    E_shifted = E - shift
    return np.where(E_shifted > 0, A * E_shifted**0.5 * np.exp(-E_shifted / kT), 0)

E_min = max(10, min(energies))
E_max = max(energies) + 10
E_fit = np.linspace(E_min, E_max, 1000)

# calculating the fit
fit_curve = maxwell_boltzmann_shifted_fixed(E_fit, A2, kT, shift)

# plotting just the data
plt.figure(figsize=(8, 5))
plt.hist(energies, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")
plt.xlabel("Energy [a.u.]")
plt.ylabel("Probability distribution")
plt.title("Histogram of energies")
plt.grid(True)
plt.tight_layout()
plt.show()

# plotting the data and the fit
plt.figure(figsize=(8, 5))
plt.hist(energies, bins=50, density=True, alpha=0.5, color="skyblue", edgecolor="black", label="Input data")
plt.plot(E_fit, fit_curve, 'r-', linewidth=2, label="Maxwell-Boltzmann fit")
plt.xlabel("Energy [a.u.]")
plt.ylabel("Probability distribution")
plt.title("Histogram with Maxwell-Boltzmann fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# calculating the area under the fit
area_input = np.trapz(fit_curve, E_fit)

"""
Plotting the energy distributions of the output data-----------------------------------------------------------------------------------------------------------
"""
# Fitparameter, may need to be adjusted
kT = 3.5
shift = 15.0  
A = 0.16 * np.exp(0.5) * np.sqrt(2 / kT)

# Fitfunction
def maxwell_boltzmann_shifted_fixed(E, A, kT):
    E_shifted = E - 15.0
    return np.where(E_shifted > 0, A * E_shifted**0.5 * np.exp(-E_shifted / kT), 0)

# plotting just the data
plt.figure(figsize=(8, 5))
plt.hist(energies_generatd, bins=50, density=True, alpha=0.8, color="blue", edgecolor="black")
plt.xlabel("Energy [a.u.]")
plt.ylabel("Probability distribution")
plt.title("Energy distribution of the output data")
plt.grid(True)
plt.tight_layout()
plt.show()

# plotting the data and the fit
counts, bin_edges = np.histogram(energies_generatd, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
E_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)

fit_curve = maxwell_boltzmann_shifted_fixed(E_fit, A, kT)

plt.figure(figsize=(10, 6))
plt.hist(energies_generatd, bins=50, density=True, alpha=0.5, color="skyblue", edgecolor="k", label="Output data")
plt.plot(E_fit, fit_curve, 'r-', linewidth=2, label="Maxwell-Boltzmann fit")
plt.xlabel("Energy [a.u.]")
plt.ylabel("Probability distribution")
plt.title("Probability distribution with Maxwell-Boltzmann fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# calculating the area
area_output = np.trapz(fit_curve, E_fit)


"""
Comparing the results--------------------------------------------------------------------------------------------------------------------------------- 
"""
# --- Mittelwerte vergleichen ---
mean_real = np.mean(energies)
mean_generated = np.mean(energies_generatd)

print(f"Mean energies of input data:      {mean_real:.4f}")
print(f"Mean energies of output data: {mean_generated:.4f}")
print(f"Area under the fit of input data: {area_input:.6f}")
print(f"Area under the fit of output data: {area_output:.6f}")
