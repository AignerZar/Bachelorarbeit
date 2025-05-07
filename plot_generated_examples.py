import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CSV-Datei mit generierten Daten laden
pfad = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelor/generated_configurations.csv"
data = pd.read_csv(pfad, delimiter=",", header=None)  

# Anzahl der Konfigurationen und Beads
num_samples, num_beads = data.shape

# Plotten der einzelnen Konfigurationen
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    plt.plot(data.iloc[i], label=f"Sample {i+1}")

plt.title("Generated PIMC-configuration")
plt.xlabel("Bead Index")
plt.ylabel("Position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# nun wird der mittelwert der generierten daten geplottet, ebenso wie die stdabw um eine funktion zu erstellen
mean = data.mean(axis=0)
std = data.mean(axis=0)


# ausgabe des histograms -> im besten fall soll es eine normalverteilung representieren
plt.figure(figsize=(8, 5))
plt.hist(data.values.flatten(), bins=50, density=True, alpha=0.7, color="orange")
plt.title("Histogram of reconstructed Bead-Positions")
plt.xlabel("Position")
plt.grid(True)
plt.tight_layout()
plt.show()
