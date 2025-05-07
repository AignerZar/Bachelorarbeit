import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Lade latente Variablen (z1, z2)
z_df = pd.read_csv('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelor/latent_variables.csv')
# Überprüfen, welche Spalten vorhanden sind
print("Spalten:", z_df.columns)

# 2D-Scatterplot der ersten beiden latenten Dimensionen
plt.figure(figsize=(8, 6))
plt.scatter(z_df.iloc[:, 0], z_df.iloc[:, 1], alpha=0.7)
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.title("Latent space (2D)")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
plt.hist(z_df.iloc[:, 0], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("z1")
plt.ylabel("frequency")
plt.title("Histogram of the latent space of z1")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
plt.hist(z_df.iloc[:, 1], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("z2")
plt.ylabel("frequency")
plt.title("Histogram of the latentspace of z2")
plt.tight_layout()
plt.show()