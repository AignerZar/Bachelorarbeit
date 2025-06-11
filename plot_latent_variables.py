"""
Code for plotting the latent variables
Date: 04 April 2025
@author: Zarah Aigner
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['text.usetex'] = True # for better descriptions 

"""
Loading the data---------------------------------------------------------------------------------------------------------------
"""
z_df = pd.read_csv('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/latent_variables.csv')


# calculating the mean values
mean_z1 = np.mean(z_df.iloc[:, 0])
mean_z2 = np.mean(z_df.iloc[:, 1])

"""
Plotting the latent variable z1-----------------------------------------------------------------------------------------------------
"""
plt.figure(figsize=(6, 4))
data = z_df.iloc[:, 0]
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Data')

# normal distribution
mu, std = norm.fit(data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label=f'Normal distribution n$\mu={mu:.2f}$, $\sigma={std:.2f}$')

plt.xlabel(r'$z_1$')
plt.ylabel('frequency (normed)')
plt.title('Histogram of the latent variable $z_1$')
plt.legend()
plt.tight_layout()
plt.show()

"""
Plotting the latent variable z2---------------------------------------------------------------------------------------------------------
"""
plt.figure(figsize=(6, 4))
data = z_df.iloc[:, 1]
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Data')

mu, std = norm.fit(data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label=f'Normal distribution \n$\mu={mu:.2f}$, $\sigma={std:.2f}$')

plt.xlabel(r'$z_2$')
plt.ylabel('frequency (normed)')
plt.title('Histogram of the latent variable $z_2$')
plt.legend()
plt.tight_layout()
plt.show()
