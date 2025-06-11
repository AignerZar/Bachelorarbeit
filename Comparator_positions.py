"""
Code for plotting the position distributions and fitting
Date: 04 April 2025
@author: Zarah Aigner
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import re
from scipy.stats import norm

"""
Loading the input data---------------------------------------------------------------------------------------------------------------------------------------------
"""
def parse_bead_data(filepath):
    with open('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv', 'r') as f:
        text = f.read()
    
    # extracting the vectors
    matches = re.findall(r'\[\[([-+eE0-9.,\s\-]+?)\]\]', text)
    
    bead_positions = []
    for match in matches:
        # converting a string into a list of floats
        vec = np.fromstring(match, sep=' ')
        if len(vec) == 3:
            bead_positions.append(vec)
    
    return np.array(bead_positions)


beads = parse_bead_data('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv')  # adjusting the path if necessary

"""
Plotting the input data----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# preparing the plot for three histograms 
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
labels = ['x', 'y', 'z']


for i in range(3):
    data_i = beads[:, i]
    axs[i].hist(data_i, bins=30, density=True, alpha=0.7, color=f'C{i}')
    # Fit normal distribution
    mu, std = np.mean(data_i), np.std(data_i)
    x = np.linspace(data_i.min(), data_i.max(), 300)
    pdf = norm.pdf(x, mu, std)
    axs[i].plot(x, pdf, 'k--', label=f'Normal distribution')

    axs[i].set_xlabel(f'{labels[i]}')
    axs[i].set_ylabel(f'P({labels[i]})')
    axs[i].set_title(f'Input data: {labels[i]}-coordinate')
    
    axs[i].legend()

plt.tight_layout()
plt.show()


"""
Loading the output data----------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def parse_flat_bead_data(filepath):
    data = np.loadtxt(filepath, delimiter=',') 
    
    return data

filepath = '/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations.csv' #adjusting path if needed
data = parse_flat_bead_data(filepath)

x_coords = data[:, 0::3].flatten()
y_coords = data[:, 1::3].flatten()
z_coords = data[:, 2::3].flatten()

"""
Plotting the output data--------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# preparing the plot for three histograms 
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
labels = ['x', 'y', 'z']
coords = [x_coords, y_coords, z_coords]

# normal distribution from input data
real_beads = parse_bead_data('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv')
real_means = [np.mean(real_beads[:, i]) for i in range(3)]
real_stds = [np.std(real_beads[:, i]) for i in range(3)]


#fig, axs = plt.subplots(1, 3, figsize=(15, 4))
#labels = ['x', 'y', 'z']

x_lims = []
y_max = 0

for i in range(3):
    mu, std = real_means[i], real_stds[i]
    x_min = mu - 4 * std
    x_max = mu + 4 * std
    x_lims.append((x_min, x_max))

    hist_vals, bin_edges = np.histogram(coords[i], bins=30, density=True)
    y_max = max(y_max, max(hist_vals))

for i in range(3):
    data_i = coords[i]
    mu, std = real_means[i], real_stds[i]
    x_min, x_max = x_lims[i]
    x = np.linspace(x_min, x_max, 300)
    pdf = norm.pdf(x, mu, std)

    axs[i].hist(data_i, bins=30, density=True, alpha=0.7, color=f'C{i}', edgecolor='black') #bin size can be adjusted
    axs[i].plot(x, pdf, 'k--', label=f'Normal distribution of the input data')

    axs[i].set_xlim(x_min, x_max)
    axs[i].set_ylim(0, y_max * 1.1)  
    axs[i].set_title(f'Output data: {labels[i]}-coordinate')
    axs[i].set_xlabel(f'{labels[i]}')
    axs[i].set_ylabel(f'P({labels[i]})')
    axs[i].legend()

plt.tight_layout()
plt.show()
