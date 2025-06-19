"""
Code for a variational autoencoder
Date: 04 April 2025
@author: Zarah Aigner
The following code represents a VAE
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
from sklearn.model_selection import train_test_split
import random
import config

# Reproduciability
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

"""
Loading and preparing the input data-----------------------------------------------------------------------------------------
"""

# function to parse into bead vector 
def parse_line(line):
    matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\]", line)
    beads = [list(map(float, m)) for m in matches[:20]]  
    return np.array(beads)  

bead_configurations = []

# loading the input file
with open(config.input_file, 'r') as file:
    for line in file:
        beads = parse_line(line)
        if beads.shape == (20, 3):  
            bead_configurations.append(beads)

# converting into an array
bead_configurations = np.array(bead_configurations)  # shape: (n_samples, 20, 3)
print("Shape der Bead-Konfigurationen:", bead_configurations.shape)
data_np = bead_configurations.reshape(-1, 3)  


# normalizing the input
mean = data_np.mean(axis=0)
std = data_np.std(axis=0)

print("Mean per coord:", mean)
print("Std per coord:", std)

bead_configurations_norm = (bead_configurations - mean) / std


# Number of beads
n_beads = bead_configurations.shape[1]
print("Anzahl Beads pro Konfiguration:", n_beads)

# COnverting into a tensor
data_tensor = torch.tensor(bead_configurations_norm, dtype=torch.float32).view(-1, 60).to(config.device)
print("Shape des Tensors:", data_tensor.shape)  # (n_samples, 20, 3)

# Splitting in training and validation data
train_data, val_data = train_test_split(data_tensor, test_size=config.validation_split, random_state=config.seed)

# DataLoader
train_loader = DataLoader(TensorDataset(train_data), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=config.batch_size, shuffle=False)

"""
Defining the architecture of the VAE-------------------------------------------------------------------------------------------------
"""
# definition of the encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim): 
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# definition of the decoder class    
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim): 
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4(h)
    

# definition of the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    # reparametrize function to make it trainable with the backpropagation  
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # random value
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
# definition of the loss function    
def vae_loss(x, x_hat, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div, recon_loss, kl_div
        
"""
Training loop-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# using the adam optimizer 
model = VAE(config.input_dim, config.latent_dimension).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


loss_history = []

for epoch in range(config.n_epochs):
    model.train()
    total_loss = total_recon = total_kl = 0

    for x_batch, in train_loader:
        x_batch = x_batch.to(config.device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x_batch)
        loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)

    # Validation
    model.eval()
    val_loss = val_recon = val_kl = 0
    with torch.no_grad():
        for x_batch, in val_loader:
            x_batch = x_batch.to(config.device)
            x_hat, mu, logvar = model(x_batch)
            loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar)
            val_loss += loss.item()
            val_recon += recon.item()
            val_kl += kl.item()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    loss_history.append((avg_loss, avg_recon, avg_kl, val_loss, val_recon, val_kl))

    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.3f} | Val Loss: {val_loss:.3f} | Recon: {avg_recon:.3f}/{val_recon:.3f} | KL: {avg_kl:.3f}/{val_kl:.3f}")

"""
Latent space-----------------------------------------------------------------------------------------------------------------------------------------------------------------
"""    
model.eval()
with torch.no_grad():
    mu, logvar = model.encoder(data_tensor)
    z = model.reparameterize(mu, logvar)
    np.savetxt(config.latent_output_file, mu.cpu().numpy(), delimiter=",", header="latent", comments='')


"""
Saving the data----------------------------------------------------------------------------------------------------------------------------------------------------------
"""
z_samples = torch.randn(config.num_samples, config.latent_dimension).to(config.device)
with torch.no_grad():
    generated = model.decoder(z_samples)

generated_np = generated.cpu().numpy().reshape(config.num_samples, 20, 3)
generated_np_denorm = generated_np * std + mean
np.savetxt(config.generated_output_file, generated_np.reshape(config.num_samples, -1), delimiter=",")
np.savetxt(config.generated_output_denorm_file, generated_np_denorm.reshape(config.num_samples, -1), delimiter=",")

"""
Plotting the training and validation loss course------------------------------------------------------------------------------------------------------------------------------------
"""
loss_history = np.array(loss_history)
plt.figure(figsize=(10, 5))
plt.plot(loss_history[:, 0], label='Train Total Loss')
plt.plot(loss_history[:, 3], label='Val Total Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f"loss_{config.n_epochs}.pdf")
plt.show()
