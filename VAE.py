"""
Code for a variational autoencoder
Date: 04 April 2025
@author: Zarah Aigner
Quellen für die erstellung des VAE werden in der Bachelorarbeit aufgelistet
Für plotten des Latent space und für plotten der rekonsturierten Daten wurden zwei separate Codes erstellt
"""

#importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F


# loading the data 
df = pd.read_csv('/Users/zarahaigner/Documents/Physik_6 Semester/Bachelor/input_data.csv', header=None)
df[0] = df[0].str.replace(r'\[\[|\]\]', '', regex=True)
df = df[0].str.split(expand=True).astype(float)
data = df.to_numpy()

# printing the shape 
print(data.shape)
n_beads = data.shape[1]

# konfiguration to a tensor, explanation of tensor in bachelorthesis
data_tensor = torch.tensor(data, dtype=torch.float32)



# definition of the encoder class
class Encoder(nn.Module):
    # definition of the layers of the encoder 
    def __init__(self, input_dim=20, latent_dim=2): #input_dim == bead_dim
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    # definition of how input runs through the encoder, using relu as activation function -> sometimes also leaky relu is used
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# definition of the decoder class    
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim): #input_dim == output_dim da bead dimensionen gleich sein sollen
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)
    

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
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div /= x.shape[0]
    return recon_loss + kl_div, recon_loss, kl_div
        

#training loop
dataset = TensorDataset (data_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)


# using the adam optimizer 
model = VAE(input_dim=n_beads, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
loss_history = []

for epoch in range(n_epochs):
    # initialization
    total_loss = 0
    total_recon = 0
    total_kl = 0
    # inner loop to run through the batches
    for batch in loader:
        x = batch[0]
        # sets gradient back to zero that the gradients do not sum up
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, recon_loss, kl_div = vae_loss(x, x_hat, mu, logvar)

        #loss, recon_loss, kl_div = vae_loss(x, *model(x))  # unpacking direkt hier
        # backpropagation
        loss.backward()
        # optimization step
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()

    #avg_loss = total_loss / len(dataset)
    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(dataset)
    avg_kl = total_kl / len(dataset)
    loss_history.append((avg_loss, avg_recon, avg_kl))

    print(f"Epoch {epoch+1:03d} | Total: {avg_loss:.2f} | Recon: {avg_recon:.2f} | KL: {avg_kl:.2f}")

model.eval()
with torch.no_grad():
    mu, logvar = model.encoder(data_tensor)
    z = model.reparameterize(mu, logvar)

# ursprünglich ausgeben von ein paar variablen zur überprüfung
print("Latente Variablen (z):")
print(z)


# Mittlerer Rekonstruktionsfehler
model.eval()
with torch.no_grad():
    x_hat, _, _ = model(data_tensor)
    mse = F.mse_loss(x_hat, data_tensor, reduction='mean')
    print(f"Mean Reconstruction Error: {mse}")

# Falls z noch ein Tensor ist, zuerst in NumPy umwandeln
z_numpy = z.cpu().numpy()

# Als CSV-Datei abspeichern
np.savetxt("latent_variables.csv", z_numpy, delimiter=",", header="z1,z2", comments='')



### Nun möchte ich neue Daten erzeugen, dafür werden random datenpunkte aus dem latent raum verwendet und durch den decoder geschickt
model.eval()
num_samples = 1000 
latent_dim = 2

z_samples = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    new_data = model.decoder(z_samples)

new_data_numpy = new_data.cpu().numpy()
np.savetxt("generated_configurations.csv", new_data_numpy, delimiter=",")
print("Neue Konfigurationen gespeichert als 'generated_configurations.csv'")


# Rekonstruktionen erzeugen
num_to_plot = 10  # Anzahl der Beispiele, die geplottet werden sollen
plt.figure(figsize=(12, 8))
for i in range(num_to_plot):
    plt.subplot(num_to_plot, 1, i + 1)
    plt.plot(data_tensor[i].numpy(), label='Original', marker='o')
    plt.plot(x_hat[i].numpy(), label='Rekonstruiert', marker='x')
    plt.title(f"Konfiguration {i}")
    plt.ylabel("Position")
    plt.xlabel("Bead Index")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()



