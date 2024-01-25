# -*- coding: utf-8 -*-
"""
CPSC 8430: HW3 - DCGAN implimentation

@author: domini4
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
import matplotlib.pyplot as plt
from model import DCGAN_Discriminator, DCGAN_Generator, DCGAN_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

lr = 2e-4
batch_size = 64
img_size = 64
channels = 3
z_dim = 100
epochs = 100
features = 64


transform = transforms.Compose(
    [transforms.Resize(img_size),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download data
train_data = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = transform, 
    download = True           
)

# load data
train_load = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

disc = DCGAN_Discriminator(channels, features).to(device)
DCGAN_weights(disc)

gen = DCGAN_Generator(z_dim, features, channels).to(device)
DCGAN_weights(gen)

optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

loss = nn.BCELoss()

fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

# for tensorboard
writer_real = SummaryWriter(f"dclogs/real")
writer_fake = SummaryWriter(f"dclogs/fake")
step = 0

disc.train()
gen.train()

G_losses = []
D_losses = []

for epoch in range(epochs):
    for idx, (real, _) in enumerate(train_load):
        # train discriminator with real data
        disc.zero_grad()
        real = real.to(device)
        batch_size = real.shape[0]
        
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        disc_real = disc(real).reshape(-1)
        loss_disc_real = loss(disc_real, torch.ones_like(disc_real))
        loss_disc_real.backward()
        
        # train discriminator with fake data
        fake = gen(noise)
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc_fake.backward()
        optimizerD.step()
        loss_disc = (loss_disc_real + loss_disc_fake)/2  # book keeping
        
        # train generator
        gen.zero_grad()
        output = disc(fake).reshape(-1)
        loss_gen = loss(output, torch.ones_like(output))
        loss_gen.backward()
        optimizerG.step()
        
        G_losses.append(loss_gen.item())
        D_losses.append(loss_disc.item())
        
        if idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {idx} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
                  
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:10], nrow=5, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:10], nrow=5, normalize=True)
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            
            step+=1




plt.plot(range(len(G_losses)), G_losses, 'r', label="Generator Loss")
plt.plot(range(len(D_losses)), D_losses, 'g', label="Discriminator Loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('DCGAN training loss')
plt.legend()
plt.savefig('dcgan_loss.png', dpi=500)
plt.show()








