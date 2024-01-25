# -*- coding: utf-8 -*-
"""
CPSC 8430: HW3 - WGAN implimentation

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
from model import WGAN_Critic, WGAN_Generator, WGAN_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

lr = 5e-5
batch_size = 64
img_size = 64
channels = 3
z_dim = 100
epochs = 100
features = 64
critic_itterations = 5
weight_clip = 1e-2

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

critic = WGAN_Critic(channels, features).to(device)
WGAN_weights(critic)

gen = WGAN_Generator(z_dim, channels, features).to(device)
WGAN_weights(gen)

optimizerC = optim.RMSprop(critic.parameters(), lr=lr)
optimizerG = optim.RMSprop(gen.parameters(), lr=lr)

writer_real = SummaryWriter(f"wlogs/real")
writer_fake = SummaryWriter(f"wlogs/fake")
step = 0

critic.train()
gen.train()

G_losses = []
D_losses = []

for epoch in range(epochs):
    for idx, (real, _) in enumerate(train_load):
        real = real.to(device)
        batch_size = real.shape[0]
        
        # train critic
        for _ in range(critic_itterations):

            # train critic with real
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            critic_real = critic(real).view(-1)
            critic_real_loss = torch.mean(critic_real)
            
            # train critic with fake
            fake = gen(noise)
            critic_fake = critic(fake).view(-1)
            critic_fake_loss = torch.mean(critic_fake)
            loss_critic = -critic_real_loss + critic_fake_loss
            optimizerC.zero_grad()
            loss_critic.backward()
            optimizerC.step()
            
            # clip critic weights
            for param in critic.parameters():
                param.data.clamp_(-weight_clip, weight_clip)
                
        # train gen
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)
        output = critic(fake).view(-1)
        loss_gen = -torch.mean(output)
        optimizerG.zero_grad()
        loss_gen.backward()
        optimizerG.step()
        
        G_losses.append(loss_gen.item())
        D_losses.append(loss_critic.item())
        
        if idx % 100 == 0:
            critic.eval()
            gen.eval()
            
            print(f"Epoch [{epoch+1}/{epochs}] Batch {idx} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
                  
            with torch.no_grad():
                fake = gen(noise)
                img_grid_real = torchvision.utils.make_grid(real[:10], nrow=5, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:10], nrow=5, normalize=True)
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            
            step+=1
            critic.train()
            gen.train()

plt.plot(range(len(G_losses)), G_losses, 'r', label="Generator Loss")
plt.plot(range(len(D_losses)), D_losses, 'g', label="Discriminator Loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('WGAN training loss')
plt.legend()
plt.savefig('wgan_loss.png', dpi=500)
plt.show()

    