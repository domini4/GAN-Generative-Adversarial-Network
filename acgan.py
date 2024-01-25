# -*- coding: utf-8 -*-
"""
CPSC 8430: HW3 - ACGAN implimentation

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
import random
import matplotlib.pyplot as plt
from model import ACGAN_Discriminator, ACGAN_Generator, ACGAN_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

lr = 2e-4
batch_size = 64
img_size = 64
channels = 3    
z_dim = 100     
epochs = 100
features = 64   
classes = 10

real_label = 0.7 + 0.5 * torch.rand(10).to(device)
fake_label = 0.3 * torch.rand(10).to(device)

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

disc = ACGAN_Discriminator(channels, features).to(device)
ACGAN_weights(disc)

gen = ACGAN_Generator(z_dim, channels, features, classes).to(device)
ACGAN_weights(gen)

disc_loss = nn.BCELoss().to(device)
nll_loss = nn.NLLLoss().to(device)


optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0

writer_real = SummaryWriter(f"aclogs/real")
writer_fake = SummaryWriter(f"aclogs/fake")
step = 0

disc.train()
gen.train()

G_losses = []
D_losses = []

for epoch in range(epochs):
    for idx, (real, label) in enumerate(train_load):
        real = real.to(device)
        label = label.to(device)
        batch_size = real.shape[0]
        
        # switching up real and fake labels
        x = random.randint(0,9)
        
        real_target = torch.full((batch_size,), real_label[x]).to(device)
        fake_target = torch.full((batch_size,), fake_label[x]).to(device)
        
        # train discriminator with real data
        optimizerD.zero_grad()
        dis_output, aux_output = disc(real)        
        dis_errD_real = disc_loss(dis_output.view(-1), real_target)
        aux_errD_real = nll_loss(aux_output, label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()        
        
        # train discriminator with fake data
        noise = torch.randn((batch_size, z_dim+classes, 1, 1)).to(device)
        fake = gen(noise)
        dis_output, aux_output = disc(fake.detach())
        dis_errD_fake = disc_loss(dis_output.view(-1), fake_target)
        aux_errD_fake = nll_loss(aux_output, label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        
        # train generator
        optimizerG.zero_grad()
        dis_output, aux_output = disc(fake)
        dis_errG = disc_loss(dis_output.view(-1), real_target)
        aux_errG = nll_loss(aux_output, label)
        errG = dis_errG + aux_errG
        errG.backward()
        optimizerG.step()
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        
        if idx % 100 == 0:
            disc.eval()
            gen.eval()
            
            print(f"Epoch [{epoch+1}/{epochs}] Batch {idx} \
                  Loss D: {errD:.4f}, loss G: {errG:.4f}")
                  
            with torch.no_grad():
                fake = gen(noise)
                img_grid_real = torchvision.utils.make_grid(real[:10], nrow=5, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:10], nrow=5, normalize=True)
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            
            step+=1
            disc.train()
            gen.train()
        
        
        
        
        
        
plt.plot(range(len(G_losses)), G_losses, 'r', label="Generator Loss")
plt.plot(range(len(D_losses)), D_losses, 'g', label="Discriminator Loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('ACGAN training loss')
plt.legend()
plt.savefig('acgan_loss.png', dpi=500)
plt.show()
        
        
        
        
        
        
        
        
        