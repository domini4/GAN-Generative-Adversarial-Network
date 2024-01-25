# -*- coding: utf-8 -*-
"""
CPSC 8430: HW3 - GAN models

@author: domini4
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(3, 128, 5, 1, 2),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(128, 256, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc = nn.Linear(256 * 8 * 8, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.fc(x)
        return output
    
class GAN_Discriminator(nn.Module):
    def __init__(self, in_size):
        super(GAN_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
            )
        
    def forward(self, x):
        return self.disc(x)
    
    
class GAN_Generator(nn.Module):
    def __init__(self, z_dim, in_size):
        super(GAN_Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, in_size),
            nn.Tanh(),
            )
        
    def forward(self, x):
        return self.gen(x)       
    
class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super(DCGAN_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features, out_features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*2, out_features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*4, out_features*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.disc(x)
    
class DCGAN_Generator(nn.Module):
    def __init__(self, noise, in_features, out_features):
        super(DCGAN_Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise, in_features*8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(in_features*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*8, in_features*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*4, in_features*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*2, in_features, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features, out_features, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
            )
    
    def forward(self, x):
        return self.gen(x)
    
def DCGAN_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias.data)
    
    
class WGAN_Critic(nn.Module):
    def __init__(self, in_features, out_features):
        super(WGAN_Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features, out_features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*2, out_features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*4, out_features*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_features*8, 1, kernel_size=4, stride=1, padding=0)
            )
    
    def forward(self, x):
        return self.critic(x)
    
class WGAN_Generator(nn.Module):
    def __init__(self, noise, channels, in_features):
        super(WGAN_Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise, in_features*8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(in_features*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*8, in_features*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*4, in_features*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features*2, in_features, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_features, channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
            )
    
    def forward(self, x):
        return self.gen(x)
    
def WGAN_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    
class ACGAN_Discriminator(nn.Module):
    def __init__(self, channels, features):
        super(ACGAN_Discriminator, self).__init__()
        self.features = features
        self.disc = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(features, features*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(features*2, features*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(features*4, features*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(features*8, features*1, kernel_size = 4, stride = 1, padding = 0, bias=False),
            )
        self.fc_dis = nn.Linear(features, 1)
        self.sig = nn.Sigmoid()
        self.fc_aux = nn.Linear(features, 10)
        self.soft = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.disc(x)
        x = x.view(-1, self.features)
        
        out = self.fc_dis(x)        
        out = self.sig(out)
        classes = self.fc_aux(x)
        classes = self.soft(classes)
        
        
        return out, classes

class ACGAN_Generator(nn.Module):
    def __init__(self, noise, channels, features, classes):
        super(ACGAN_Generator, self).__init__()
        self.noise = noise
        self.features = features
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise+classes, features*8, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features*8, features*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features*4, features*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features*2, features, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, channels, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
            )
        
    def forward(self, x):
        # x = x.view(-1, self.noise)
        # x = self.fc(x)
        # x = x.view(-1, self.features*8, 1, 1)
        x = self.gen(x)
        
        return(x)

def ACGAN_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    