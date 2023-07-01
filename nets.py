import torch
import torch.nn as nn
import torch.nn.functional as F
from config import train_config, setup



import torch.nn.init as init
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np



class VIBNet(nn.Module):
    def __init__(self, seed, latent_dim=train_config['nets']['VIBNet']['emb_dim'], num_classes=10):
        super(VIBNet, self).__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_combined = nn.Linear(16 * 7 * 7, latent_dim * 2)
        self.fc_classifier = nn.Linear(latent_dim, num_classes)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        x_input = x
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        combined = self.fc_combined(x)
        mean, logvar = torch.chunk(combined, 2, dim=1)
        z = self.reparameterize(mean, logvar)

        # Reshape z to match the expected input shape of the decoder
        z = z.view(z.size(0), -1)

        x_hat = self.decoder(z)
        classification_output = self.fc_classifier(z)

        # Reconstruction loss
        recon_loss = (x_hat - x_input) ** 2
        recon_loss = recon_loss.flatten(start_dim=1).mean(1)


        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

        self.reconstruction_loss = recon_loss
        self.kl_loss = kl_loss

        self.out = classification_output

        return z, classification_output

    def plot_input_reconstruction(self, x, x_hat):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # Plot input image
        axes[0].imshow(x.detach().cpu().squeeze(), cmap='gray')
        axes[0].set_title('Input')
        axes[0].axis('off')

        # Plot reconstruction image
        axes[1].imshow(x_hat.detach().cpu().squeeze(), cmap='gray')
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

class VIBNetResNet(nn.Module):
    def __init__(self, seed, latent_dim=train_config['nets']['VIBNetResNet']['emb_dim'], num_classes=10):
        super(VIBNetResNet, self).__init__()
        torch.manual_seed(seed)
        resnet18 = ResNet18(seed)
        self.encoder = resnet18


        self.fc_classifier = nn.Linear(int(latent_dim/2), num_classes)

        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim/2), 256*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        x_input = x
        x = self.encoder(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mean, logvar)

        x_hat = self.decoder(z)
        classification_output = self.fc_classifier(z)

        # Reconstruction loss
        # recon_loss = (x_hat - x_input) ** 2
        # recon_loss = recon_loss.flatten(start_dim=1).mean(1)
        batch_size = x_input.size(0)
        loss = F.mse_loss(x_input, x_hat)
        recon_loss = loss / batch_size


        # KL divergence loss
        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


        self.reconstruction_loss = recon_loss
        self.kl_loss = kl_loss

        self.out = classification_output

        return z, classification_output

    def plot_input_reconstruction(self, x, x_hat):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # Transpose the tensor dimensions (C, H, W) to (H, W, C) for visualization
        x = np.transpose(x.detach().cpu().numpy(), (1, 2, 0))
        x_hat = np.transpose(x_hat.detach().cpu().numpy(), (1, 2, 0))

        # Plot input image
        axes[0].imshow(x)
        axes[0].set_title('Input')
        axes[0].axis('off')

        # Plot reconstruction image without specifying cmap
        axes[1].imshow(x_hat)
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        z = out.view(out.size(0), -1)
        out = self.linear(z)
        self.out = out
        return z

def ResNet18(e):
    torch.manual_seed(e)
    return ResNet(BasicBlock, [2, 2, 2, 2])










