import torch
import torch.nn as nn
import torch.nn.functional as F
from config import train_config, setup



import torch.nn.init as init
from torchvision import models
import matplotlib.pyplot as plt



class VIBNet(nn.Module):
    def __init__(self, seed, latent_dim=4, num_classes=10):
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
        recon_loss = nn.MSELoss(reduction='none')(x_hat, x_input)
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









