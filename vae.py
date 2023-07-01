import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last FC layer
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 32 * 32 * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        x = x.view(x.size(0), 3, 32, 32)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        # Reconstruction loss
        recon_loss = nn.MSELoss(reduction='sum')(x_hat, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return x_hat, recon_loss, kl_loss


# Training the VAE
def train_vae():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 100

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = CIFAR10(root='path_to_cifar10_data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create VAE instance
    vae = VAE(latent_dim=256).to(device)

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop
    total_steps = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            recon_images, recon_loss, kl_loss = vae(images)

            # Compute total loss
            loss = recon_loss + kl_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], "
                      f"Reconstruction Loss: {recon_loss.item():.4f}, KL Divergence: {kl_loss.item():.4f}")

    # Save the trained model
    torch.save(vae.state_dict(), 'vae_model.pth')


# Example usage
train_vae()
