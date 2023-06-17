import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nets

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Create an instance of the VIBNet model
model = nets.VIBNet(seed=seed)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the transformation for the input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        z, outputs = model(inputs)
        loss = criterion(outputs, labels) + 0.001 * torch.mean(model.kl_loss) + torch.mean(model.reconstruction_loss)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        # Print the loss every 200 mini-batches
        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

# # Test the reconstruction
# with torch.no_grad():
#     images, labels = next(iter(trainloader))
#     z, reconstructions = model(images)
#     model.plot_input_reconstruction(images[0], reconstructions[0])
