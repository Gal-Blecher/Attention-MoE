import torch
import torch.optim as optim

def train_vae(model, data):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = data['train_loader']

    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 50

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        recon_losses = []
        kl_losses = []
        total_loss = 0

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            recon_images, recon_loss, kl_loss = model(images)

            loss = recon_loss + 0.000001 * kl_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            total_loss += loss.item()

            # Print training progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}]"
                      f"Reconstruction Loss: {recon_loss.item():.4f}, KL Divergence: {kl_loss.item():.4f}")

        # Print epoch summary
        avg_recon_loss = sum(recon_losses) / len(recon_losses)
        avg_kl_loss = sum(kl_losses) / len(kl_losses)
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  Average KL Divergence Loss: {avg_kl_loss:.4f}")

        # Save the model
        torch.save(model, f'/model_vae.pkl')

    print("\nTraining completed!")
    return model

