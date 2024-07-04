import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Define the Autoencoder for single-channel images
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Load dataset and define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Instantiate model, loss function, and optimizer
autoencoder = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# Function to compute the median image of a batch
def compute_median_image(batch):
    batch_np = batch.cpu().numpy()
    median_image = np.median(batch_np, axis=0)
    median_image_tensor = torch.tensor(median_image).float().cuda()
    return median_image_tensor

# Training loop
for epoch in range(20):
    for img, _ in data_loader:
        img = img.cuda()
        median_img = compute_median_image(img)
        optimizer.zero_grad()
        decoded, encoded = autoencoder(median_img)
        loss = criterion(decoded, median_img)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Function to compute and store features and reconstruction errors for triplets
def extract_features_and_errors(autoencoder, data_loader):
    autoencoder.eval()
    features_dict = {}
    reconstruction_error_dict = {}
    with torch.no_grad():
        for batch_idx, (img, _) in enumerate(data_loader):
            img = img.cuda()
            decoded, encoded = autoencoder(img)
            losses = torch.mean((decoded - img) ** 2, dim=[1, 2, 3])
            for i in range(img.size(0)):
                sample_idx = batch_idx * data_loader.batch_size + i
                features_dict[sample_idx] = encoded[i].cpu().numpy()
                reconstruction_error_dict[sample_idx] = losses[i].cpu().numpy()
    return features_dict, reconstruction_error_dict

# Extract features and reconstruction errors
features_dict, reconstruction_error_dict = extract_features_and_errors(autoencoder, data_loader)

# Example triplets and their median reconstruction errors
triplets = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]  # Example triplet indices
triplet_errors = {triplet: np.median([reconstruction_error_dict[idx] for idx in triplet]) for triplet in triplets}

# Visualize the top anomalies based on median reconstruction error
top_triplets = sorted(triplet_errors, key=triplet_errors.get, reverse=True)[:10]

fig, axes = plt.subplots(len(top_triplets), 3, figsize=(15, len(top_triplets) * 5))
for i, triplet in enumerate(top_triplets):
    for j, idx in enumerate(triplet):
        img, _ = dataset[idx]
        axes[i, j].imshow(img.permute(1, 2, 0).squeeze() * 0.5 + 0.5, cmap='gray')
        axes[i, j].axis('off')
plt.show()
