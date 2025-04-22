import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define a custom dataset class for flat image directories
class SimpleImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = [
            fname for fname in os.listdir(directory)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.directory, self.files[index])
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # placeholder label

# Set up transformation pipeline (scaling and normalization)
image_transforms = transforms.Compose([
    transforms.Resize((240, 240)),  # slightly different resize
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4, 0.4, 0.4), std=(0.6, 0.6, 0.6))  # altered normalization
])

# Load datasets from respective folders
sat_path = "./data/satellite"
map_path = "./data/map"

sat_data = SimpleImageDataset(sat_path, transform=image_transforms)
map_data = SimpleImageDataset(map_path, transform=image_transforms)

# Initialize data loaders
batch_size = 12
sat_loader = DataLoader(sat_data, batch_size=batch_size, shuffle=True)
map_loader = DataLoader(map_data, batch_size=batch_size, shuffle=True)

# Visualize a random image from the satellite set
def visualize_random_sample(dataset, title="Satellite Image Example"):
    random_index = random.randint(0, len(dataset) - 1)
    img_tensor, _ = dataset[random_index]
    img_tensor = img_tensor * 0.6 + 0.4  # reverse normalization
    img_np = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    plt.title(title)
    plt.axis("off")
    plt.show()

visualize_random_sample(sat_data)




# -----------
# Models
# -----------
class BasicGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # [B, 64, 128, 128]
            nn.ReLU(True)
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # [B, 3, 256, 256]
            nn.Tanh()
        )

    def forward(self, x):
        return self.decode(self.encode(x))


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(128 * 60 * 60, 1),  # Adjust input size based on your image dimensions
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Setup device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = BasicGenerator().to(device)
D = SimpleDiscriminator().to(device)



# -----------
# Training
# -----------
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

EPOCHS = 10
for epoch in range(EPOCHS):
    for (sat_imgs, _), (map_imgs, _) in zip(sat_loader, map_loader):
        sat_imgs = sat_imgs.to(device)
        map_imgs = map_imgs.to(device)

        # Forward pass
        fake_maps = G(sat_imgs)

        # Discriminator
        D_real = D(map_imgs)
        D_fake = D(fake_maps.detach())
        d_loss = (adversarial_loss(D_real, torch.ones_like(D_real)) +
                  adversarial_loss(D_fake, torch.zeros_like(D_fake))) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Generator
        g_adv = adversarial_loss(D(fake_maps), torch.ones_like(D_fake))
        g_recon = reconstruction_loss(fake_maps, map_imgs)
        g_loss = g_adv + 10 * g_recon

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")



# -----------
# Visualize Translation
# -----------
test_img, _ = sat_data[0]
test_img = test_img.unsqueeze(0).to(device)
translated = G(test_img).detach().squeeze().cpu().permute(1, 2, 0)
original = test_img.squeeze().cpu().permute(1, 2, 0)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original * 0.5 + 0.5)
ax[0].set_title("Original (Satellite)")
ax[0].axis("off")

ax[1].imshow(translated * 0.5 + 0.5)
ax[1].set_title("Generated (Map)")
ax[1].axis("off")
plt.show()
