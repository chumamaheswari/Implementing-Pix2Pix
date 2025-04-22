import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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