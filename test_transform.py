from torchvision import transforms as T
from PIL import Image
from glob import glob
import numpy as np
import random
import torch
import cv2
import os

from torch_transforms import *
from dataset import Dataset
import config

# Define your training transform (unchanged)
training_transform = Compose([
    Resize(config.INPUT_SIZE),
    CenterCrop(config.INPUT_SIZE),
    ColorJitter(
        brightness=0.25,
        contrast=0.15,
        saturation=0.3,
        hue=0.083
    ),
    RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
    ToTensor(),
    AddRandomNoise(noise_prob=0.001),  # 0.1% noise
    ScaleImage(scale_min=0.5, scale_max=1),
    AffineTransform(degrees=15, shear=15),
    ScaleImage(scale_min=1, scale_max=1.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Function to denormalize images for visualization
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean  # reverse normalization

dataset_path = f'{config.DATASET_NAME}-{config.DATASET_VERSION}/train'
training_dataset = Dataset()
all_files = [os.path.splitext(os.path.basename(path))[0] for path in glob(os.path.join(dataset_path, '*.jpg'))]
for file in all_files:
    training_dataset.add_path(dataset_path, file)

if os.path.exists("./output"):
    for file in os.listdir("./output"):
        os.remove(os.path.join("./output", file))
else:
    os.makedirs("./output")

num_batches = training_dataset.create_batches(config.BATCH_SIZE)
imgs, masks = training_dataset.get_batch(0, training_transform, config.INPUT_SIZE)

for i,(img, mask) in enumerate(zip(imgs, masks)):
    denormalized_img = denormalize(img)

    # Convert to numpy array and save
    img_array = (denormalized_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # overlay mask with [0,1,2] to r g b on to img_array
    overlay_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mask_array = np.array(mask)
    overlay_array = overlay_colors[mask_array]*255
    img_array = (img_array * 0.5 + overlay_array * 0.5).astype(np.uint8)

    out_path = os.path.join('./output', f'transformed_{i}.png')
    cv2.imwrite(out_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    print(f'Saved transformed image to {out_path}')