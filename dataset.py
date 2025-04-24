
from PIL import Image
import numpy as np
import random
import torch

class Dataset:

    def __init__(self):
        self.dataset = []
        self.batches = []

    def add_path(self, path, file):
        self.dataset.append((f'{path}/{file}.jpg', f'{path}/{file}_mask.png'))
    
    def create_batches(self, batch_size):
        random.shuffle(self.dataset)
        num_batches = len(self.dataset) // batch_size
        self.batches = [self.dataset[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        return num_batches
    
    def get_batch(self, index, transform, crop_size):
        imgs = []
        masks = []
        for entry in self.batches[index]:
            # Open the image and mask
            img = Image.open(entry[0]).convert('RGB')
            mask = Image.open(entry[1])
            # Apply the composed transform that handles both img and mask
            img, mask = transform(img, mask)
            mask = mask.squeeze(0).long()

            imgs.append(img.unsqueeze(0))  # Add batch dimension
            masks.append(mask.unsqueeze(0))  # Add batch dimension

        # Concatenate all images and masks into batches
        return torch.cat(imgs, dim=0), torch.cat(masks, dim=0)

