
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
            img = Image.open(entry[0]).convert('RGB')
            img = transform(img).unsqueeze(0)
            mask = Image.open(entry[1])
            mask = torch.tensor(np.array(mask)).unsqueeze(0)
            if mask.shape[1] != crop_size[0] or mask.shape[2] != crop_size[1]:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).float(), size=crop_size, mode='nearest').long().squeeze(0)
            imgs.append(img)
            masks.append(mask)
        return torch.cat(imgs, dim=0), torch.cat(masks, dim=0)
