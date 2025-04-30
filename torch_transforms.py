from torchvision import transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return img, mask

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = F.center_crop(img, self.size)
        mask = F.center_crop(mask, self.size)
        return img, mask
    
class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, mask):
        img = self.jitter(img)
        return img, mask

class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            for t in self.transforms:
                img, mask = t(img, mask)
        return img, mask

class GaussianBlur:
    def __init__(self, kernel_size, sigma):
        self.blur = T.GaussianBlur(kernel_size, sigma)

    def __call__(self, img, mask):
        img = self.blur(img)
        return img, mask

class ToTensor:
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = torch.tensor(np.array(mask)).unsqueeze(0).float()
        return img, mask

class Normalize:
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean, std)

    def __call__(self, img, mask):
        img = self.normalize(img)
        return img, mask

class AddRandomNoise:
    def __init__(self, noise_prob=0.001):  # 0.1% of pixels
        self.noise_prob = noise_prob

    def __call__(self, img, mask):
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        n_mask = torch.rand_like(img[0]) < self.noise_prob
        noise = torch.rand_like(img) * n_mask.unsqueeze(0)
        img = torch.clamp(img + noise, 0.0, 1.0)
        return img, mask

class ScaleImage:
    def __init__(self, scale_min=0.9, scale_max=1.1):  
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img, mask):
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        h, w = img.shape[1:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        img = T.Resize((new_h, new_w))(img)
        img = T.CenterCrop((h, w))(img)
        
        # Resize mask (same scale) using nearest neighbor interpolation
        mask = T.Resize((new_h, new_w), interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = T.CenterCrop((h, w))(mask)
        
        return img, mask

class AffineTransform:
    def __init__(self, degrees=0, shear=0):
        self.degrees = degrees
        self.shear = shear

    def __call__(self, img, mask):
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        shear_x = torch.empty(1).uniform_(-self.shear, self.shear).item()
        shear_y = torch.empty(1).uniform_(-self.shear, self.shear).item()

        # Apply affine transform to image
        img = T.functional.affine(img, angle=angle, translate=(0, 0), scale=1, shear=[shear_x, shear_y])
        
        # Apply the same affine transform to mask using nearest neighbor interpolation
        mask = T.functional.affine(mask, angle=angle, translate=(0, 0), scale=1, shear=[shear_x, shear_y], interpolation=T.InterpolationMode.NEAREST)
        
        return img, mask

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.p = flip_prob

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

