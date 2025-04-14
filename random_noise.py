from torchvision import transforms as T
import torch

class AddRandomNoise:
    def __init__(self, noise_prob=0.001):  # 0.1% of pixels
        self.noise_prob = noise_prob

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        mask = torch.rand_like(img[0]) < self.noise_prob
        noise = torch.rand_like(img) * mask.unsqueeze(0)
        img = torch.clamp(img + noise, 0.0, 1.0)
        return img