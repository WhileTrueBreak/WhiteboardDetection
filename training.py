from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torch_transforms import *
from roboflow import Roboflow
from dataset import Dataset
from PIL import Image
from glob import glob
import numpy as np
import colorsys
import network
import config
import torch
import os

console_cols, _ = os.get_terminal_size()

with open('key.txt') as f:
    rf = Roboflow(api_key=f.read().strip())
project = rf.workspace('whiletrue-xopuj').project(config.DATASET_NAME.lower())
version = project.version(config.DATASET_VERSION)
version.download('png-mask-semantic')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
val_transform = Compose([
    Resize(config.INPUT_SIZE),
    CenterCrop(config.INPUT_SIZE),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth'):
    try:
        print(f'Loading pretrained weights from models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
        model.load_state_dict(torch.load(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth', map_location=device, weights_only=True), strict=False)
    except Exception as e:
        print(e)
        print()
        continue_input = input('Could not load weights, continue? [y/n] ')
        if continue_input.lower() != 'y':
            exit()
model.to(device)

# load training dataset
dataset_path = f'{config.DATASET_NAME}-{config.DATASET_VERSION}/train'
training_dataset = Dataset()
val_dataset = Dataset()

all_files = [os.path.splitext(os.path.basename(path))[0] for path in glob(os.path.join(dataset_path, '*.jpg'))]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
for file in train_files:
    training_dataset.add_path(dataset_path, file)
for file in val_files:
    val_dataset.add_path(dataset_path, file)

print("loaded dataset")

# training
weights =  torch.tensor([1, 2.5, 0.5],dtype=torch.float32,device=device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

print("start training")
best_val_loss = float('inf')
for epoch in range(config.TRAINING_EPOCHS):
    print(f'Epoch {epoch} | LR: {optimizer.param_groups[0]["lr"]}')

    num_batches = training_dataset.create_batches(config.BATCH_SIZE)
    epoch_loss = 0
    model.train()
    for i in range(num_batches):
        imgs, masks = training_dataset.get_batch(i, training_transform, config.INPUT_SIZE)
        imgs = imgs.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(' '*console_cols, end='\r')
        print(f'Epoch {epoch+1}/{config.TRAINING_EPOCHS} | {i+1}/{num_batches}, Loss: {epoch_loss/(i+1)}', end='\r')
    model.eval()
    print(' '*console_cols, end='\r')
    print(f'Epoch {epoch+1}/{config.TRAINING_EPOCHS} | {num_batches}/{num_batches}, Loss: {epoch_loss/num_batches}')
    
    # Validation Pass
    val_batches = val_dataset.create_batches(config.BATCH_SIZE)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for j in range(val_batches):
            imgs, masks = val_dataset.get_batch(j, val_transform, config.INPUT_SIZE)
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            output = model(imgs)
            loss = criterion(output, masks)
            val_loss += loss.item()

            print(' ' * console_cols, end='\r')
            print(f'[Val] Epoch {epoch+1}/{config.TRAINING_EPOCHS} | {j+1}/{val_batches}, Loss: {val_loss/(j+1):.4f}', end='\r')
    avg_val_loss = val_loss/val_batches
    print(' ' * console_cols, end='\r')
    print(f'[Val] Epoch {epoch+1} complete, Avg Loss: {avg_val_loss:.4f}{"*" if avg_val_loss < best_val_loss else ""}')
    
    ####
    if avg_val_loss < best_val_loss:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
        best_val_loss = avg_val_loss
    scheduler.step(avg_val_loss)
