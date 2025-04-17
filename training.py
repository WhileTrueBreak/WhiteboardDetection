from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from random_noise import AddRandomNoise
import matplotlib.pyplot as plt
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

training_transform = T.Compose([
    T.Resize(config.INPUT_SIZE),
    T.CenterCrop(config.INPUT_SIZE),
    T.ColorJitter(
        brightness=0.25,
        contrast=0.15,
        saturation=0.3,
        hue=0.083
    ),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
    T.ToTensor(),
    AddRandomNoise(noise_prob=0.001),  # 0.1% noise
    T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    T.Normalize(mean=(0.485, 0.456, 0.406),

                std=(0.229, 0.224, 0.225))
])
val_transform = T.Compose([
    T.Resize(config.INPUT_SIZE),
    T.CenterCrop(config.INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
model = network.modeling.deeplabv3plus_resnet50(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)

print("start training")
best_val_loss = float('inf')
for epoch in range(config.TRAINING_EPOCHS):
    print(f'Epoch {epoch} | LR: {optimizer.param_groups[0]["lr"]}')

    # Training Pass
    for param in model.backbone.parameters():
        param.requires_grad = False
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
    print(f'[Val] Epoch {epoch+1} complete, Avg Loss: {avg_val_loss:.4f}')
    
    ####
    if avg_val_loss < best_val_loss:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
        best_val_loss = avg_val_loss
    scheduler.step(avg_val_loss)

color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])

with torch.no_grad():
    model = model.eval()
    img = Image.open('test.jpg').convert('RGB')
    img_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)

    pred = model(img_tensor).max(1)[1].cpu().numpy()[0]
    pred = np.array(pred)
    colorized_pred = (color_mapping[pred,:]*255).astype('uint8')
    colorized_pred = Image.fromarray(colorized_pred)
    colorized_pred = colorized_pred.resize(img_size)

    # Overlay the original image and the colorized prediction
    overlay = Image.blend(img, colorized_pred, alpha=0.5)
    overlay.save('overlay.png')
