from torchvision import transforms as T
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
transform = T.Compose([
    T.Resize(config.INPUT_SIZE),
    T.CenterCrop(config.INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth'):
    try:
        print(f'Loading pretrained weights from {config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
        model.load_state_dict(torch.load(f'{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth', map_location=device, weights_only=True))
    except Exception as e:
        print(e)
        print()
        continue_input = input('Could not load weights, continue? [y/n] ')
        if continue_input.lower() != 'y':
            exit()

# load training dataset
dataset = Dataset()
dataset_path = f'{config.DATASET_NAME}-{config.DATASET_VERSION}/train'
training_inputs = glob(os.path.join(dataset_path, '*.jpg'))
for training_input in training_inputs:
    data = os.path.splitext(os.path.basename(training_input))[0]
    dataset.add_path(dataset_path, data)

print("loaded dataset")

# training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("start training")

for epoch in range(config.TRAINING_EPOCHS):
    model.train()
    num_batches = dataset.create_batches(config.BATCH_SIZE)

    epoch_loss = 0
    for i in range(num_batches):
        imgs, masks = dataset.get_batch(i, transform, config.INPUT_SIZE)
        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
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
    torch.save(model.state_dict(), f'{config.MODEL_NAME}.pth')

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
