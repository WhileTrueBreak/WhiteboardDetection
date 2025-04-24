
from torchvision import transforms as T
from place_solver import solve_mask_quad
from glob import glob
import numpy as np
import matplotlib
import colorsys
import network
import config
import torch
import cv2
import os

matplotlib.use('TkAgg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToTensor(),   
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth'):
    print(f'Loading pretrained weights from models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
    model.load_state_dict(torch.load(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth', map_location=device, weights_only=True))
model.to(device)


color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1)
                          for i in range(config.NUM_CLASSES)])
image_paths = glob("./res/*.*")
if not image_paths:
    print("No images found in the './res' folder.")
    exit()

os.makedirs('./output', exist_ok=True)

with torch.no_grad():
    model = model.eval()
    for img_path in image_paths:
        # Load image using OpenCV
        ci = cv2.imread(img_path)
        if ci is None:
            print(f"Failed to load image: {img_path}")
            continue

        ci_rgb = cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)
        cam_res = ci_rgb.shape

        input_img = cv2.resize(ci_rgb, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
        img_tensor = transform(input_img).unsqueeze(0).to(device)
        pred = model(img_tensor).max(1)[1].cpu().numpy()[0]
        pred = np.array(pred)

        color_mapping = np.array([colorsys.hsv_to_rgb(i/2, 1, 1) for i in range(2)])
        pred_full_mask = (pred > 0).astype(int)

        kernel = np.ones((11,11), np.uint8)
        pred_full_mask = cv2.morphologyEx(pred_full_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        pred_full_mask = cv2.morphologyEx(pred_full_mask, cv2.MORPH_CLOSE, kernel)

        colorized_pred = (color_mapping[pred_full_mask,:]*255).astype('uint8')
        colorized_pred = cv2.resize(colorized_pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay the original image and the colorized prediction
        overlay = cv2.addWeighted(ci_rgb, 0.5, colorized_pred, 0.5, 0)
        frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # mask the vertex image with whiteboard mask
        pred_cam_res = cv2.resize(pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)
        whiteboard_mask = (pred_cam_res == 1)
        # whiteboard_full_mask = (pred_cam_res > 0)
        whiteboard_full_mask = cv2.resize(pred_full_mask, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)

        shapes = solve_mask_quad(whiteboard_full_mask)
        for shape in shapes:
            cv2.drawContours(frame_out, [shape], -1, (0, 0, 255), 2)

        filename = os.path.basename(img_path)
        out_path = os.path.join('./output', filename)
        cv2.imwrite(out_path, frame_out)
        print(f'Saved predicted output to {out_path}')
