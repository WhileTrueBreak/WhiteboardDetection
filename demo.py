
from torchvision import transforms as T
from realsense import RealsenseCamera
from place_solver import solve_plane
import matplotlib.pyplot as plt
from roboflow import Roboflow
from dataset import Dataset
from PIL import Image
from glob import glob
import numpy as np
import threading
import colorsys
import network
import config
import torch
import cv2
import os

def update_images():
    global ci, vi, is_cam_updating
    camera = RealsenseCamera()
    camera.setAveragingCount(7)
    while is_cam_updating:
        cf, df, ci, di, dc, vi = camera.get_next_frame()
    camera.stop()

console_cols, _ = os.get_terminal_size()
nextIndex = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'{config.MODEL_NAME}.pth'):
    print(f'Loading pretrained weights from {config.MODEL_NAME}.pth')
    model.load_state_dict(torch.load(f'{config.MODEL_NAME}.pth', map_location=device, weights_only=True))

color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])

ci = None
vi = None
is_cam_updating = True

cam_update_thread = threading.Thread(target=update_images)
cam_update_thread.start()


plt_x = np.linspace(-10,10,100)
plt_y = np.linspace(-10,10,100)
plt_x,plt_y = np.meshgrid(plt_x,plt_y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

A_coeff_arr = []
B_coeff_arr = []
C_coeff_arr = []

with torch.no_grad():
    model = model.eval()
    while True:
        if ci is None or vi is None: continue
        # cf, df, ci, di, dc, vi = camera.get_next_frame()
        ci_rgb = cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)
        cam_res = ci_rgb.shape

        input_img = cv2.resize(ci_rgb, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
        img_tensor = transform(input_img).unsqueeze(0).to(device)
        pred = model(img_tensor).max(1)[1].cpu().numpy()[0]
        pred = np.array(pred)

        colorized_pred = (color_mapping[pred,:]*255).astype('uint8')
        colorized_pred = cv2.resize(colorized_pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay the original image and the colorized prediction
        overlay = cv2.addWeighted(ci_rgb, 0.5, colorized_pred, 0.5, 0)
        frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # mask the vertex image with whiteboard mask
        pred_cam_res = cv2.resize(pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)
        whiteboard_mask = (pred_cam_res == 1)
        whiteboard_vertices = vi[whiteboard_mask]
        whiteboard_vertices = whiteboard_vertices[~(whiteboard_vertices==0).all(axis=1)]

        # solve the plane
        A, B, C = solve_plane(whiteboard_vertices)
        print(f'{A:.2f}x+{B:.2f}y+{C:.2f}=z')

        A_coeff_arr.append(A)
        B_coeff_arr.append(B)
        C_coeff_arr.append(C)

        ax.cla()
        plt_z = (A*plt_x + B*plt_y + C)
        ax.plot_surface(plt_z, plt_x, plt_y, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        plt.pause(0.001)

        cv2.imshow('Camera', frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            while os.path.exists(f'res/image{nextIndex}.png'): nextIndex += 1
            cv2.imwrite(f'res/image{nextIndex}.png', ci)
            print(f'Saved res/image{nextIndex}.png')
        if key == ord('q'):
            break

is_cam_updating = False
cam_update_thread.join()
cv2.destroyAllWindows()

x = np.arange(len(A_coeff_arr))
A_coeff_arr = np.array(A_coeff_arr)
B_coeff_arr = np.array(B_coeff_arr)
C_coeff_arr = np.array(C_coeff_arr)

plane_offset_dist = np.abs(C_coeff_arr)/np.sqrt(A_coeff_arr*A_coeff_arr + B_coeff_arr*B_coeff_arr + 1)
plane_offset_average = np.average(plane_offset_dist)

plt.clf()
plt.plot(x, plane_offset_dist, label='offset')
plt.axhline(plane_offset_average, label='offset avg', color='red', linestyle='--')
plt.axhline(plane_offset_average+0.05, label='offset avg +5cm', color='green', linestyle='--')
plt.axhline(plane_offset_average-0.05, label='offset avg -5cm', color='green', linestyle='--')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Dist (M)')
plt.title('Whiteboard Plane Offset')
plt.show()
