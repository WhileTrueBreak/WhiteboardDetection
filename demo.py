
from torchvision import transforms as T
from realsense import RealsenseCamera
from place_solver import solve_plane, solve_mask_quad, solve_depth, uv2xyz, plane2uv, uv2planeuv
import matplotlib.pyplot as plt
from roboflow import Roboflow
from dataset import Dataset
from PIL import Image
from glob import glob
import numpy as np
import matplotlib
import threading
import colorsys
import network
import config
import torch
import cv2
import os

def update_images():
    global ci, di, vi, camera, is_cam_updating
    camera.setAveragingCount(7)
    while is_cam_updating:
        cf, df, ci, di, dc, vi = camera.get_next_frame()
    camera.stop()

camera = RealsenseCamera()
matplotlib.use('TkAgg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToTensor(),   
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'{config.MODEL_NAME}.pth'):
    print(f'Loading pretrained weights from {config.MODEL_NAME}.pth')
    model.load_state_dict(torch.load(f'{config.MODEL_NAME}.pth', map_location=device, weights_only=True))
model.to(device)

nextIndex = 0
color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])

ci = None
di = None
vi = None
is_cam_updating = True

cam_update_thread = threading.Thread(target=update_images)
cam_update_thread.start()


plt_x = np.linspace(-2.5,2.5,2)
plt_y = np.linspace(-2.5,2.5,2)
plt_x,plt_y = np.meshgrid(plt_x,plt_y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.set_xlim(0,5)
ax.set_ylim(-2.5,2.5)
ax.set_zlim(-2.5,2.5)
ax.invert_xaxis()
ax.invert_zaxis()

surface_artist = None
scatter_artists = []

A_coeff_arr = []
B_coeff_arr = []
C_coeff_arr = []

with torch.no_grad():
    model = model.eval()
    while True:
        if ci is None or vi is None or di is None: continue
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
        whiteboard_full_mask = (pred_cam_res > 0)
        whiteboard_vertices = vi[whiteboard_mask]
        whiteboard_vertices = whiteboard_vertices[~(whiteboard_vertices==0).all(axis=1)]

        dA, dB, dC = solve_depth(whiteboard_mask, di)

        shapes = solve_mask_quad(whiteboard_mask)
        for shape in shapes:
            cv2.drawContours(frame_out, [shape], -1, (0, 0, 255), 2)

        # solve the plane
        A, B, C = solve_plane(whiteboard_vertices)
        print(f'{A:.2f}x{"+" if B>0 else ""}{B:.2f}y{"+" if C>0 else ""}{C:.2f}=z')

        A_coeff_arr.append(A)
        B_coeff_arr.append(B)
        C_coeff_arr.append(C)

        # visualize key points
        if surface_artist is not None:
            surface_artist.remove()
        for scatter_artist in scatter_artists:
            scatter_artist.remove()
        scatter_artists = []  # Reset the list
        plt_z = (A*plt_x + B*plt_y + C)
        surface_artist = ax.plot_surface(plt_z, plt_x, plt_y, alpha=0.5)
        shape_u = []
        shape_v = []
        for shape in shapes:
            for vertex in shape:
                u, v = vertex[0]
                shape_u.append(u)
                shape_v.append(v)
                x, y, z = uv2xyz(u, v, dA, dB, dC, camera)
                scatter_artist = ax.scatter(z, x, y, c='r', marker='o')
                scatter_artists.append(scatter_artist)
        ax.figure.canvas.flush_events()
        plt.pause(0.0001)

        u = np.linspace(0, camera.color_intrinsics.width-1, int(camera.color_intrinsics.width/4), dtype=int)
        v = np.linspace(0, camera.color_intrinsics.height-1, int(camera.color_intrinsics.height/4), dtype=int)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        plane_u, plane_v = uv2planeuv(u, v, dA, dB, dC, camera, A, B, C)
        plane_u /= 1
        plane_v /= 1
        vis_plane = np.zeros((480, 640, 3), np.uint8)
        for i in range(plane_u.shape[0]):
            color = tuple(int(c) for c in ci[v[i], u[i]])
            cv2.circle(vis_plane, (int(plane_u[i]), int(plane_v[i])), 2, tuple(color), -1)
        shape_u = np.array(shape_u)
        shape_v = np.array(shape_v)
        plane_u, plane_v = uv2planeuv(shape_u, shape_v, dA, dB, dC, camera, A, B, C)
        plane_u /= 1
        plane_v /= 1
        for i in range(plane_u.shape[0]):
            cv2.circle(vis_plane, (int(plane_u[i]), int(plane_v[i])), 2, (0, 0, 255), -1)
        cv2.imshow('Plane', vis_plane)

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

A_coeff_arr = np.array(A_coeff_arr)[len(A_coeff_arr)//2:]
B_coeff_arr = np.array(B_coeff_arr)[len(B_coeff_arr)//2:]
C_coeff_arr = np.array(C_coeff_arr)[len(C_coeff_arr)//2:]
x = np.arange(len(A_coeff_arr))

# plt.close('all')
# plt.clf()
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(x, A_coeff_arr, label='A')
# ax[0].axhline(np.average(A_coeff_arr), label='A avg', color='red', linestyle='--')
# ax[1].plot(x, B_coeff_arr, label='B')
# ax[1].axhline(np.average(B_coeff_arr), label='B avg', color='red', linestyle='--')
# ax[2].plot(x, C_coeff_arr, label='C')
# ax[2].axhline(np.average(C_coeff_arr), label='C avg', color='red', linestyle='--')
# for a in ax:
#     a.legend()
# plt.show()

plane_offset_dist = np.abs(C_coeff_arr)/np.sqrt(A_coeff_arr*A_coeff_arr + B_coeff_arr*B_coeff_arr + 1)
plane_offset_average = np.average(plane_offset_dist)
plane_angle = np.arccos(1 / np.sqrt(A_coeff_arr**2+B_coeff_arr**2+1))
plane_angle_average = np.average(plane_angle)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x, plane_offset_dist, label='offset')
ax[0].axhline(plane_offset_average, label='offset avg', color='red', linestyle='--')
ax[0].axhline(plane_offset_average+0.05, label='offset avg +5cm', color='green', linestyle='--')
ax[0].axhline(plane_offset_average-0.05, label='offset avg -5cm', color='green', linestyle='--')
ax[0].legend()
ax[0].set_xlabel('Frame')
ax[0].set_ylabel('Dist (M)')
ax[0].set_title('Whiteboard Plane Offset')

ax[1].plot(x, plane_angle, label='angle')
ax[1].axhline(plane_angle_average, label='angle avg', color='red', linestyle='--')
ax[1].axhline(plane_angle_average+0.087, label='angle avg +5°', color='green', linestyle='--')
ax[1].axhline(plane_angle_average-0.087, label='angle avg -5°', color='green', linestyle='--')
ax[1].legend()
ax[1].set_xlabel('Frame')
ax[1].set_ylabel('Angle (Rad)')
ax[1].set_title('Whiteboard Plane Angle')
plt.tight_layout()
plt.show()
