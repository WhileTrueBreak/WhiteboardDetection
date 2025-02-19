
from torchvision import transforms as T
from realsense import RealsenseCamera
from place_solver import solve_plane, solve_mask_quad, solve_depth, uv2xyz, camera2planeuv, plane2camerauv
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
import easyocr
import config
import torch
import cv2
import os

def update_images():
    global ci, di, vi, camera, is_cam_updating
    camera.setAveragingCount(5)
    while is_cam_updating:
        cf, df, ci, di, dc, vi = camera.get_next_frame()
    camera.stop()

# init realsense camera
camera = RealsenseCamera()

# init matplotlib
matplotlib.use('TkAgg')
# init plots/plot varibles
A_coeff_arr = []
B_coeff_arr = []
C_coeff_arr = []

#init detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToTensor(),   
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
if os.path.exists(f'{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth'):
    print(f'Loading pretrained weights from {config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
    model.load_state_dict(torch.load(f'{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth', map_location=device, weights_only=True))
else:
    print(f'Model {config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth does not exist')
    exit()
model.to(device)

# init easyocr
ocr_reader = easyocr.Reader(['en'])

# global variables
nextIndex = 0
color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])

ci = None
di = None
vi = None
is_cam_updating = True

# start getting camera feed
cam_update_thread = threading.Thread(target=update_images)
cam_update_thread.start()

with torch.no_grad():
    model = model.eval()
    while True:
        # continue if images exist
        if ci is None or vi is None or di is None: continue
        ci_rgb = cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)
        cam_res = ci_rgb.shape

        # run detection model on color image
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


        shapes = solve_mask_quad(whiteboard_mask)
        shape_u = []
        shape_v = []
        for shape in shapes:
            cv2.drawContours(frame_out, [shape], -1, (0, 0, 255), 2)
            for vertex in shape:
                u, v = vertex[0]
                shape_u.append(u)
                shape_v.append(v)
        shape_u = np.array(shape_u)
        shape_v = np.array(shape_v)

        # solve the plane
        A, B, C = solve_plane(whiteboard_vertices)
        # print(f'{A:.2f}x{"+" if B>0 else ""}{B:.2f}y{"+" if C>0 else ""}{C:.2f}=z')

        A_coeff_arr.append(A)
        B_coeff_arr.append(B)
        C_coeff_arr.append(C)

        print('----')
        # render adjusted whiteboard if detected
        plane_content_centers = []
        if shape_u.shape[0] > 0 and shape_v.shape[0] > 0:
            # get uvs in plane frame
            shape_u2, shape_v2 = camera2planeuv(shape_u, shape_v, A, B, C, camera)
            min_u2, max_u2 = np.min(shape_u2), np.max(shape_u2)
            min_v2, max_v2 = np.min(shape_v2), np.max(shape_v2)
            u2_offset = int(np.floor(min_u2))
            v2_offset = int(np.floor(min_v2))
            u2, v2 = np.meshgrid(
                np.arange(u2_offset ,np.floor(max_u2)+1, 1, dtype=int), 
                np.arange(v2_offset ,np.floor(max_v2)+1, 1, dtype=int)
            )
            vis_plane = np.zeros((u2.shape[0], u2.shape[1], 3), np.uint8) # offset plane space image
            u2f = u2.flatten() # plane space u
            v2f = v2.flatten() # plane space v
            # map plane frame uvs back to camera frame
            u1, v1 = plane2camerauv(u2f, v2f, A, B, C, camera) # camera space uvs
            u1 = u1.astype(int)
            v1 = v1.astype(int)
            # update plane frame image with colors from camera frame
            valid_mask = (u1 >= 0) & (u1 < ci.shape[1]) & (v1 >= 0) & (v1 < ci.shape[0])
            vis_plane[v2f[valid_mask]-v2_offset,u2f[valid_mask]-u2_offset] = ci[v1[valid_mask],u1[valid_mask]]

            # read whiteboard
            scale = 800/vis_plane.shape[1]
            vis_plane = cv2.resize(vis_plane, (int(vis_plane.shape[1]*scale), int(vis_plane.shape[0]*scale)))
            vis_plane = cv2.flip(vis_plane, 1)
            whiteboard_text_results = ocr_reader.readtext(vis_plane)
            # results are in plane frame
            # can be mapped to camera frame later
            # label whiteboard
            plane_content_centers.append((u2_offset,v2_offset))
            plane_content_centers.append((max_u2-min_u2+u2_offset,max_v2-min_v2+v2_offset))
            for (bbox, text, prob) in whiteboard_text_results:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                center_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4
                center_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4
                plane_content_centers.append(((vis_plane.shape[1]-center_x)/scale+u2_offset, center_y/scale+v2_offset))
                cv2.circle(vis_plane, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)
                cv2.line(vis_plane, tl, tr, (0, 0, 255), 1)
                cv2.line(vis_plane, tr, br, (0, 0, 255), 1)
                cv2.line(vis_plane, br, bl, (0, 0, 255), 1)
                cv2.line(vis_plane, bl, tl, (0, 0, 255), 1)
            cv2.imshow('Whiteboard', vis_plane)
        if len(plane_content_centers) >= 1:
            plane_content_centers = np.array(plane_content_centers)
            ccc_u, ccc_v = plane2camerauv(plane_content_centers[:,0], plane_content_centers[:,1], A, B, C, camera)
            for (x, y) in zip(ccc_u, ccc_v):
                cv2.circle(frame_out, (int(x), int(y)), 5, (255, 0, 0), -1)

        cv2.imshow('Camera', frame_out)

        # key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            while os.path.exists(f'res/image{nextIndex}.png'): nextIndex += 1
            cv2.imwrite(f'res/image{nextIndex}.png', ci)
            print(f'Saved res/image{nextIndex}.png')
        if key == ord('q'):
            break

# clean up
is_cam_updating = False
cam_update_thread.join()
cv2.destroyAllWindows()

# show plane stability
A_coeff_arr = np.array(A_coeff_arr)#[len(A_coeff_arr)//2:]
B_coeff_arr = np.array(B_coeff_arr)#[len(B_coeff_arr)//2:]
C_coeff_arr = np.array(C_coeff_arr)#[len(C_coeff_arr)//2:]
x = np.arange(len(A_coeff_arr))

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

