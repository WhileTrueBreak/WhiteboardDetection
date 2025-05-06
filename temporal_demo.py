from torchvision import transforms as T
import numpy as np
import colorsys
import network
import torch
import time
import cv2
import os

import matplotlib
matplotlib.use('TkAgg')

from place_solver import solve_mask_quad
import config

def overlay_image(image, pred):
    # calc colorized prediction
    colorized_pred = (color_mapping[pred,:]*255).astype('uint8')
    colorized_pred = cv2.resize(colorized_pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)

    # overlay the original image and the colorized prediction
    overlay = cv2.addWeighted(image, 0.5, colorized_pred, 0.5, 0)
    frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # mask the vertex image with whiteboard mask
    pred_cam_res = cv2.resize(pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)
    whiteboard_mask = (pred_cam_res == 1)
    shapes = solve_mask_quad(whiteboard_mask)
    for shape in shapes:
        cv2.drawContours(frame_out, [shape], -1, (0, 0, 255), 2)
    return frame_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
    if os.path.exists(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth'):
        print(f'Loading pretrained weights from models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth')
        model.load_state_dict(torch.load(f'models/cp_{config.MODEL_NAME}_{config.NUM_CLASSES}cls.pth', map_location=device, weights_only=True))
    model.to(device)

    transform = T.Compose([
        T.ToTensor(),   
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    save_period = 10
    pred_list = np.zeros((save_period, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))
    pred_prob_list = np.zeros((save_period, config.NUM_CLASSES, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))
    prev_pred = np.zeros((config.NUM_CLASSES, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))

    next_index = 0
    frame_index = 0

    start_time = time.time()
    pred_frames = []
    raw_frames = []
    timestamps = []

    color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        print('failed to open webcam')
        exit()

    with torch.no_grad():
        model = model.eval()
        while True:
            ret, ci = cap.read()
            if not ret:
                print('failed to read webcam')
                break
            ci_rgb = cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)
            cam_res = ci_rgb.shape
            
            input_img = cv2.resize(ci_rgb, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
            img_tensor = transform(input_img).unsqueeze(0).to(device)
            pred = model(img_tensor)
            pred_prob = torch.nn.functional.softmax(pred, dim=1)[0].cpu().numpy()
            pred = pred.max(1)[1].cpu().numpy()[0]
            pred = np.array(pred)

            # save to predlist
            pred_list[frame_index%save_period] = pred
            pred_prob_list[frame_index%save_period] = pred_prob

            prev_pred *= 0.8
            prev_pred += pred_prob

            # majority_vote = np.median(pred_list, axis=0).astype(np.int32)
            temporal_pred = np.argmax(prev_pred, axis=0)

            frame_out = overlay_image(ci_rgb, temporal_pred)

            cv2.imshow('Camera', frame_out)

            pred_frames.append(frame_out)
            raw_frames.append(ci)
            timestamps.append(time.time()-start_time)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                while os.path.exists(f'res/image{nextIndex}.png'): nextIndex += 1
                cv2.imwrite(f'res/image{nextIndex}.png', ci)
                print(f'Saved res/image{nextIndex}.png')
            if key == ord('q'):
                break
            frame_index += 1
    cv2.destroyAllWindows()

    fps = 30
    frame_time = 1/fps
    frame_durations = [t - s for s, t in zip(timestamps, timestamps[1:])]
    frame_durations.append(frame_durations[-1])
    pred_video = cv2.VideoWriter('output/pred_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (pred_frames[0].shape[1], pred_frames[0].shape[0]))
    raw_video = cv2.VideoWriter('output/raw_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (raw_frames[0].shape[1], raw_frames[0].shape[0]))
    next_frame = 1
    running = frame_durations[0]
    elapsed = 0
    while next_frame < len(frame_durations):
        pred_video.write(pred_frames[next_frame-1])
        raw_video.write(raw_frames[next_frame-1])
        elapsed += frame_time
        if elapsed >= running:
            running += frame_durations[next_frame]
            next_frame += 1
