import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480

class RealsenseCamera:
    def __init__(self):
        # Configure the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.pc = rs.pointcloud()

        self.depth_history = np.zeros((1,HEIGHT,WIDTH),dtype=np.uint16)
        self.vertex_history = np.zeros((1,HEIGHT,WIDTH,3),dtype=np.float32)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        self.color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.color_intrinsics = self.color_profile.get_intrinsics()

        self.v, self.u = np.indices((HEIGHT, WIDTH))

    def depth2vertex(self, depth_image):
        z = depth_image*self.depth_scale
        x = (self.u-self.color_intrinsics.ppx)*z/self.color_intrinsics.fx
        y = (self.v-self.color_intrinsics.ppy)*z/self.color_intrinsics.fy

        vertices = np.stack((x, y, z), axis=-1)
        return vertices

    def uvd2xyz(self, u, v, d):
        z = d*self.depth_scale
        x = (u-self.color_intrinsics.ppx)*z/self.color_intrinsics.fx
        y = (v-self.color_intrinsics.ppy)*z/self.color_intrinsics.fy
        return np.array([x,y,z])

    def get_next_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise Exception("Could not retrieve frames")

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_history_new = np.roll(self.depth_history, 1, axis=0)
        self.depth_history = depth_history_new
        self.depth_history[0] = depth_image
        depth_image = np.average(self.depth_history, axis=0)
        depth_image[np.any(self.depth_history == 0, axis=0)] = 0

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_colormap[depth_image == 0] = 10 # Set to grey if depth_image is 0

        vertex_img = self.depth2vertex(depth_image)
        vertex_history_new = np.roll(self.vertex_history, 1, axis=0)
        self.vertex_history = vertex_history_new
        self.vertex_history[0] = vertex_img
        vertex_img = np.average(self.vertex_history, axis=0)
        mask = np.any(np.all(self.vertex_history == 0, axis=-1), axis=0)
        vertex_img[mask] = [0, 0, 0]

        return color_frame, depth_frame, color_image, depth_image, depth_colormap, vertex_img
    
    def setAveragingCount(self, count): 
        self.depth_history = np.zeros((count,HEIGHT,WIDTH),dtype=np.uint16)
        self.vertex_history = np.zeros((count,HEIGHT,WIDTH,3),dtype=np.float32)
    
    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    camera = RealsenseCamera()
    camera.setAveragingCount(10)
    while True:
        _,_,ci,_,dc,vi = camera.get_next_frame()

        vi[:,:,2] = 0

        cv2.imshow('xy', vi)

        # Display the color and depth colormap side by side
        # images = np.hstack((ci, dc))
        # cv2.namedWindow('Color and Depth', cv2.WINDOW_NORMAL)
        # cv2.imshow('Color and Depth', images)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    camera.stop()
