import pyrealsense2 as rs
import numpy as np
import cv2
from maestro import Controller                                                     

MOTORS = 1
TURN = 2
BODY = 0

tango = Controller()
motors = 6000
turns = 6000
body = 6000

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


frames = pipeline.wait_for_frames()
# Align the depth frame to color frame
aligned_frames = align.process(frames)
color_frame = frames.get_color_frame()

# Convert images to numpy arrays
color_image = np.asanyarray(color_frame.get_data())



try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        color_colormap_dim = color_image.shape

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        orange_lower = np.array([0, 100, 20], np.uint8)
        orange_upper = np.array([60, 255, 255], np.uint8)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

        cv2.namedWindow('OrangeMask', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('OrangeMask', orange_mask)
        cv2.waitKey(1)
finally:

    # Stop streaming
    pipeline.stop()