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
depth_frame = frames.get_depth_frame()

# Convert images to numpy arrays
color_image = np.asanyarray(color_frame.get_data())

markersize = 200
markerImage22 = np.zeros((markersize,markersize), dtype = np.uint8)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        corners, ids = cv2.aruco.detectMarkers(color_frame, cv2.arucoDict)
        depthToMine = None
        
        try:
            for i in range(len(ids)):
                if(ids[i]) == 22:
                    print("found mine")
                    box = corners[i][0]
                    cX = int((box[0][0] + box[1][0]) / 2)
                    cY = int((box[1][1] + box[3][1]) / 2)
                    depthToMine = depth_frame.get_distance(cX,cY)

                    if cX >= 400:
                        motors = 5100
                        tango.setTarget(MOTORS,motors)
                    elif cX < 200:
                        motors = 6900
                        tango.setTarget(MOTORS,motors)
                    elif cX < 400 and cX > 200:
                        motors = 6000
                        tango.setTarget(MOTORS,motors)
                        body = 4900
                        tango.setTarget(BODY,body)
        except TypeError:
            body = 6000
            tango.setTarget(BODY,body)
            motors = 5100
            tango.setTarget(MOTORS,motors)

finally:    
    # Stop streaming
    pipeline.stop()