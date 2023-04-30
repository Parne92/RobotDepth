import pyrealsense2 as rs
import numpy as np
import cv2
import time
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

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

inMiningArea = False
foundFace = False
savedColor = None


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
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        if(inMiningArea == False):
            orange_lower = np.array([0, 50, 20], np.uint8)
            orange_lower = np.array([0, 200, 20], np.uint8)
            orange_upper = np.array([60, 255, 255], np.uint8)
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            Moments = cv2.moments(orange_mask)
            if Moments["m00"] != 0:
                cX = int(Moments["m10"] / Moments["m00"])
                cY = int(Moments["m01"] / Moments["m00"])
            else:
                cX, cY = 0,0
            cv2.circle(color_image, (cX, cY), 5, (0, 165, 255), -1)

            distance = depth_frame.get_distance(cX,cY)

        
            if (cX > 370):
                motors -= 200
                if(motors < 5000):
                    motors = 5000
                    tango.setTarget(MOTORS, motors)
            elif (cX < 270):
                motors += 200
                if(motors > 7000):
                    motors = 7000
                    tango.setTarget(MOTORS, motors)
            else:
                motors = 6000
                tango.setTarget(MOTORS, motors)

            if(distance > 1.5):
                motors = 6000
                tango.setTarget(MOTORS,motors)
                body = 5200            
                tango.setTarget(BODY,body)
            else:
                body = 6000
                tango.setTarget(BODY,body)
                print("Entered Mining Area!")
                inMiningArea = True
        if(inMiningArea == True and foundFace == False):
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5,)

            if(faces == 0):
                motors = 5400
                tango.setTarget(MOTORS,motors)
            elif(faces != 0):
                for (x,y,w,h) in faces:
                    cv2.rectangle(color_image,(x,y),(x+w,y+h),(255,0,0),2)
                cX = (x + (w/2))
                cY = (y + (h/2))

                distance = depth_frame.get_distance(cX,cY)

                if(distance > 1.5):
                    motors = 6000
                    tango.setTarget(MOTORS,motors)
                    body = 5200            
                    tango.setTarget(BODY,body)
                else:
                    body = 6000
                    tango.setTarget(BODY,body)
                    print("Moved to Face!")
                    foundFace = True
        if(inMiningArea == True and foundFace == True and savedColor == None):
            print("AWAITING ICE")
            time.sleep(5)

            while(savedColor == None):
                yellow_lower = np.array([33, 80, 56], np.uint8)
                yellow_upper = np.array([55, 125, 197], np.uint8)
                yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
  
                green_lower = np.array([61, 120, 101], np.uint8)
                green_upper = np.array([66, 155,212], np.uint8)
                green_mask = cv2.inRange(hsv, green_lower, green_upper)
  
                pink_lower = np.array([120, 111, 126], np.uint8)
                pink_upper = np.array([179, 255, 255], np.uint8)
                pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

                kernel = np.ones((5, 5), "uint8")
      
                yellow_mask = cv2.dilate(yellow_mask, kernel)
                res_yellow = cv2.bitwise_and(color_image, color_image, mask = yellow_mask)
      
                green_mask = cv2.dilate(green_mask, kernel)
                res_green = cv2.bitwise_and(color_image, color_image, mask = green_mask)
      
                pink_mask = cv2.dilate(pink_mask, kernel)
                res_pink = cv2.bitwise_and(color_image, color_image, mask = pink_mask)
   
                contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area > 1000):
                        savedColor = "yellow"
                        x, y, w, h = cv2.boundingRect(contour)
                        color_image = cv2.rectangle(color_image, (x, y), 
                                       (x + w, y + h), 
                                       (51, 255, 255), 2)
                

                # Creating contour to track green color
                contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area > 1000):
                        savedColor = "green"
                        x, y, w, h = cv2.boundingRect(contour)
                        color_image = cv2.rectangle(color_image, (x, y), 
                                       (x + w, y + h),
                                       (0, 255, 0), 2)
                

                contours, hierarchy = cv2.findContours(pink_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
            
                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if(area > 1000):
                        savedColor = "pink"
                        x, y, w, h = cv2.boundingRect(contour)
                        color_image = cv2.rectangle(color_image, (x, y),
                                       (x + w, y + h),
                                       (255, 77, 255), 2)
            print("COLOR DETECTED: " + savedColor)  


finally:
    # Stop streaming
    pipeline.stop()