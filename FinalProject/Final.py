# Python code for Multiple Color Detection
  
import pyrealsense2 as rs
import numpy as np
import cv2
from pupil_apriltags import Detector
from maestro import Controller                                                    

MOTORS = 1
TURN = 2
BODY = 0

tango = Controller()
motors = 6000
turns = 6000
body = 6000

#tag detector
at_detector = Detector(
    families = "tag36h11",
    nthreads = 1,
    quad_decimate=1.0,
    quad_sigma = 0.0,
    refine_edges = 1,
    decode_sharpening = 0.25,
    debug=0
)


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

color_to_save = ""

def orientation(id,say):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    detected = at_detector.detect(gray)
    if detected:
        tag = detected[0]
        if tag.tag_id == id:
            oriented = True
            #Stop Rotation
        (cX,cY) = int (int(tag.center[0]),int(tag.center[1]))
        depth = depth_frame.get_distance(cX,cY)
        distance = int(0.01 / depth)
        distanceText = "distance from tag" + str(tag.tag_id) + ": " + str(round(depth,2))
        cv2.putText(color_image,distanceText,(20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        x1,y1 = int(tag.corners[0][0]), int(tag.corners[0][1])
        x2,y2 = int(tag.corners[2][0]), int(tag.corners[2][1])
        cv2.rectangle(color_image, (x1,y1), (x2,y2), (0,255,0), 2)
        ##Move until distance is within threshold

        speak(say)

def findFace():
    #Move head of robot up if needed to find a face.
    #Slowly Spin until face is found
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 5,)
    for (x,y,w,h) in faces:
        cv2.rectangle(color_image,(x,y),(x+w,y+h),(255,0,0),2)
    #Make Robot slowly approach face until it is within .6 meters of face.
    speak("Ice Please :)")

def speak(say):
    print(say)
    print(say)
    print(say)


def findColor():
    # Convert Image to Image HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Set range for red color and 
    # define mask
    yellow_lower = np.array([33, 80, 56], np.uint8)
    yellow_upper = np.array([55, 125, 197], np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

  
    # Set range for green color and 
    # define mask
    green_lower = np.array([61, 120, 101], np.uint8)
    green_upper = np.array([66, 155,212], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
  
    # Set range for pink color and
    # define mask
    pink_lower = np.array([120, 111, 126], np.uint8)
    pink_upper = np.array([179, 255, 255], np.uint8)
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")
      
    # For red color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(color_image, color_image, mask = yellow_mask)
      
    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(color_image, color_image,mask = green_mask)
      
    # For blue color
    pink_mask = cv2.dilate(pink_mask, kernel)
    res_pink = cv2.bitwise_and(color_image, color_image,mask = pink_mask)
   
    # Creating contour to track red color
    contours = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            color_to_save = "yellow"
            x, y, w, h = cv2.boundingRect(contour)
            color_image = cv2.rectangle(color_image, (x, y), (x + w, y + h), (51, 255, 255), 2)
                

    # Creating contour to track green color
    contours = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            color_to_save = "green"
            x, y, w, h = cv2.boundingRect(contour)
            color_image = cv2.rectangle(color_image, (x, y), (x + w, y + h),(0, 255, 0), 2)
                

    contours = cv2.findContours(pink_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            color_to_save = "pink"
            x, y, w, h = cv2.boundingRect(contour)
            color_image = cv2.rectangle(color_image, (x, y),(x + w, y + h),(255, 77, 255), 2)
    
    speak("Color of Ice: " + color_to_save)
    return color_to_save

def findGoal(color_to_save):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), "uint8")
    if color_to_save == "yellow":
        yellow_lower = np.array([33, 80, 56], np.uint8)
        yellow_upper = np.array([55, 125, 197], np.uint8)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        yellow_mask = cv2.dilate(yellow_mask, kernel)
        res_yellow = cv2.bitwise_and(color_image, color_image, mask = yellow_mask)

        contours = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                color_image = cv2.rectangle(color_image, (x, y), (x + w, y + h), (51, 255, 255), 2)

        ##Go to Point
    elif color_to_save == "green":
        green_lower = np.array([61, 120, 101], np.uint8)
        green_upper = np.array([66, 155,212], np.uint8)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(color_image, color_image,mask = green_mask)

        contours = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                color_image = cv2.rectangle(color_image, (x, y), (x + w, y + h),(0, 255, 0), 2)
        
        ##Go to point

    elif color_to_save == "pink":
        pink_lower = np.array([120, 111, 126], np.uint8)
        pink_upper = np.array([179, 255, 255], np.uint8)
        pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)

        pink_mask = cv2.dilate(pink_mask, kernel)
        res_pink = cv2.bitwise_and(color_image, color_image,mask = pink_mask)

        contours = cv2.findContours(pink_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                color_image = cv2.rectangle(color_image, (x, y),(x + w, y + h),(255, 77, 255), 2)

        ##Go to point


def stopMovement():
    body = 6000
    tango.setTarget(BODY, body)
    motors = 6000
    tango.setTarget(MOTORS, motors)



try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_colormap_dim = color_image.shape

        # Display Image and Mask
        cv2.imshow("Image", color_image)

        orientation(1, "Entering Mining Area")
        findFace()
        findColor()
        orientation(0,"Entering Goal Area")
        findGoal(color_to_save)
        stopMovement()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
finally:

    # Stop streaming
    pipeline.stop()