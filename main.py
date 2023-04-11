# capture.py
import numpy as np
import cv2


lowH = 0
lowS = 0
lowV = 0
highH = 179
highS = 255
highV = 255
def mouseHSV(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = newcol[y,x,0]
        colorsG = newcol[y,x,1]
        colorsR = newcol[y,x,2]
        colors = newcol[y,x]
        print("Red: ", colorsR)
        hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
        hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
        print ("HSV : " ,hsv)

def changelowH(value):
    global lowH
    lowH = value

def changelowS(value):
    global lowS
    lowS = value

def changelowV(value):
    global lowH
    lowH = value

def changehighH(value):
    global highH
    highH = value

def changehighS(value):
    global highS
    highS = value

def changehighV(value):
    global highV
    highV = value

cv2.namedWindow('MotionTracking')
cv2.setMouseCallback('MotionTracking',mouseHSV)

video = cv2.VideoCapture(0)

cv2.createTrackbar('minH','MotionTracking', 0,178, changelowH)
cv2.createTrackbar('maxH','MotionTracking', 1,179, changehighH)
cv2.createTrackbar('minS','MotionTracking', 0,254, changelowS)
cv2.createTrackbar('maxS','MotionTracking', 1,255, changehighS)
cv2.createTrackbar('minV','MotionTracking', 0,254, changelowV)
cv2.createTrackbar('maxV','MotionTracking', 1,255, changehighV)


kernel = np.ones((5, 5), np.uint8)

while(video.isOpened()):
    ret, frame = video.read()

    minArray = np.array([lowH,lowS,lowV])
    maxArray = np.array([highH,highS,highV])
    #Convert to HSV
    newcol = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Display
    cv2.imshow('OriginalImage',frame)
    threshold = cv2.inRange(newcol, minArray, maxArray)
    dilate = cv2.dilate(threshold,kernel,iterations=1)
    erode = cv2.erode(dilate,kernel,iterations=1)
    cv2.imshow('MotionTracking', erode)
    if cv2.waitKey(1) == 27:
        break

video.release()

cv2.destroyAllWindows()
