# capture.py
import numpy as np
import cv2


video = cv2.VideoCapture(0)

ret, frame = video.read()
image1 = frame.copy()
avg1 = np.float32(frame)
t = 100
t1=200


while(video.isOpened()):
    ret, frame = video.read()
    image1 = frame.copy()

    bright = cv2.convertScaleAbs(frame, 1, 1.25)
    blur = cv2.GaussianBlur(bright, (7,7), 0)

    accumulated = cv2.accumulateWeighted(blur, avg1, .01)

    conversion = cv2.convertScaleAbs(accumulated)

    image1 = image1.astype(np.uint8)
    conversion = conversion.astype(np.uint8)

    diff = cv2.absdiff(image1,conversion)

    greyscale = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    cv2.imshow("Greyscale", greyscale)

    (t, threshold) = cv2.threshold(greyscale, 100,255,cv2.THRESH_BINARY)
    thresholdblur = cv2.GaussianBlur(threshold, (7,7), 0)

    (t1, threshold2) = cv2.threshold(thresholdblur, 200,255,cv2.THRESH_BINARY,)

    cv2.imshow("threshold", threshold2)

    contour, heir = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white = np.ones((525,650,3),np.uint8)
    white = 255 * white
    draw = cv2.drawContours(white,contour,-1,(0,0,1),3)
    cv2.imshow("contour", draw)

    for i in contour:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(image1, (x,y), (x+w, y+h), (0,0,255), 3)
    cv2.imshow("boxes",image1)

    if cv2.waitKey(1) == 27:
        break



video.release()

cv2.destroyAllWindows()