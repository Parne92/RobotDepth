
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
#recognizer = cv.face.LBPHFaceRecognizer_create()
#recognizer.read("traner.yml")

cv.namedWindow("Image")

img = cv.imread("images/Hunter/hunter.png", cv.IMREAD_COLOR)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.5, 5)
print("faces ")
print(faces)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #id, conf = recognizer.predict(roi_gray)
    #print(conf)
    #if conf >= 45 and conf <= 85:
    #    print("id", id)
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    for (ex,ey,ew,eh) in eyes:
        print("eyes ")
        print((ex,ey),(ex+ew+10,ey+eh))
        cv.rectangle(roi_color,(ex,ey),(ex+ew+10,ey+eh),(0,255,0),2)  
##            
cv.imshow('Image',img)
cv.waitKey(0)
cv.destroyAllWindows()




















##    roi_erase = 100*np.zeros((h,w,3), np.uint8)
##    ##cv.imshow("first", roi_erase)
##    roi_erase = cv.absdiff(img[y:y+h, x:x+w], roi_erase)
##    cv.imshow("second", roi_erase)
##
##    eyes = eye_cascade.detectMultiScale(roi_gray, 1.5)
##    for (ex,ey,ew,eh) in eyes:
##        print("eyes ")
##        print((ex,ey),(ex+ew+10,ey+eh))
##        cv.rectangle(roi_color,(ex,ey),(ex+ew+10,ey+eh),(0,255,0),2)
