import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

image = cv2.imread("mugshot.jpg", cv2.IMREAD_COLOR)

height, width = image.shape[:2]

newmat = np.zeros((height,width,3), np.uint8)

newmat[:] = 30

mixedimg = np.add(image,newmat)

textimg = cv2.putText(mixedimg,"CSCI442 - Nathan Parnell", (30,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


gray = cv2.cvtColor(textimg, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5,)
print("faces ")
print(faces)
for (x,y,w,h) in faces:
    cv2.rectangle(textimg,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = textimg[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)
    for (ex,ey,ew,eh) in eyes:
        print("eyes ")
        print((ex,ey),(ex+ew+10,ey+eh))
        cv2.rectangle(roi_color,(ex,ey),(ex+ew+10,ey+eh),(0,255,0),2)            


cv2.imshow('Nathan Parnell',textimg)

cv2.waitKey(0)

cv2.destroyAllWindows()