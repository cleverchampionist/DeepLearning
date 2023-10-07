import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray, 1.03, 7)

for (ex, ey, ew, eh) in eyes:
    img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
cv2.imwrite('Eye_AB.jpg', img)
cv2.imshow('eyes detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()