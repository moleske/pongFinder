import cv2
import os
import numpy as np

for file in os.listdir('images'):
    input = cv2.imread('images/'+file)
    mask = cv2.inRange(input, np.array([6, 42, 128]), np.array([90, 210, 255]))
    result = cv2.bitwise_and(input, input, mask= mask)
    cv2.imwrite('result/' + file, result)

for file in os.listdir('result'):
    input = cv2.imread('result/'+file)
    original = cv2.imread('images/'+file)
    input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(input,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rectangle = cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite('training/'+file, original)