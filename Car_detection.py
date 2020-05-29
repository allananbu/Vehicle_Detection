# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:48:35 2020

@author: Allan
"""

import cv2

#capture frames from video
cap = cv2.VideoCapture(r'C:\Users\Allan\Desktop\Computer-Vision---Object-Detection-in-Python-master\Cars.mp4')
#save output file
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('cars_output.avi',fourcc, 20.0, (int(width),int(height)))

#use trained XML classifiers
car_cascade = cv2.CascadeClassifier(r'C:\Users\Allan\Desktop\Computer-Vision---Object-Detection-in-Python-master\xml files\cars.xml')

while True:
    ret,frame = cap.read()
    #convert frame into grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect cars using classifier
    cars = car_cascade.detectMultiScale(gray,1.5,2)
    #draw rectangle around detected car
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Car',(x + 6, y - 6), font, 0.5, (0, 0, 255), 1)
        cv2.imshow('detected cars',frame)
        #write video to output file
        out.write(frame)
        
        #press Esc or Q key to exit output screen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

