from helper_main import *
import cv2

#indices = get_camera_indices()
#print(indices)
# 0,2,700,702,1400

dimensions = (400,800)
cap_live = cv2.VideoCapture(2)
while True:
    ok, frame = cap_live.read()
    resized = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    cv2.imshow("window", resized)
    cv2.imshow('window', frame)