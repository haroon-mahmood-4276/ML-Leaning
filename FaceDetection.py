import cv2
import matplotlib.pyplot as plt
import numpy as np

video = cv2.VideoCapture(2)

faceCascade = cv2.CascadeClassifier("harcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("harcascades/haarcascade_eye.xml")

if video.isOpened() == False:
    print("Error")

while True:
    ret, frame = video.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        grayFrame, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25)
    )
    eyes = eyeCascade.detectMultiScale(
        grayFrame, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25)
    )

    for x, y, w, h in faces:
        cv2.rectangle(grayFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for xe, ye, we, he in eyes:
        cv2.rectangle(grayFrame, (xe, ye), (xe + we, ye + he), (0, 255, 0), 2)

    if ret == True:
        cv2.imshow("Video", grayFrame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break

video.release()

cv2.destroyAllWindows()
