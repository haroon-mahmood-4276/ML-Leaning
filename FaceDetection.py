import cv2
import matplotlib.pyplot as plt
import numpy as np

video = cv2.VideoCapture(2)

faceCascade = cv2.CascadeClassifier("harcascades/haarcascade_frontalface_default.xml")

if video.isOpened() == False:
    print("Error")

while True:
    ret, frame = video.read()
    
    faces = faceCascade.detectMultiScale(
        frame, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25)
    )

    for x, y, w, h in faces:
        center = np.array(((x + (x + w)) / 2, (y + (y + h)) / 2)).astype(int)

        mask = np.zeros(shape=frame.shape, dtype=np.uint8)
        frameBlured = cv2.blur(src=frame, ksize=(25, 25), borderType=cv2.BORDER_DEFAULT)
    
        mask = cv2.circle(mask, center, 130, (255, 255, 255), -1)
        frame = np.where(mask != np.array([255, 255, 255]), frame, frameBlured)

    if ret == True:
        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    else:
        break

video.release()

cv2.destroyAllWindows()
