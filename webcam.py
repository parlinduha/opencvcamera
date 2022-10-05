import cv2
cam = cv2.VideoCapture(0)
while True:
    retV, frame = cam.read()
    cv2.imshow('webcamku', frame)
    k = cv2.waitKey(1) & 0xFF
    if  k == 27 or k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()