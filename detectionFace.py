import cv2, os
faceDir = 'faceData' #direktori

cam = cv2.VideoCapture(0)
cam.set(3,648) #lebar
cam.set(4, 488) #tinggi

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')

faceID = input("Masukan Face ID yang akan direkam Datanya [Kemudian ENter]")
print("Tampilkan wajah di webcam")
getData = 1

while True:
    retV, frame = cam.read()
    grayCam = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #frame
    faces = faceDetector.detectMultiScale(grayCam,1.3,5) #frame, scale, 
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,255), 2)
        nameFile = 'wajah.'+str(faceID)+'.'+str(getData)+'.jpg' #directori file face 
        cv2.imwrite(faceDir+'/'+nameFile,frame)  #save faceData
        getData += 1

        roiGray = grayCam[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiGray)
        for(xe, ye, we, he) in eyes:
            cv2.rectangle(roiColor, (xe,ye), (xe+we,ye+he), (0,0,255,1))

    cv2.imshow('webcamku', frame)
    k = cv2.waitKey(1) & 0xFF
    if  k == 27 or k == ord('q'):
        break
    elif getData > 15:
        break
print("get face data success")
cam.release()
cv2.destroyAllWindows()