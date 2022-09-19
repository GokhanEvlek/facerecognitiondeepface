import cv2 as cv
from deepface.commons import functions
from deepface import DeepFace
from mtcnn import MTCNN
import time



cap=cv.VideoCapture(0)
detector = MTCNN()
while True:
    start=time.time()
    bir,frame1=cap.read()
    
    frame = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(frame)
    if len(detections) !=0:
        print(type(detections[0]))
        print(detections[0]["box"])
        coordinates=detections[0]["box"]
        
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #cv.imwrite('yuz.jpg', frame)
        obj = DeepFace.verify("C:\\WhatsAppImage2022-09-1920.23.10.jpeg",frame, model_name='Facenet',enforce_detection= True, detector_backend='mtcnn')
        frame1=cv.rectangle(frame1,(coordinates[0],coordinates[1]),(coordinates[0]+coordinates[2],coordinates[1]+coordinates[3]),(255,255,255),1)
        print(obj)
        if obj["verified"]==True:
            frame1=cv.putText(frame1,"Tanınacak kişinin adı",(coordinates[0],coordinates[1]+coordinates[3]+20),1,2,(255,0,0),2,cv.LINE_AA)
            frame1=cv.putText(frame1,str(1/(time.time()-start)),(30,30),1,2,(255,0,0),2,cv.LINE_AA)
        cv.imshow("Cam",frame1)
        k=cv.waitKey(20)
        if k==27:
            break
    else:
        cv.imshow("Cam", frame1)
        k = cv.waitKey(20)
        if k == 27:
            break
        continue
cap.release()
cv.destroyAllWindows()
