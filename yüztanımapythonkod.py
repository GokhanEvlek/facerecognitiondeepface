import cv2 as cv
from deepface.commons import functions
from deepface import DeepFace
from mtcnn import MTCNN

cap=cv.VideoCapture(0)
detector = MTCNN()
while True:
    bir,frame=cap.read()
    

    detections = detector.detect_faces(frame)
    if len(detections) !=0:
        print(type(detections[0]))
        print(detections[0]["box"])
        coordinates=detections[0]["box"]
        frame=cv.rectangle(frame,(coordinates[0],coordinates[1]),(coordinates[0]+coordinates[2],coordinates[1]+coordinates[3]),(255,255,255),1)
        obj = DeepFace.verify("tanımak istediğiniz yüz fotoğrafının yolu",frame, model_name='ArcFace',enforce_detection= False, detector_backend='mtcnn')
        print(obj)
        if obj["verified"]==True:
            frame=cv.putText(frame,"Tanınacak kişinin adı",(coordinates[0],coordinates[1]+coordinates[3]+20),1,2,(255,0,0),2,cv.LINE_AA)
        cv.imshow("Cam",frame)
        k=cv.waitKey(20)
        if k==27:
            break
    else:
        cv.imshow("Cam", frame)
        k = cv.waitKey(20)
        if k == 27:
            break
        continue
cap.release()
cv.destroyAllWindows()