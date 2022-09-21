from dis import dis
import cv2 as cv
from deepface.commons import functions
from deepface import DeepFace
from mtcnn import MTCNN
import time
from deepface.basemodels import Facenet,ArcFace
import numpy as np

from deepface.commons import functions, realtime, distance as dst
from deepface.DeepFace import *
from deepface.detectors import FaceDetector

from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion



#definitions
model_name='Facenet'
distance_metric='euclidean'
model=None

enforce_detection = True
detector_backend = 'mtcnn'
align = True
prog_bar = True 
normalization = 'base'

#detector = MTCNN()

cap=cv.VideoCapture(0)

blank_foto= cv.imread("images.jpg")


blank_foto = cv.resize(blank_foto, (112,112))



img_base = cv.imread("C:\\WhatsAppImage2022-09-1920.23.10.jpeg")


if model == None:
    if model_name == 'Ensemble':
        models = Boosting.loadModel()
    else:
        model = build_model(model_name)
        models = {}
        models[model_name] = model
else:
    if model_name == 'Ensemble':
        Boosting.validate_model(model)
        models = model.copy()
    else:
        models = {}
        models[model_name] = model

if model_name == 'Ensemble':
    model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
else:
    model_names = []; metrics = []
    model_names.append(model_name)
    metrics.append(distance_metric)

#--------------------------------


#işlemler

#img_list, bulkProcess = functions.initialize_input(img_base)

#img_list, bulkProcess = functions.initialize_input("C:\\WhatsAppImage2022-09-1920.23.10.jpeg") verilen iki resmin adresini liste olarak döndürüyor
resp_objects = []
#print(img_list)
#--------------------------------





#------------------------------

#disable_option = (False if len(img_list) > 1 else True) or not prog_bar #True 


img1_representation,detected_face_base = represent(img_path = img_base
        , model_name = model_name, model = model
        , enforce_detection = enforce_detection, detector_backend = detector_backend
        , align = align
        , normalization = normalization
        )
print(img1_representation)






while True:
    start=time.time()
    bir,frame1=cap.read()
    
    try:
        img2_representation,detected_face = represent(img_path = frame1
            , model_name = model_name, model = model
            , enforce_detection = enforce_detection, detector_backend = detector_backend
            , align = align
            , normalization = normalization
            )
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
        distance = np.float64(distance)
        if distance<10:
            print("Eslesme oldu: ",distance)
        else:
            print("Eslesme olmadi: ", distance)
        cv.imshow("Cam", frame1)
        cv.imshow("Detected_face", detected_face)
        k = cv.waitKey(20)
        if k == 27:
            break
        continue
    except:
        print("Yuz bulunamadi ")
        cv.imshow("Cam", frame1)
        k = cv.waitKey(20)
        if k == 27:
            break
        continue
cap.release()
cv.destroyAllWindows()