import cv2 as cv
from deepface.commons import functions
from deepface import DeepFace
from mtcnn import MTCNN
import time
from deepface.basemodels import Facenet
import numpy as np

from deepface.commons import functions, realtime, distance as dst


model = Facenet.loadModel()



cap=cv.VideoCapture(0)
img = cv.imread("C:\\WhatsAppImage2022-09-1920.23.10.jpeg")
input_shape=(1,160,160,3)

img1,img2 = functions.detect_face(img, detector_backend = 'mtcnn')
#img1 = functions.preprocess_face(img, input_shape,detector_backend = 'mtcnn')
img1 = cv.resize(img1, (160,160))

img1=np.reshape(img1, input_shape)
print(img1.shape)
img2_representation = model.predict(img1)[0,:]
#print(img2_representation)
#cv.imshow("foto",img1)
#cv.waitKey(0)






"""
img = cv.imread("C:\\WhatsAppImage2021-07-21at20.25.39.jpeg")
input_shape=(1,160,160,3)

img1,img2 = functions.detect_face(img, detector_backend = 'mtcnn')
#img1 = functions.preprocess_face(img, input_shape,detector_backend = 'mtcnn')
img1 = cv.resize(img1, (160,160))

img1=np.reshape(img1, input_shape)
print(img1.shape)
img1_representation = model.predict(img1)[0,:]
print(img1_representation)

distance = dst.findCosineDistance(img1_representation, img2_representation)


#distance_vector = np.square(img1_representation - img2_representation)

#distance = np.sqrt(distance_vector.sum())
print("Euclidean distance: ",distance)
if distance <0.4:
    print("eslesme oldu")
"""







while True:

    bir,img4=cap.read()
    
    input_shape=(1,160,160,3)

    img3,img2 = functions.detect_face(img4, detector_backend = 'mtcnn')
    if img3 is None:
        continue
    cv.imshow("surat",img3)
    cv.waitKey(1)
    img3 = cv.resize(img3, (160,160))

    img3=np.reshape(img3, input_shape)
    print(img1.shape)
    img1_representation = model.predict(img3)[0,:]
    #print(img1_representation)

    distance = dst.findCosineDistance(img1_representation, img2_representation)


    #distance_vector = np.square(img1_representation - img2_representation)

    #distance = np.sqrt(distance_vector.sum())
    print("Euclidean distance: ",distance)
    if distance <0.4:
        print("eslesme oldu")
