from PyQt5 import QtWidgets
import sys
from hui_form import Ui_MainWindow
from deepface import DeepFace
from mtcnn import MTCNN
import cv2 as cv
import time
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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


tespityapildimi=0




class myApp(QtWidgets.QMainWindow):

    def __init__(self):
        super(myApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """
        pixmap = QPixmap("foto.jpg")
        self.ui.fototutucu.setPixmap(pixmap)
        self.ui.fototutucu.resize(pixmap.width(),pixmap.height())
        """
        self.disply_width = 640
        self.display_height = 480

        self.ui.baslatmabutonu.clicked.connect(self.yuztanima)
        


        #self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image,kafa):
        global tespityapildimi
        self.ui.fototutucu.setPixmap(QPixmap.fromImage(Image))
        pixmap = QPixmap("foto.jpg")
        self.ui.veritabanfoto.setPixmap(pixmap)
        #self.ui.veritabanfoto.resize(pixmap.width(),pixmap.height())
        self.ui.kafatutucu.setPixmap(QPixmap.fromImage(kafa))
        if tespityapildimi==1:
            self.ui.kisiadi.setText('Tespit edilen kisi: '+ 'Gökhan')

    def CancelFeed(self):
        self.Worker1.stop()
    def yuztanima(self):
        print("yuz tanima basladı")
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)




class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage,QImage)


    def run(self):
        global tespityapildimi
        self.ThreadActive = True
        Capture = cv.VideoCapture(0)
        model_name='Facenet'
        distance_metric='euclidean'
        model=None

        enforce_detection = True
        detector_backend = 'mtcnn'
        align = True
        prog_bar = True 

        normalization = 'base'


        
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





        img1_representation,detected_face_base = represent(img_path = img_base
            , model_name = model_name, model = model
            , enforce_detection = enforce_detection, detector_backend = detector_backend
            , align = align
            , normalization = normalization
            )
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                
                try:
                    img2_representation,detected_face = represent(img_path = frame
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

        




                except:
                    print("Yuz bulunamadi ")
                    detected_face=blank_foto
                dim = (480, 640)
                dim2= (64,64)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)                
                frame = cv.resize(frame,dim,interpolation = cv.INTER_AREA)
                detected_face = cv.cvtColor(detected_face, cv.COLOR_BGR2RGB)   
                detected_face = cv.resize(detected_face,dim2,interpolation = cv.INTER_AREA)
                flippedkafa=cv.flip(detected_face,1)
                ConvertToQtFormatkafa = QImage(flippedkafa.data, flippedkafa.shape[1], flippedkafa.shape[0], QImage.Format_RGB888)
                FlippedImage = cv.flip(frame, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                #Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(ConvertToQtFormat,ConvertToQtFormatkafa)
    def stop(self):
        self.ThreadActive = False
        self.quit()
