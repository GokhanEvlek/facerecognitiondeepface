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

import mysql.connector 
import json
import csv




Capture = cv.VideoCapture(0)
model_name='Facenet'
distance_metric='euclidean'

enforce_detection = True
detector_backend = 'mtcnn'
align = True
prog_bar = True 

normalization = 'base'
blank_foto= cv.imread("images.jpg")


blank_foto = cv.resize(blank_foto, (112,112))

model = build_model(model_name)



class myApp(QtWidgets.QMainWindow):

    def __init__(self):
        super(myApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)












        self.ui.stackedWidget.setCurrentWidget(self.ui.secim_ekrani)
        self.ui.ogrenci_ekleme.clicked.connect(self.ogrencikaydi)
        self.ui.yoklama_alma.clicked.connect(self.yoklama)
        self.ui.anamenuyedonus.clicked.connect(self.anamenu)

    def ogrencikaydi(self):

        self.ui.stackedWidget.setCurrentWidget(self.ui.kayit_ekrani)
        print("yuz tanima basladı")
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.ui.anamenuyedonus.clicked.connect(self.CancelFeed)
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)


    def yoklama(self):
        
        self.ui.stackedWidget.setCurrentWidget(self.ui.yoklama)


    def anamenu(self):

        self.ui.stackedWidget.setCurrentWidget(self.ui.secim_ekrani)



    def ImageUpdateSlot(self, Image,kafa):
        #self.ui.fototutucu.setPixmap(QPixmap.fromImage(Image))
        pixmap = QPixmap("foto.jpg")
        self.ui.kafatutucu.setPixmap(QPixmap.fromImage(kafa))
        #self.ui.veritabanfoto.setPixmap(pixmap)
        self.ui.fototutucu.setPixmap(QPixmap.fromImage(Image))
        #self.ui.kafatutucu_base.setPixmap(QPixmap.fromImage(kafa_base))


    def CancelFeed(self):
        self.Worker1.stop()







class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage,QImage)


    def run(self):

        global Capture,model_name,model,enforce_detection,detector_backend,align,normalization,blank_foto

        self.ThreadActive = True
        
        
        student_header = ['StudentId', 'Name', 'Surname']

        """

        img_base = cv.imread("C:\\WhatsAppImage2022-09-1920.23.10.jpeg")




        img1_representation,detected_face_base = represent(img_path = img_base
            , model_name = model_name, model = model
            , enforce_detection = enforce_detection, detector_backend = detector_backend
            , align = align
            , normalization = normalization
            )
        detected_face_base= cv.cvtColor(detected_face_base, cv.COLOR_BGR2RGB)
        

        print(type(img1_representation))
        jsonStr = json.dumps(img1_representation)
        print(type(jsonStr))
        print(jsonStr)
        """
        connection = mysql.connector.connect(host='localhost',
                                         database='deneme',
                                         user='root',
                                         password='1234')

        
        sql_select_Query = "select * from ogrenci"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        # get all records
        records = cursor.fetchall()
        print("Total number of rows in table: ", cursor.rowcount)
        print(type(records[0][3]))
        base_embedding=json.loads(records[0][3])
        print(type(base_embedding))
        """
        cursor = connection.cursor()
        mySql_insert_query = "INSERT INTO ogrenci (idogrenci, ograd, ogrsoyad, embedding) VALUES (109,'Gokhan','Evlek', '"+jsonStr+ "') "

        cursor.execute(mySql_insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Laptop table")
        cursor.close()
        """



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
                    for row in records:
                        base_embedding=json.loads(row[3])
                        distance = dst.findEuclideanDistance(base_embedding, img2_representation)
                        distance = np.float64(distance)




                        if distance<10:
                            print("Eslesme oldu: ",distance)
                            with open('students.csv', 'w') as file:
                                writer = csv.writer(file)
                                student_data=[row[0],row[1],row[2]]
                                # 3. Write data to the file
                                writer.writerow(student_header)
                                writer.writerow(student_data)



        
                    



                except:
                    print("Yuz bulunamadi ")
                    detected_face=blank_foto

                dim = (480, 640)
                dim2= (64,64)


                """
                detected_face_base = cv.resize(detected_face_base,dim2,interpolation = cv.INTER_AREA)
                flippedkafa_base=cv.flip(detected_face_base,1)
                ConvertToQtFormatkafa_base = QImage(flippedkafa_base.data, flippedkafa_base.shape[1], flippedkafa_base.shape[0], QImage.Format_RGB888)
                """

                detected_face = cv.cvtColor(detected_face, cv.COLOR_BGR2RGB)   
                detected_face = cv.resize(detected_face,dim2,interpolation = cv.INTER_AREA)
                flippedkafa=cv.flip(detected_face,1)
                ConvertToQtFormatkafa = QImage(flippedkafa.data, flippedkafa.shape[1], flippedkafa.shape[0], QImage.Format_RGB888)
                
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)                
                frame = cv.resize(frame,dim,interpolation = cv.INTER_AREA)          
                
                FlippedImage = cv.flip(frame, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                #Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(ConvertToQtFormat,ConvertToQtFormatkafa)
    def stop(self):
        self.ThreadActive = False
        self.quit()



















def app():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QPushButton { color: red}")
    win = myApp()
    win.show()
    sys.exit(app.exec_())

app()