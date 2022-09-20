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
        detector = MTCNN()

        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                detections = detector.detect_faces(frame)
                print("yuz tanima yapiliyor")
                if len(detections) !=0:
                    print(type(detections[0]))
                    print(detections[0]["box"])
                    coordinates=detections[0]["box"]
                    
                    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    #cv.imwrite('yuz.jpg', frame)
                    obj = DeepFace.verify("foto.jpg",frame, model_name='Facenet',enforce_detection= True, detector_backend='mtcnn')
                    
                    frame=cv.rectangle(frame,(coordinates[0],coordinates[1]),(coordinates[0]+coordinates[2],coordinates[1]+coordinates[3]),(255,255,255),1)
                    print(obj)
                    if obj["verified"]==True:
                        #frame=cv.putText(frame,"Tanınacak kişinin adı",(coordinates[0],coordinates[1]+coordinates[3]+20),1,2,(255,0,0),2,cv.LINE_AA)
                        tespityapildimi=1
                        kafa=frame[coordinates[1]:coordinates[1]+coordinates[3],coordinates[0]:coordinates[0]+coordinates[2]]

                dim = (480, 640)
                dim2= (64,64)                
                frame = cv.resize(frame,dim,interpolation = cv.INTER_AREA)
                if tespityapildimi==0:
                    kafa=frame[80:280, 150:330]
                kafa = cv.resize(kafa,dim2,interpolation = cv.INTER_AREA)
                flippedkafa=cv.flip(kafa,1)
                ConvertToQtFormatkafa = QImage(flippedkafa.data, flippedkafa.shape[1], flippedkafa.shape[0], QImage.Format_RGB888)
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