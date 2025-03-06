import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QAction,QWidget, QCheckBox, QFrame, QVBoxLayout,QHBoxLayout, QScrollArea,QSlider,QButtonGroup, QComboBox,QLineEdit,QSpinBox,QInputDialog,QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
import random
import zipfile
import rasterio
import os
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
#from osgeo import gdal
from PIL import Image
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import numpy as np
import re
import lDA_last_2025latest
import lda_changes2023
from PyQt5.QtWebEngineWidgets import QWebEngineView
from rasterio.windows import Window
import pickle
import GMM_n_certainty as gmm
#import Math



class EnlargedView(QWidget):
    def __init__(self, pixmap, pointer_pos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enlarged View")
        self.setGeometry(800, 400, 1500, 800)
        #self.backGroundImage("backImage.jpg")

        self.label = QLabel(self)
        self.pixmap = pixmap
        self.focusImgWidth = self.pixmap.width()
        #print(self.focusImgWidth/3)
        #print(self.focusImgWidth//3)
        self.focusImgHeight = self.pixmap.height()
        #print(self.focusImgHeight/3)
        #print(self.focusImgHeight//3)
        self.focusButton = QPushButton(" ", self)
        self.focusButton.setGeometry(self.focusImgWidth//3, self.focusImgHeight//3, self.focusImgWidth//3, self.focusImgHeight//3)
        self.focusButton.setStyleSheet("background-color: transparent;")
        self.focusButton.setStyleSheet("border: 3px solid red;")
        

       # painter = QPainter(self.pixmap)
       # pen = QPen(Qt.red, 3)
       # painter.setPen(pen)
       # painter.drawEllipse(self.pixmap.width()//2 - 5, self.pixmap.height()//2 - 5, 10, 10)
       # painter.end()

        self.label.setPixmap(self.pixmap)
        self.label.setScaledContents(True)
        self.label.setGeometry(0, 0, self.focusImgWidth, self.focusImgHeight)

class ImagefocusungApp(QMainWindow):
    def __init__(self):
        super(ImagefocusungApp,self).__init__()
        self.setWindowTitle("Explainable Uncertainty-aware Machine Learning Tool for unsupervised classification")
        self.setGeometry(100, 100, 1800, 900)
        self.backGroundImage("backImage.jpg")
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(0, 0, 1800, 900)
        self.scroll_area.setWidgetResizable(True)
        #self.setCentralWidget(self.scroll_area)
        
        #self.container = QWidget(self)
        self.container = QWidget()
          
        self.scroll_area.setWidget(self.container)  
        self.container.setFixedSize(3000, 3000)
        
        
        
        
        #self.scroll_area = QScrollArea(self)
        #self.scroll_area.setGeometry(0, 0, 1000, 1000)
        
        #self.scroll_area.setWidgetResizable(True)
        #self.setCentralWidget(self.scroll_area) 
        #self.scroll_area.setWidget(self)
        #self.setFixedSize(1000, 1200)
        
        #self.content_widget = QWidget()
        #self.content_widget.setFixedSize(1000, 1200)
        self.startApp()

    def startApp(self):
    
        self.imgWidth = 0                   #Initial img width
        self.imgHeight = 0                  #Initial img height
        self.InitLabelheight=500            #Initial label width    
        self.InitLabelwidth=750             #Initial label height 
        self.leftBuffer=50                  #space left on the left side as buffer
        self.topBuffer=50                   #space left on the top as buffer
        self.buttonWidth = 150              #width of buttons
        self.buttonHeight = 30              #Height of buttons
        self.indX = 50                      #X axis coordinate of indicator
        self.indY = 50                      #Y axis coordinate of indicator
        self.indXJump=6                     #X axis Jump variable of indicator
        self.indYJump=4                     #Y axis Jump variable of indicator
       # self.indButtonDim=30               #Height and Widtj of indicator
        self.indButtonWidth=30              #indicator width
        self.indButtonHeight=30             #indicator height
        self.checkBoxWidth=160
        self.smallPatchDim=4
        self.smallPatchDimVariable=self.smallPatchDim
        self.transparentButtomXDim=self.InitLabelwidth//self.smallPatchDim
        self.transparentButtomYDim=self.InitLabelheight//self.smallPatchDim
        self.transparentCBBuffer=25
        self.noOfBigPatches=0
        self.isNewLabelProject=-1
        self.pixmap=0
        self.gridSize=self.smallPatchDim
        self.scaling_factor=1
        self.ldaExecuted=0
        self.transparencyBoxItems=[]
        self.selectedButtonList=[]
        self.toolPath="C:/Users/goya_sh/Desktop/Neuer Ordner"
        self.isNewLDAProject=-1
        self.max_size=1200
        self.focusImgHeight=0
        self.startedLabelling=0

        #np.zeros(((len(self.opticalImages)),))
        #self.classificationList = [[random.randint(0, 4)]*(self.self.transparentButtomXDim*self.transparentButtomXDim) for _ in range (len(self.opticalImages))]#random.randint(0, 255)
        #self.classificationList = [0 for _ in range (len(self.opticalImages)) ]random.randint(0, 255)
       # print(type(self.classificationList))
        self.loadclicked=0
        self.imageCoordinates=[[11111132222222,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        #self.download_sentinel2_data()
        self.zippedFIle = "S2A_MSIL2A_20250103T044201_N0511_R033_T45RXL_20250103T080459.SAFE.zip" 
        self.extractedToFolder = "C:/Users/goya_sh/Desktop/Uncertainity Analysis/Sentinel2" 
        #self.extract_from_zip(self.zippedFIle, self.extractedToFolder)
        #self.bandImages = self.list_of_images(self.extractedToFolder,0)
       # self.bandImages = self.read_images("C:/Users/goya_sh/Desktop/Neuer Ordner/readImage")
        #self.bandImages = self.list_of_images("C:/Users/goya_sh/Desktop/Neuer Ordner/readImage",0)
        #self.opticalImages=self.list_of_images("C:/Users/goya_sh/Desktop/Neuer Ordner/LDA_Images",0)
        
        
        self.opticalImageIndex=0
        self.scaled_ratio=1
        self.classificationImages=None
        self.ldaOutputImages=None
        
        self.sentinel2CheckBoxclicked=False
        #print(self.bandImages)
        self.max_size=500
        self.images = {}
        self.current_image_key = None
       # self.sampleBand = [f for f in self.bandImages if "B01" in f]
       # print(self.sampleBand)
        self.button_group = QButtonGroup(self.container)
        self.button_group.buttonClicked.connect(self.on_checkbox_clicked)
        self.transparencyChechBoxList=[0]*13
        self.imageCount=0
        self.canvas=0
        self.focusbox=-1
        self.selectedClass=-1
        self.classificationCodes={0:"Fire",1:"Smoke",2:"Road",3:"Vegetation",4:"settelment",5:"Industries",6:"building",7:"forest",8:"airport",9:"water",10:"cloud",11:"coastline",12:"sand"}
        self.comboBoxItems=[]
        self.ldaExperimentName=''
        self.experimentName=''
        self.labelExperimentName=''
        self.gmmExperimentName=''
        self.bandNumber=''
        self.noOfTopics=''
        self.bigPatchDim=128
        self.isNewGMMProject=1
        
        self.latitude = 48.1351  
        self.longitude = 11.5820
        for ik in self.classificationCodes.items():
            self.comboBoxItems.append(ik[1]+' - '+str(ik[0]))
        


        self.imgLabel = QLabel(self.container)
        self.imgLabel.setGeometry(self.leftBuffer, self.topBuffer, self.InitLabelwidth, self.InitLabelheight)
        self.imgLabel.setStyleSheet("border: 1px solid black;")
        
        self.newLabel = QLabel(self.container)
        self.newLabel.setGeometry(0,0,0,0)
        
        self.web_view = QWebEngineView(self.container)
        self.web_view.setGeometry(self.leftBuffer, self.topBuffer, self.InitLabelwidth, self.InitLabelheight)
        self.web_view.hide()

        self.showLabelButtons = QPushButton("show labels", self.container)
        self.showLabelButtons.setGeometry(self.leftBuffer, self.InitLabelheight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.showLabelButtons.clicked.connect(self.showLabels)
        
        self.clearLabel = QPushButton("Clear labels", self.container)
        self.clearLabel.setGeometry(self.leftBuffer+self.InitLabelwidth-self.buttonWidth, self.InitLabelheight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.clearLabel.clicked.connect(self.clearLabels)
       
        self.combo_box = QComboBox(self.container)
        self.combo_box.addItems(self.comboBoxItems)
        self.combo_box.setGeometry(self.leftBuffer+(self.buttonWidth)*1+self.InitLabelwidth, self.topBuffer*5, self.buttonWidth, self.buttonHeight)
        self.combo_box.activated.connect(self.selectClass)
        
        self.combo_box_Transparency = QComboBox(self.container)
        self.combo_box_Transparency.addItems(self.transparencyBoxItems)
        self.combo_box_Transparency.setGeometry(self.leftBuffer+self.InitLabelwidth, self.InitLabelheight + (self.topBuffer)*6, self.buttonWidth, self.buttonHeight)
        self.combo_box_Transparency.activated.connect(self.transparencyImageSelection)
        self.combo_box_Transparency.hide()
        self.initImagesTruncated=[]


        self.combo_box_Images = QComboBox(self.container)
        self.combo_box_Images.addItems(self.initImagesTruncated)
        self.combo_box_Images.setGeometry(self.leftBuffer+self.InitLabelwidth, self.topBuffer*3, self.buttonWidth, self.buttonHeight)
        self.combo_box_Images.activated.connect(self.showLdaImages)
        self.combo_box_Images.setMaxVisibleItems(10)
        self.combo_box_Images.hide()

        
        
        self.loadImageForLabelling = QPushButton("Label", self.container)
        self.loadImageForLabelling.setGeometry(self.leftBuffer+self.buttonWidth+self.buttonWidth//3, self.InitLabelheight+(self.topBuffer)*2, self.buttonWidth//2, self.buttonHeight)
        self.loadImageForLabelling.clicked.connect(self.initiateLabelling)
        
        self.clearLabelling = QPushButton("Clear", self.container)
        self.clearLabelling.setGeometry(self.leftBuffer+self.InitLabelwidth-(self.buttonWidth+(2*self.buttonWidth//3)), self.InitLabelheight+(self.topBuffer)*2, self.buttonWidth//2, self.buttonHeight)
        self.clearLabelling.clicked.connect(self.clearSelection)
        
        
        self.loadButtomLDA = QPushButton("LDA", self.container)
        self.loadButtomLDA.setGeometry(self.leftBuffer+(self.buttonWidth)*0+self.InitLabelwidth, self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.loadButtomLDA.clicked.connect(self.getLdaVariables)
        
        self.loadButtomGMM = QPushButton("GMM", self.container)
        self.loadButtomGMM.setGeometry((self.leftBuffer+self.buttonWidth)*1+self.InitLabelwidth, self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.loadButtomGMM.clicked.connect(self.getGmmVariables)
        
        self.loadButtomBayesian = QPushButton("Bayesian", self.container)
        self.loadButtomBayesian.setGeometry(self.leftBuffer+(self.buttonWidth)*2+self.InitLabelwidth, self.topBuffer, self.buttonWidth, self.buttonHeight)
        
        self.loadButtomModelE = QPushButton("Model Explainability", self.container)
        self.loadButtomModelE.setGeometry(self.leftBuffer+(self.buttonWidth)*0+self.InitLabelwidth, self.topBuffer+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight+self.buttonHeight//2)
        
        self.loadButtomUncertainityA = QPushButton("Uncertainity Analysis", self.container)
        self.loadButtomUncertainityA.setGeometry(self.leftBuffer+(self.buttonWidth)*0+self.InitLabelwidth+self.buttonWidth+self.buttonWidth//2, self.topBuffer+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight+self.buttonHeight//2)
        
        self.saveWorkButton = QPushButton("Save Labels", self.container)
        self.saveWorkButton.setGeometry(self.leftBuffer+(self.buttonWidth)*2+self.InitLabelwidth, self.topBuffer*5, self.buttonWidth, self.buttonHeight)
        self.saveWorkButton.clicked.connect(self.saveWork)
        
        self.addLabelButton = QPushButton("Add Label", self.container)
        self.addLabelButton.setGeometry(self.leftBuffer+(self.buttonWidth)*2+self.InitLabelwidth, self.topBuffer*5-self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.addLabelButton.clicked.connect(self.addLabel)
        
        self.labelClassification = QLabel("", self.container)
        self.labelClassification.setGeometry(self.InitLabelwidth//2-self.buttonWidth, 10, self.buttonWidth*3, self.buttonHeight)
        self.labelClassification.setStyleSheet("border: 1px solid black;")
        self.labelClassification.setStyleSheet("color: blue; background-color: transparent; font-size: 18px;")
       # self.labelClassification.setAlignment(Qt.AlignCenter)
        
        self.labelVerticalCounter = QLabel("Small patch size", self.container)
        self.labelVerticalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*2+self.InitLabelwidth, self.topBuffer*4-(self.buttonHeight)*2,(self.buttonWidth)*2, self.buttonHeight)
        self.labelVerticalCounter.hide()
        
        #self.labelHorizontalCounter = QLabel("horizontal boxes", self.container)
        #self.labelHorizontalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*1+self.InitLabelwidth, self.topBuffer*4-(self.buttonHeight)*1,(self.buttonWidth)*1, self.buttonHeight)
        #self.labelHorizontalCounter.hide()
        
        self.verticalCounter= QSpinBox(self.container)
        self.verticalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*3+self.InitLabelwidth, self.topBuffer*4-(self.buttonHeight)*2, self.buttonWidth, self.buttonHeight)
        self.verticalCounter.setRange(0,100)
        self.verticalCounter.setValue(self.smallPatchDim)
        self.verticalCounter.valueChanged.connect(self.setNewSmallPatchsize)
        self.verticalCounter.hide()
        
        self.executeLdaButton = QPushButton("Execute LDA", self.container)
        self.executeLdaButton.setGeometry(self.leftBuffer+(self.buttonWidth)*4+self.InitLabelwidth, self.topBuffer*4-(self.buttonHeight)*2, self.buttonWidth, self.buttonHeight)
        self.executeLdaButton.clicked.connect(self.verticalBoxesChange)
        self.executeLdaButton.hide()
        
        #self.horizontalCounter= QSpinBox(self.container)
        #self.horizontalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*2+self.InitLabelwidth, self.topBuffer*4-(self.buttonHeight)*1, self.buttonWidth, self.buttonHeight)
        #self.horizontalCounter.setRange(0,100)
        #self.horizontalCounter.setValue(10)
        #self.horizontalCounter.valueChanged.connect(self.horizontalBoxesChange)
        #self.horizontalCounter.hide()
        
        self.textBox = QLineEdit(self.container)
        self.textBox.setGeometry(self.leftBuffer+(self.buttonWidth)*1+self.InitLabelwidth, (self.topBuffer*5)-self.buttonHeight, self.buttonWidth, self.buttonHeight)
        
        #label.setStyleSheet("""
        #    color: white;
        #    background-color: green;
        #    border: 2px solid black;
        #    border-radius: 5px;
        #    font-size: 20px;
        #    font-weight: bold;""")
        self.focusButton = QPushButton(" ", self.container)
        self.focusButton.setGeometry(0,0,0,0)
        
 
        self.topLeftCoor = QLabel("Top left:", self.container)
        self.topLeftCoor.setGeometry(0,0,0,0)
        
        self.topLeftCoorshow = QLabel("", self.container)
        self.topLeftCoorshow.setGeometry(0,0,0,0)

        self.topRightCoor = QLabel("Top Right:", self.container)
        self.topRightCoor.setGeometry(0,0,0,0)
        
        self.topRightCoorshow = QLabel("", self.container)
        self.topRightCoorshow.setGeometry(0,0,0,0)

        self.bottomLeftCoor = QLabel("Bottom left:", self.container)
        self.bottomLeftCoor.setGeometry(0,0,0,0)
        
        self.bottomLeftCoorshow = QLabel("", self.container)
        self.bottomLeftCoorshow.setGeometry(0,0,0,0)

        self.bottomRightCoor = QLabel("Bottom Right:", self.container)
        self.bottomRightCoor.setGeometry(0,0,0,0)
        
        self.bottomRightCoorshow = QLabel("", self.container)
        self.bottomRightCoorshow.setGeometry(0,0,0,0)

       
        self.main_checkbox = QCheckBox("Sentinel 2",self.container)
        self.main_checkbox.setGeometry(self.leftBuffer, self.InitLabelheight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        self.main_checkbox.stateChanged.connect(self.create_checkboxes)
        
        self.sourceImageCheckbox = QCheckBox("Source",self.container)
        self.sourceImageCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*3, self.InitLabelheight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        self.sourceImageCheckbox.stateChanged.connect(lambda _,k=-1:self.load_or_remove_optical_image(k))

        self.classificationMapoCheckbox = QCheckBox("Classification map",self.container)
        self.classificationMapoCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*7, self.InitLabelheight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        self.classificationMapoCheckbox.stateChanged.connect(lambda _,k=-2:self.load_or_remove_optical_image(k))
        
        self.mapServicesCheckbox = QCheckBox("Map services",self.container)
        self.mapServicesCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*11, self.InitLabelheight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        self.mapServicesCheckbox.stateChanged.connect(self.load_map)
        
        self.startLabel = QPushButton("Start Labeling", self.container)
        #self.startLabel.setGeometry(self.leftBuffer+self.InitLabelwidth//2-self.buttonWidth//2, self.InitLabelheight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.startLabel.setGeometry(self.leftBuffer+self.InitLabelwidth//2-self.buttonWidth//2, self.InitLabelheight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.startLabel.clicked.connect(self.changeLabelFlag)
        self.previousButton = QPushButton("Previous", self.container)
        self.previousButton.setGeometry(self.leftBuffer, self.InitLabelheight+(self.topBuffer)*2, self.buttonWidth, self.buttonHeight)
        self.previousButton.hide()
        self.previousButton.clicked.connect(self.moveToPrevious)
        
        self.nextButton = QPushButton("Next", self.container)
        self.nextButton.setGeometry(self.leftBuffer+self.InitLabelwidth-self.buttonWidth, self.InitLabelheight+(self.topBuffer)*2, self.buttonWidth, self.buttonHeight)
        self.nextButton.hide()
        self.nextButton.clicked.connect(self.moveToNext)
        
        self.transparency_slider = QSlider(Qt.Horizontal, self.container)
        self.transparency_slider.setRange(0, 100) 
        self.transparency_slider.setValue(100)
        self.transparency_slider.setGeometry(50, self.InitLabelheight + (self.topBuffer)*6, 500, 30)  
        self.transparency_slider.valueChanged.connect(self.adjust_transparency)
        
        
        
        

        self.indButton = QPushButton(" ", self.container)
        self.indButton.setGeometry(self.indX, self.indY, self.indButtonWidth, self.indButtonHeight)
        self.indButton.setStyleSheet("background-color: transparent;")
        self.indButton.setStyleSheet("border: 1px solid black;")
        self.indButton.hide()

        self.moveIndButton = QPushButton("Move", self.container)
        self.moveIndButton.setGeometry(self.leftBuffer+self.InitLabelwidth-self.buttonWidth, self.InitLabelheight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.moveIndButton.clicked.connect(self.move_indicator)
        self.moveIndButton.hide()
        #self.transparentButtomTemp = QPushButton("  ", self)
        self.transparentButtom = [0]*(100000)
        self.bandCheckBoxes=[0]*13
        self.transparencyChechBoxList=[0]*16#tranparency
        self.tbuttonwidth=self.imgWidth//self.transparentButtomXDim
        self.tbuttonHeight=self.imgHeight//self.transparentButtomYDim
        self.displayClasses=[0]*50
        for y in range(50):
            if (y in self.classificationCodes.keys()):
                self.displayClasses[y]=QLabel(str(y)+' - '+self.classificationCodes[y], self.container)
            else:
                self.displayClasses[y]=QLabel(str(y), self.container)
           # self.displayClasses[y].setGeometry(10, self.topBuffer+20*y, self.buttonWidth, 15)
            self.displayClasses[y].setGeometry((self.InitLabelwidth+self.leftBuffer*5)*self.scaled_ratio, (self.InitLabelheight)+25*y, self.buttonWidth, 20)
            self.displayClasses[y].hide()
        self.displayClasses[0].setStyleSheet("color: red; background-color: transparent; font-size: 18px;")
        self.displayClasses[1].setStyleSheet("color: green; background-color: transparent; font-size: 18px;")
        self.displayClasses[2].setStyleSheet("color: yellow; background-color: transparent; font-size: 18px;")
        self.displayClasses[3].setStyleSheet("color: magenta; background-color: transparent; font-size: 18px;")
        self.displayClasses[4].setStyleSheet("color: cyan; background-color: transparent; font-size: 18px;")
        self.displayClasses[5].setStyleSheet("color: orange; background-color: transparent; font-size: 18px;")
        self.displayClasses[6].setStyleSheet("color: purple; background-color: transparent; font-size: 18px;")
        self.displayClasses[7].setStyleSheet("color: teal; background-color: transparent; font-size: 18px;")
        self.displayClasses[8].setStyleSheet("color: brown; background-color: transparent; font-size: 18px;")
        self.displayClasses[9].setStyleSheet("color: crimson; background-color: transparent; font-size: 18px;")
        #self.classDisplayLAbel = QLabel(" ", self.container)
        
        for j in range(13):
            self.bandCheckBoxes[j] = QCheckBox(f"Band{j+1}",self.container)
            is_visible = self.bandCheckBoxes[j].isVisible()
            #print(is_visible)
            self.bandCheckBoxes[j].setVisible(is_visible)
            self.bandCheckBoxes[j].stateChanged.connect(lambda _,k=j:self.load_or_remove_image(k))
            #self.main_checkbox.setGeometry(0,0,0,0)
        
        for n in range(13):
            self.transparencyChechBoxList[n] = QCheckBox(f"Band{n+1}",self.container)
            is_transparecyBoxvisible = self.transparencyChechBoxList[n].isVisible()
            self.transparencyChechBoxList[n].setVisible(is_transparecyBoxvisible)
            
        self.transparencyChechBoxList[13] = QCheckBox("Source",self.container)
        is_transparecyBoxvisible = self.transparencyChechBoxList[13].isVisible()
        self.transparencyChechBoxList[13].setVisible(is_transparecyBoxvisible)
        self.transparencyChechBoxList[14] = QCheckBox("Classification",self.container)
        is_transparecyBoxvisible = self.transparencyChechBoxList[14].isVisible()
        self.transparencyChechBoxList[14].setVisible(is_transparecyBoxvisible)
        self.transparencyChechBoxList[15] = QCheckBox("ldaOutput",self.container)
        is_transparecyBoxvisible = self.transparencyChechBoxList[15].isVisible()
        self.transparencyChechBoxList[15].setVisible(is_transparecyBoxvisible)
        self.opticalImages=self.read_images(self.toolPath+"/bigPatches")
        self.initImages=self.read_images(self.toolPath+"/NaturalColorImages")
        self.atmosPenImages=self.read_images(self.toolPath+"/AtmosphericPeneterationImages")
        self.natColAtmosPenRatio=2.0
        
        self.bandImages=self.read_images(self.toolPath+"/readImage")
        self.certainity_numpy =np.random.rand(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim))
        #self.classificationList = ["River"]*(self.transparentButtomXDim*self.transparentButtomYDim)self.opticalImageIndex+=1
        self.classificationList = np.random.randint(0,4,size=(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim)))
        for tempImg in self.initImages:
            print(type(tempImg))
            self.initImagesTruncated.append((tempImg[57:])[:-4])
        self.combo_box_Images.addItems(self.initImagesTruncated)    
        for i in range(100000):
            self.transparentButtom[i] = QPushButton("  ", self.container)#
            self.transparentButtom[i].hide()
            #self.transparentButtom[i].setGeometry(self.leftBuffer+(i%self.transparentButtomXDim)*self.tbuttonwidth, #self.topBuffer+(i//self.transparentButtomYDim)*self.tbuttonHeight, self.tbuttonwidth, self.tbuttonHeight)
        
        
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            #self.transparentButtom[i] = QPushButton("  ", self.container)
           # print('the value iof self.transparentButtomXDim is ',self.transparentButtomXDim)
            
            self.transparentButtom[i].setGeometry(self.leftBuffer+(i%self.transparentButtomXDim)*self.tbuttonwidth, self.topBuffer+(i//self.transparentButtomYDim)*self.tbuttonHeight, self.smallPatchDim, self.smallPatchDim)
            #self.classificationList[i]=i%5


    def load_or_remove_image(self,l):
        if self.bandCheckBoxes[l].isChecked():
            self.load_image(l)
        else:
            self.remove_image(l)

            
    def remove_image(self,l):
        del self.images[f"Band{l+1}"]
        self.transparencyBoxItems.remove(f"Band{l+1}")
        self.combo_box_Transparency.clear()
        self.combo_box_Transparency.addItems(self.transparencyBoxItems)

        if len(self.images)==0:
            self.combo_box_Transparency.hide()
        self.transparencyChechBoxList[l].setVisible(False)
        self.button_group.removeButton(self.transparencyChechBoxList[l])
        self.imageCount+=1
        self.update_image_display()
        
        
    def load_or_remove_optical_image(self,l):
        
        if l==-1:
            if self.sourceImageCheckbox.isChecked():
                self.previousButton.show()
                self.nextButton.show()
                self.load_optical_image(l)
            else:
                self.remove_optical_image(l)
                if (not self.sourceImageCheckbox.isChecked()) and (not self.classificationMapoCheckbox.isChecked()):
                    self.opticalImageIndex=0
                    self.previousButton.hide()
                    self.nextButton.hide()
            
        elif l==-2:
            if self.classificationMapoCheckbox.isChecked():
                self.previousButton.show()
                self.nextButton.show()
                self.load_optical_image(l)
            else:
                self.remove_optical_image(l)
                if (not self.sourceImageCheckbox.isChecked()) and (not self.classificationMapoCheckbox.isChecked()):
                    self.opticalImageIndex=0
                    self.previousButton.hide()
                    self.nextButton.hide()
        else:
            if self.sourceImageCheckbox.isChecked():
                print('checkpoint1')
                self.remove_optical_image(-1)
                self.load_optical_image(-1)
                
            elif self.classificationMapoCheckbox.isChecked():
                print('checkpoint2')
                self.remove_optical_image(-2)
                self.load_optical_image(-2)

            
    def load_optical_image(self,l):
        if l==-1:
            if self.ldaExecuted==1:
                self.add_image(self.opticalImages[self.opticalImageIndex], "Source",l)
                print(self.opticalImages[self.opticalImageIndex])
                self.add_image(self.ldaOutputImages[self.opticalImageIndex], "ldaOutput",l)
                print(self.ldaOutputImages[self.opticalImageIndex])
            else:
                self.add_image(self.opticalImages[self.opticalImageIndex], "Source",l)
                print(self.opticalImages[self.opticalImageIndex])
        elif l==-2:
            self.add_image(self.classificationImages[self.opticalImageIndex], "Classification",l)
            
            
    def remove_optical_image(self,l):
        if l==-1:
            if self.ldaExecuted==1:
                del self.images["Source"]
                del self.images["ldaOutput"]
                self.transparencyBoxItems.remove("Source")
                self.transparencyBoxItems.remove("ldaOutput")
                self.combo_box_Transparency.clear()
                print('ttransparency box value is ',self.transparencyBoxItems)
                self.combo_box_Transparency.addItems(self.transparencyBoxItems)
                self.transparencyChechBoxList[13].setVisible(False)
                self.button_group.removeButton(self.transparencyChechBoxList[13])
                self.transparencyChechBoxList[15].setVisible(False)
                self.button_group.removeButton(self.transparencyChechBoxList[15])
            else:
                del self.images["Source"]     
                
                self.transparencyBoxItems.remove("Source")
                self.combo_box_Transparency.clear()
                self.combo_box_Transparency.addItems(self.transparencyBoxItems)
                self.transparencyChechBoxList[13].setVisible(False)
                self.button_group.removeButton(self.transparencyChechBoxList[13])
        elif l==-2:
            del self.images["Classification"]
            self.transparencyBoxItems.remove("Classification")
            self.combo_box_Transparency.clear()
            self.combo_box_Transparency.addItems(self.transparencyBoxItems)
            self.transparencyChechBoxList[14].setVisible(False)
            self.button_group.removeButton(self.transparencyChechBoxList[14])
        if len(self.images)==0:
           self.combo_box_Transparency.hide()
            
        self.update_image_display()
        




    def load_image(self,l):
    
        if self.bandCheckBoxes[l].isChecked():
           if l<=9:
               sText="B0"+str(l)
           else:
               sText="B"+str(l+1)
           #print(sText)
                
           self.sampleBand = [f for f in self.bandImages if sText in f]
           #print(self.sampleBand)
           self.add_image(self.sampleBand[0], f"Band{l+1}",l) 
           for j in range(13):
               self.bandCheckBoxes[j].setGeometry(self.leftBuffer+((self.buttonWidth*2)*(j%5)), self.imgHeight+(self.topBuffer)*3+(self.buttonHeight*((j//5)+2)), self.checkBoxWidth*2,self.buttonHeight*2)
            

            
            #self.start_labelling(self.certainity_numpy)
            #self.transparentButtomTemp.setGeometry(150,150,100,100)            
           # self.indButton.setGeometry(self.leftBuffer, self.topBuffer, self.indButtonWidth, self.indButtonHeight)
     
    #def move_indicator(self):
     #       return 


     
     
  #  def load_image(self):
    
    
   #     for l in range(13):
            #print(self.bandCheckBoxes[l].isChecked())
    #        if self.bandCheckBoxes[l].isChecked():
     #           self.sampleBand = [f for f in self.bandImages if f"B03" in f]
      #          self.add_image(self.sampleBand[0], f"Band{l+1}") 
            #self.create_checkboxes()
            #self.load_coordinates(l)
       #     for j in range(13):
        #        self.bandCheckBoxes[j].setGeometry(self.leftBuffer+((self.buttonWidth*2)*(j%5)), self.imgHeight+self.topBuffer+(self.buttonHeight*((j//5)+2)), #self.buttonWidth*2,self.buttonHeight*2)
            

            
            #self.start_labelling(self.certainity_numpy)
            #self.transparentButtomTemp.setGeometry(150,150,100,100)            
           # self.indButton.setGeometry(self.leftBuffer, self.topBuffer, self.indButtonWidth, self.indButtonHeight)
     
    #def move_indicator(self):
     #       return 
        
    def move_indicator(self):
        if self.imgWidth == 0 or self.imgHeight == 0:
            return 
        
        self.indX+=self.indXJump
        if self.indX > (self.imgWidth + self.leftBuffer-self.indButtonWidth):
           self.indX=self.leftBuffer
           self.indY+=self.indYJump
        if self.indY > (self.imgHeight + self.topBuffer-self.indButtonHeight):
           self.indX=self.leftBuffer
           self.indY=self.topBuffer
        self.indButton.setGeometry(self.indX, self.indY, self.indButtonWidth, self.indButtonHeight)
        
    def changeLabelFlag(self):
        if self.startedLabelling==0:
            self.startedLabelling=1
            self.start_labelling()
        elif self.startedLabelling==1:
            self.startedLabelling=0
            self.stop_labelling()
            
    def stop_labelling(self):
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            self.transparentButtom[i].hide()
            self.verticalCounter.show()
            self.executeLdaButton.show()
            self.labelVerticalCounter.show()
            
    def update_labelling(self):
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            transparencValue=abs((int(255*self.certainity_numpy[self.opticalImageIndex][i]))-255)
            if transparencValue==1:
                transparencValue=2
            self.transparentButtom[i].setStyleSheet(f"""background-color: rgba(0, 0, 0, {transparencValue});
            border: 1px solid black;""")
        
        
        
    def start_labelling(self):
        if len(self.images)==0:
            return

        #print("Going onto this functrion")
        #print('value of self.smallPatchDim is ',self.smallPatchDim)
        #print(type(self.smallPatchDim))
        #self.gridSize=int(self.smallPatchDim*self.scaling_factor)
        #print('value of self.smallPatchDim is ',self.gridSize)
        #print(type(self.gridSize))
 
        self.tbuttonwidth=self.gridSize
        #print('value of self.imgWidth is ',self.imgWidth)
        #print('value of self.transparentButtomXDim is ',self.transparentButtomXDim)
       # print('imgwidth is ',self.imgWidth)
        #print('tbuttonwidth is ',self.tbuttonwidth)
        self.tbuttonHeight=self.gridSize
        #print('tbuttonHeight is ',self.tbuttonHeight)
        #print('imgHeight is ',self.imgHeight)
        #print('transparentButtomXDim is ',self.transparentButtomXDim)
        #print('transparentButtomYDim is ',self.transparentButtomYDim)
        transparencValue=0
        

       # print('THe value of self.certainity_numpy uis ',self.certainity_numpy)
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            #print('checkpoint1')
            self.transparentButtom[i].setGeometry(self.leftBuffer+(i%self.transparentButtomXDim)*self.tbuttonwidth, self.topBuffer+(i//self.transparentButtomXDim)*self.tbuttonHeight, self.tbuttonwidth, self.tbuttonHeight)
            
            transparencValue=abs((int(255*self.certainity_numpy[self.opticalImageIndex][i]))-255)
            if transparencValue==1:
                transparencValue=2
            print(transparencValue)
            #print('checkpoint2')
            #print ('X coordinate is ', self.leftBuffer+(i%self.transparentButtomXDim)*self.tbuttonwidth)
            #print ('Y coordinate is ', self.topBuffer+(i//self.transparentButtomYDim)*self.tbuttonHeight)
            self.transparentButtom[i].setStyleSheet(f"""background-color: rgba(0, 0, 0, {transparencValue});
            border: 1px solid black;""")
            #self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            #print('checkpoint3')
            self.transparentButtom[i].clicked.connect(lambda _, x=((i%self.transparentButtomXDim)-1)*self.tbuttonwidth, y=((i//self.transparentButtomYDim)-1)*self.tbuttonHeight,w=self.tbuttonwidth*3,h=self.tbuttonHeight*3,boxind=i: self.enlarge_selected_image(x,y,w,h,boxind))
            #print('checkpoint4')
            self.transparentButtom[i].show()
            #print('checkpoint5')
            self.transparentButtom[i].raise_()
            #print('checkpoint6')
            #value,ok=QInputDialog.getText(self.container,"Input Dialog", "SmallPatchDim")
            
        self.verticalCounter.show()
        self.executeLdaButton.show()
        
        #print('checkpoint7')
       # self.horizontalCounter.show()
       # self.labelHorizontalCounter.show()
        self.labelVerticalCounter.show()
        print('checkpoint8')
        
        print('range is ', self.transparentButtomXDim*self.transparentButtomYDim)
        
            
    def enlarge_selected_image(self,x,y,w,h,boxind):
        x=((boxind%self.transparentButtomXDim)-1)*self.tbuttonwidth
        y=((boxind//self.transparentButtomYDim)-1)*self.tbuttonHeight
        w=self.tbuttonwidth*3
        h=self.tbuttonHeight*3
        print('transparency value is ', abs((int(255*self.certainity_numpy[self.opticalImageIndex][boxind]))-255))
        print('certainity value is ',self.certainity_numpy[self.opticalImageIndex][boxind])
        if boxind in self.selectedButtonList:
            self.selectedButtonList.remove(boxind)
            transparencValue=abs((int(255*self.certainity_numpy[self.opticalImageIndex][boxind]))-255)
            if transparencValue==1:
                transparencValue=2
            self.transparentButtom[boxind].setStyleSheet(f"""background-color: rgba(0, 0, 0, {transparencValue});
            border: 1px solid black;""")
        else:
            self.selectedButtonList.append(boxind)
            self.transparentButtom[boxind].setStyleSheet("border: 3px solid blue;")
        
        
        cropped_image = self.canvas.copy(x,y,w,h)
        #self.enlarged_view = EnlargedView(cropped_image.scaled(self.tbuttonwidth*6, self.tbuttonHeight*6, Qt.KeepAspectRatio),(self.tbuttonwidth, self.tbuttonHeight), self)
        #self.enlarged_view.show()
        enlargementratio=300//self.tbuttonwidth
        self.pixmap1 = cropped_image.scaled(self.tbuttonwidth*enlargementratio, self.tbuttonHeight*enlargementratio, Qt.KeepAspectRatio)
        self.focusImgWidth = self.pixmap1.width()
      #  print(self.focusImgWidth/3)
      #  print(self.focusImgWidth//3)
        self.focusImgHeight = self.pixmap1.height()
       # print(self.focusImgHeight/3)
       # print(self.focusImgHeight//3) *self.scaled_ratio self.transparentButtomXDim*self.transparentButtomYDim
       
        print('boxind value is ',boxind)
        print('self.transparentButtomXDim value is ',self.transparentButtomXDim)
        print('boxind%self.transparentButtomXDim value is ',boxind%self.transparentButtomXDim)
        print('self.scaled_ratio value is ',self.scaled_ratio)
        if boxind==0:
            self.focusButton.setGeometry(((self.leftBuffer*4))*1+self.imgWidth,(self.topBuffer*4), self.focusImgWidth//2, self.focusImgHeight//2)
        elif boxind==self.transparentButtomXDim-1:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//2))*1+self.imgWidth,(self.topBuffer*4), self.focusImgWidth//2, self.focusImgHeight//2)
        elif boxind==(self.transparentButtomXDim*self.transparentButtomYDim)-self.transparentButtomXDim:
            self.focusButton.setGeometry(((self.leftBuffer*4))*1+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//2), self.focusImgWidth//2, self.focusImgHeight//2)
        elif boxind==(self.transparentButtomXDim*self.transparentButtomYDim)-1:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//2))*1+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//2), self.focusImgWidth//2, self.focusImgHeight//2)
        elif 0<boxind<self.transparentButtomXDim-1:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//3))*1+self.imgWidth,(self.topBuffer*4), self.focusImgWidth//3, self.focusImgHeight//2)
        elif (self.transparentButtomXDim*self.transparentButtomYDim)-self.transparentButtomXDim<boxind<(self.transparentButtomXDim*self.transparentButtomYDim)-1:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//3))*1+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//2), self.focusImgWidth//3, self.focusImgHeight//2)
        elif boxind%self.transparentButtomXDim==0:
             self.focusButton.setGeometry((self.leftBuffer*4)+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//3), self.focusImgWidth//2, self.focusImgHeight//3)
        elif boxind%self.transparentButtomXDim==self.transparentButtomXDim-1:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//2))*1+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//3), self.focusImgWidth//2, self.focusImgHeight//3)

        
        else:
            self.focusButton.setGeometry(((self.leftBuffer*4)+(self.focusImgWidth//3))*1+self.imgWidth,(self.topBuffer*4)+ (self.focusImgHeight//3), self.focusImgWidth//3, self.focusImgHeight//3)
        self.focusButton.setStyleSheet("background-color: transparent;")
        self.focusButton.setStyleSheet("border: 3px solid red;")
        self.newLabel.setPixmap(self.pixmap1)
        self.newLabel.setScaledContents(True)
        self.newLabel.setGeometry(self.imgWidth+((self.leftBuffer*4)*1), (self.topBuffer*4), self.focusImgWidth, self.focusImgHeight)
        #self.labelClassification.setGeometry(self.imgWidth+((self.leftBuffer*4)*1), (self.topBuffer*7)+self.focusImgHeight, self.buttonWidth*3, self.buttonHeight)
        if self.classificationList[self.opticalImageIndex][boxind]==-1 or self.classificationList[self.opticalImageIndex][boxind]==-2:
            self.labelClassification.setText("Classification: Undefined"+ " \ncertainity: "+str(self.certainity_numpy[self.opticalImageIndex][boxind]*100))
        else:
            self.labelClassification.setText("Classification: "+ self.classificationCodes[self.classificationList[self.opticalImageIndex][boxind]]+ " \ncertainity: "+str(self.certainity_numpy[self.opticalImageIndex][boxind]*100))
        #print(self.classificationCodes[self.classificationList[self.opticalImageIndex][boxind]])

        print(self.labelClassification.text())
        self.labelClassification.setStyleSheet("color: blue; background-color: transparent; font-size: 28px;")
        self.labelClassification.setGeometry(self.imgWidth+((self.leftBuffer*3)*1), (self.topBuffer*5)+self.focusImgHeight, self.buttonWidth*6, self.buttonHeight*3)
        self.focusbox=boxind
        
        
        
    def create_checkboxes(self):
        #print("Entered the function")
        #print(self.main_checkbox.isChecked())
        height=self.InitLabelheight
        
        if self.loadclicked==1:
           height=self.imgHeight
        if self.main_checkbox.isChecked() or True:# and self.loadclicked==1:
            for j in range(13):
                is_visible = self.bandCheckBoxes[j].isVisible()
                self.bandCheckBoxes[j].setVisible(not is_visible)
                self.bandCheckBoxes[j].setGeometry(self.leftBuffer+((self.buttonWidth*2)*(j%5)), height+(self.topBuffer)*3+(self.buttonHeight*((j//5)+2)), self.checkBoxWidth*2,self.buttonHeight)
                
    def load_coordinates(self,k):
        if self.bandCheckBoxes[k].isChecked():
                #print("Entering this part of code")
                #print(self.leftBuffer+self.imgWidth+30,self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.topLeftCoor.setGeometry(self.leftBuffer+self.imgWidth+30,self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.topLeftCoorshow.setGeometry(self.leftBuffer+self.imgWidth+30+(100*1),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.topLeftCoorshow.setText(str(self.imageCoordinates[k][0]))
                self.topRightCoor.setGeometry(self.leftBuffer+self.imgWidth+30+(100*2),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.topRightCoorshow.setGeometry(self.leftBuffer+self.imgWidth+30+(100*3),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.topRightCoorshow.setText(str(self.imageCoordinates[k][1]))
                self.bottomLeftCoor.setGeometry(self.leftBuffer+self.imgWidth+30+(100*4),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.bottomLeftCoorshow.setGeometry(self.leftBuffer+self.imgWidth+30+(100*5),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.bottomLeftCoorshow.setText(str(self.imageCoordinates[k][2]))
                self.bottomRightCoor.setGeometry(self.leftBuffer+self.imgWidth+30+(100*6),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.bottomRightCoorshow.setGeometry(self.leftBuffer+self.imgWidth+30+(100*7),self.imgHeight+self.topBuffer,100,self.buttonHeight)
                self.bottomRightCoorshow.setText(str(self.imageCoordinates[k][3]))
                #print("Exiting this part of code")
                
                

            
            
    def extract_from_zip(self,zippedFIle, extractedToFolder):

        with zipfile.ZipFile(zippedFIle, 'r') as readVar:
            readVar.extractall(extractedToFolder)
            
            
            
            
    def list_of_images(self,foldPath,imagek):

        bandImages=[]
        print(foldPath)

        sizeResize=self.max_size
        defaultsize=512
        ldazoomratio=self.max_size//self.bigPatchDim
        if ldazoomratio>10:
            self.max_size=600
            sizeResize=self.max_size
            ldazoomratio=self.max_size//self.bigPatchDim
        self.scaled_ratio=sizeResize//defaultsize
        
        print("imagek ",self.smallPatchDim)

        imagetype='convertedImages'
        if imagek==1:
            imagetype=self.ldaExperimentName+'/bigPatchesConverted'
            sizeResize=ldazoomratio*self.bigPatchDim
            self.gridSize=ldazoomratio*self.smallPatchDim
        elif imagek==2:
             imagetype=self.ldaExperimentName+'/fullBotConverted'
        elif imagek==3:
             imagetype=self.ldaExperimentName+'/BotConverted'
             sizeResize=ldazoomratio*self.bigPatchDim
        elif imagek==4:
             imagetype=self.labelExperimentName+'/ImagesforLabellingConverted'
             sizeResize=ldazoomratio*self.bigPatchDim
             self.gridSize=ldazoomratio*self.smallPatchDim
             self.imgWidth=sizeResize
             
        print('imagetype value is ',imagetype)
        for root, dirs, files in os.walk(foldPath):
            for file in files:
                try:
                    if file.endswith(".jp2") or file.endswith(".tif") or file.endswith(".jpg") or file.endswith(".png"):
                       image = Image.open(os.path.join(root, file))
                       original_width, original_height = image.size
                       #print('sizeResize value is ', sizeResize)
                       #print('original dimensions are  value is ', original_width,original_height)
                       self.scaling_factor = max(sizeResize / original_width, sizeResize / original_height)
                      # print('self.scaling_factor value is ', self.scaling_factor)
                       resized_img = image.resize((int(original_width * self.scaling_factor), int(original_height * self.scaling_factor)), Image.LANCZOS)
                       resized_img.save(self.toolPath+"/"+imagetype+"/"+file[:-4]+".png")
                       #gdal.Translate(join(root, file), "C:/Users/goya_sh/Desktop/Uncertainity Analysis/Semtinel2_coverted"+file[:-4]+".png")
                       #self.smallPatchDim*=self.smallPatchDim*int(self.scaling_factor)
                       bandImages.append(os.path.join(root, self.toolPath+"/"+imagetype+"/"+file[:-4]+".png"))
                except Exception as e:
                       print(e)
                       print("couldn't convert ",os.path.join(root, file))
        #print(bandImages)
        print('\n')
        print('Exiting list_of_images')
        bandImagesSorted= sorted(bandImages, key=self.extract_number)  
        #print(bandImagesSorted)
        return bandImagesSorted
        
    def read_images(self,foldPath):

        bandImages=[]
        ldazoomratio=1200//self.bigPatchDim
        self.scaled_ratio=1200//512
        if ldazoomratio>10:
            ldazoomratio=10
        sizeResize=ldazoomratio*self.bigPatchDim
        self.gridSize=ldazoomratio*self.smallPatchDim
        self.imgWidth=sizeResize
        for root, dirs, files in os.walk(foldPath):
            for file in files:
                try:
                       bandImages.append(os.path.join(root,file))
                except Exception as e:
                       print(e)
                       print("couldn't convert ",os.path.join(root, file))
        bandImagesSorted= sorted(bandImages, key=self.extract_number) 
        #print(bandImagesSorted)
        return bandImagesSorted
        
        
    def extract_band_data(self,bandFilePath):

        with rasterio.open(bandFilePath) as src:
            bandData = src.read(1)  
            metaData = src.profile  
        return bandData, metaData
        
    def adjust_transparency(self, value):

        transparency_level = value / 100.0  
        pixmap = self.pixmap

        
        transparent_image = QPixmap(pixmap.size())
        transparent_image.fill(Qt.transparent)

        
        painter = QPainter(transparent_image)
        painter.setOpacity(transparency_level)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        self.imgLabel.setPixmap(transparent_image)
        
    def update_image_display(self):

        newsize=self.scaled_ratio*self.imgLabel.size()
        if self.focusImgHeight!=0:
            classLabelHeight=self.focusImgHeight
        else:
            enlargementratio=300//max(50,self.tbuttonwidth)
            classLabelHeight=enlargementratio*max(50,self.tbuttonwidth)
       # print('newsize is ',newsize)
        newimgWidth=self.imgWidth
        newimgHeight=self.imgHeight
        self.imgLabel.setFixedSize(newimgWidth,newimgHeight)
        #print('newimgWidth is ',newimgWidth)
        #print('newimgHeight is ',newimgHeight)
        #print('self.imgWidth is ',self.imgWidth)
        self.canvas = QPixmap(self.imgLabel.size())
        self.canvas.fill(Qt.transparent)

        painter = QPainter(self.canvas)
        for image_key, data in self.images.items():
            pixmap = data["pixmap"]
            transparency = data["transparency"]

            painter.setOpacity(transparency)
            painter.drawPixmap(0, 0, pixmap.scaled(self.imgLabel.size(), Qt.KeepAspectRatio))

        painter.end()
        self.imgLabel.setPixmap(self.canvas)
        self.imgLabel.setGeometry(self.leftBuffer, self.topBuffer,newimgWidth,newimgHeight)
        self.moveIndButton.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth, newimgHeight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.web_view.setGeometry(self.leftBuffer, self.topBuffer, newimgWidth, newimgHeight)
        self.loadButtomLDA.setGeometry((self.leftBuffer+(self.buttonWidth)*0)*1+newimgWidth, self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.loadButtomGMM.setGeometry(((self.leftBuffer+self.buttonWidth)*1)*1+newimgWidth, self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.loadButtomBayesian.setGeometry((self.leftBuffer+(self.buttonWidth)*2)*1+newimgWidth , self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.startLabel.setGeometry(self.leftBuffer+newimgWidth//2-self.buttonWidth//2, newimgHeight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.loadButtomModelE.setGeometry((self.leftBuffer+(self.buttonWidth)*0)*1+newimgWidth, self.topBuffer+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight+self.buttonHeight//2)
        self.loadButtomUncertainityA.setGeometry((self.leftBuffer+(self.buttonWidth)*0)*1+newimgWidth+self.buttonWidth+self.buttonWidth//2, self.topBuffer+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight+self.buttonHeight//2)
        #self.labelClassification.setGeometry(newimgWidth//2-self.buttonWidth, 10, self.buttonWidth*3, self.buttonHeight)
        self.main_checkbox.setGeometry(self.leftBuffer, newimgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        self.sourceImageCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*3, newimgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        
        if self.scaled_ratio==1:
                                self.classificationMapoCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*7, self.imgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
                                self.mapServicesCheckbox.setGeometry(self.leftBuffer+(self.leftBuffer)*11, self.imgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth*2, self.buttonHeight)
        else:
            self.classificationMapoCheckbox.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth*2-self.buttonWidth//2, newimgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight)
            self.mapServicesCheckbox.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth, newimgHeight+(self.topBuffer)*2+self.buttonHeight, self.buttonWidth+self.buttonWidth//2, self.buttonHeight)
        self.previousButton.setGeometry(self.leftBuffer, newimgHeight+(self.topBuffer)*2, self.buttonWidth, self.buttonHeight)
        self.nextButton.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth, newimgHeight+(self.topBuffer)*2, self.buttonWidth, self.buttonHeight)
        self.textBox.setGeometry((self.leftBuffer+(self.buttonWidth)*3)*1+newimgWidth, (self.topBuffer*5)-self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.saveWorkButton.setGeometry((self.leftBuffer+(self.buttonWidth)*4)*1+newimgWidth, self.topBuffer*5, self.buttonWidth, self.buttonHeight)
        self.combo_box.setGeometry((self.leftBuffer+(self.buttonWidth)*3)*1+newimgWidth, self.topBuffer*5, self.buttonWidth, self.buttonHeight)#clearLabelling
        self.loadImageForLabelling.setGeometry(self.leftBuffer+self.buttonWidth, newimgHeight+(self.topBuffer)*2, self.buttonWidth//2, self.buttonHeight)
        self.clearLabelling.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth-self.buttonWidth//2, newimgHeight+(self.topBuffer)*2, self.buttonWidth//2, self.buttonHeight)
        self.combo_box_Transparency.setGeometry(self.leftBuffer+newimgWidth, newimgHeight + (self.topBuffer)*6, self.buttonWidth, self.buttonHeight)
        for y in range(50):
            self.displayClasses[y].setGeometry((classLabelHeight+newimgWidth+self.leftBuffer*5-self.leftBuffer//2)*1+self.buttonWidth*(y%2), self.topBuffer*6+25*(y//2), self.buttonWidth, 20)
        self.combo_box_Images.setGeometry(self.leftBuffer+newimgWidth, self.topBuffer*3, self.buttonWidth, self.buttonHeight)
        self.addLabelButton.setGeometry((self.leftBuffer+(self.buttonWidth)*4)*1+newimgWidth, self.topBuffer*5-self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.labelVerticalCounter.setGeometry((self.leftBuffer+(self.buttonWidth)*2)*1+newimgWidth, self.topBuffer*4-(self.buttonHeight)*2,(self.buttonWidth)*2, self.buttonHeight)
        #self.labelHorizontalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*3+newimgWidth, self.topBuffer*4-(self.buttonHeight)*1,(self.buttonWidth)*1, self.buttonHeight)
        self.verticalCounter.setGeometry((self.leftBuffer+(self.buttonWidth)*3)*1+newimgWidth, self.topBuffer*4-(self.buttonHeight)*2, self.buttonWidth, self.buttonHeight)
        self.executeLdaButton.setGeometry((self.leftBuffer+(self.buttonWidth)*4)*1+newimgWidth, self.topBuffer*4-(self.buttonHeight)*2, self.buttonWidth, self.buttonHeight)
        self.clearLabel.setGeometry(self.leftBuffer+newimgWidth-self.buttonWidth, newimgHeight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        self.showLabelButtons.setGeometry(self.leftBuffer, newimgHeight+self.topBuffer, self.buttonWidth, self.buttonHeight)
        #self.horizontalCounter.setGeometry(self.leftBuffer+(self.buttonWidth)*4+newimgWidth, self.topBuffer*4-(self.buttonHeight)*1, self.buttonWidth, self.buttonHeight)
        self.transparency_slider.setGeometry(self.leftBuffer, newimgHeight + (self.topBuffer)*6, newimgWidth-20, 30)
        
        
        
        for j in range(13):

                self.bandCheckBoxes[j].setGeometry(self.leftBuffer+((self.buttonWidth*2)*(j%5)), newimgHeight+(self.topBuffer)*3+(self.buttonHeight*((j//5)+2)), self.checkBoxWidth*2,self.buttonHeight)
                if self.main_checkbox.isChecked():
                    self.bandCheckBoxes[j].show()
                else:
                    self.bandCheckBoxes[j].hide()
        imageCount=0
        for cb in self.button_group.buttons():
            cb.setGeometry(self.leftBuffer+newimgWidth, (self.topBuffer)*2+self.transparentCBBuffer*(imageCount+2), self.checkBoxWidth*2, self.buttonHeight)
            #cb.setVisible(True)
            imageCount+=1

        
        
        
    def add_image(self, image_path, label,indi):

        pixmap = QPixmap(image_path)
        self.images[label] = {"pixmap": pixmap, "transparency": 1.0}
        #self.imgHeight=pixmap.height()
        #print('self.imgHeight is ',self.imgHeight)
        #print('self.imgWidth is ',self.imgWidth)
        self.imgHeight=self.imgWidth
        
        #self.imgWidth=pixmap.width()
        self.transparencyBoxItems.append(label)
        self.combo_box_Transparency.clear()
        self.combo_box_Transparency.addItems(self.transparencyBoxItems)
        self.combo_box_Transparency.show()


        #print(self.transparencyChechBoxList[indi])
        
        if self.main_checkbox.isChecked() and indi not in (-1,-2):
            #self.transparencyChechBoxList.append(label)
            self.button_group.addButton(self.transparencyChechBoxList[indi])
            self.imageCount+=1
        elif indi ==-1:
            self.button_group.addButton(self.transparencyChechBoxList[13])
            self.button_group.addButton(self.transparencyChechBoxList[15])
            self.imageCount+=1
        elif indi ==-2:
            self.button_group.addButton(self.transparencyChechBoxList[14])
            self.imageCount+=1

        self.update_image_display()

    def on_checkbox_clicked(self, checkbox):

        self.current_image_key = checkbox.text()
        
    def transparencyImageSelection(self, index):
        self.current_image_key = self.combo_box_Transparency.itemText(index) 

    def adjust_transparency(self, value):

        if self.current_image_key:
            if self.current_image_key not in self.images.keys():
                return
            transparency_level = value / 100.0
            self.images[self.current_image_key]["transparency"] = transparency_level
            self.update_image_display()
            
            
    def selectClass(self, index):
    
        self.selectedClass=index
        
    def saveWork(self):
    
        text = self.combo_box.itemText(self.selectedClass)
        
        #print(self.combo_box.itemText(self.selectedClass))
        #print(self.selectedClass)
        #print(self.focusbox)self.opticalImageIndex

        if self.classificationList[self.opticalImageIndex][self.focusbox] not in (-1,-2) and self.classificationCodes[self.classificationList[self.opticalImageIndex][self.focusbox]]==text[0:-4]:
            self.labelClassification.setStyleSheet("color: green; background-color: transparent; font-size: 28px;")
        else:
            self.labelClassification.setStyleSheet("color: red; background-color: transparent; font-size: 28px;")
           # print()
            for i in self.selectedButtonList:
                self.classificationList[self.opticalImageIndex][i]=int(text[-1])
                self.certainity_numpy[self.opticalImageIndex][i]=1
            
            
        #data = asarray(self.classificationList)
        #savetxt('data.csv', data, delimiter=',')
        #data1 = loadtxt('data.csv', delimiter=',')self.allLdaOutputImages=self.list_of_images("C:/Users/goya_sh/Desktop/Neuer Ordner"+"/"+self.ldaExperimentName+"/BoT",3)
        patchsizeDictionaries={'BP':self.bigPatchDim,'SP':self.smallPatchDim,'GS':self.gridSize,'IW':self.imgWidth}
        np.save(self.toolPath+"/"+self.experimentName+"/classificationList.npy",self.classificationList)
        np.save(self.toolPath+"/"+self.experimentName+"/certainityNumpyList.npy",self.certainity_numpy )
        with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "wb") as file:
            pickle.dump(self.classificationCodes, file)
        with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "wb") as file:
            pickle.dump(patchsizeDictionaries, file)
            
        loadclasslist=np.load("classificationList.npy")
        
        
        print(loadclasslist)
        self.clearSelection()
        #print(data1)
        
        
    def addLabel(self):
        if self.textBox.text()=="":
           return
           
        print(self.textBox.text())
        print(self.classificationCodes.values())
        
        if re.sub(r'[^a-zA-Z0-9]+', '', self.textBox.text().lower()) in [x.lower() for x in self.classificationCodes.values()]:
            return

    
        self.classificationCodes[max(self.classificationCodes.keys())+1]=re.sub(r'[^a-zA-Z0-9]+', '', self.textBox.text())
        self.displayClasses[max(self.classificationCodes.keys())].setText(str(max(self.classificationCodes.keys()))+' - '+self.classificationCodes[max(self.classificationCodes.keys())])
        self.comboBoxItems=[]
        
        self.combo_box.clear()
        for ik in self.classificationCodes.items():
            self.comboBoxItems.append(ik[1]+' - '+str(ik[0]))
        self.combo_box.addItems(self.comboBoxItems)
        self.textBox.setText('')
        
    def setNewSmallPatchsize(self, vNo):
        self.smallPatchDimVariable=vNo
       # self.container,"Are you sure you want to overwrite current LDA project with a newone?",QMessageBox.Yes | 
    def verticalBoxesChange (self):
        reply=QMessageBox.question(self.container,"Confirmation","Are you sure you want to overwrite current LDA project with a newone?",QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
        if reply==QMessageBox.No:
            return
        elif reply==QMessageBox.Yes:
            self.smallPatchDim=self.smallPatchDimVariable
            for i in range(100000):
                self.transparentButtom[i].hide()
            if self.sourceImageCheckbox.isChecked():
                self.sourceImageCheckbox.setChecked(False)
            if self.classificationMapoCheckbox.isChecked():
                self.classificationMapoCheckbox.setChecked(False)
            self.executeLDA()
        
        
        
       # self.transparentButtomXDim=self.imgWidth//self.gridSize
       # self.transparentButtomYDim=self.imgHeight//self.gridSize
       # self.tbuttonHeight=self.gridSize
        #self.certainity_numpy =np.random.rand(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim))
        #self.tbuttonwidth=self.gridSize
        #s#elf.tbuttonHeight=self.gridSize
        #self.certainity_numpy = [random.randint(0, 255) for _ in range(self.transparentButtomXDim*self.transparentButtomYDim)]
        #for i in range(100000):
            #self.transparentButtom[i] = QPushButton("  ", self.container)#
        #    self.transparentButtom[i].hide()
            
        #if self.sourceImageCheckbox.isChecked():
        #    self.sourceImageCheckbox.setChecked(False)
            
        #if self.classificationMapoCheckbox.isChecked():
        #    self.classificationMapoCheckbox.setChecked(False)
        #self.executeLDA()
       # print('Total numnber of boxwa ', self.transparentButtomXDim*self.transparentButtomYDim)
       # for j in range(self.transparentButtomXDim*self.transparentButtomYDim):
            #self.transparentButtom[i] = QPushButton("  ", self.container)
        #    self.transparentButtom[j].setGeometry(self.leftBuffer+(j%self.transparentButtomXDim)*self.tbuttonwidth, self.topBuffer+(j//self.transparentButtomXDim)*self.tbuttonHeight, self.tbuttonwidth, self.tbuttonHeight) 
         #   self.transparentButtom[j].setStyleSheet(f"background-color: rgba(0, 0, 0, {self.certainity_numpy[self.opticalImageIndex][j]});")            
          #  self.transparentButtom[j].show()
            
            
        
        
        
    def horizontalBoxesChange (self, hNo):
        self.transparentButtomYDim=hNo
        self.tbuttonwidth=self.imgWidth//self.transparentButtomXDim
        self.tbuttonHeight=self.imgHeight//self.transparentButtomYDim
        self.certainity_numpy = np.random.rand(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim))
        for i in range(100000):
           # self.transparentButtom[i] = QPushButton("  ", self.container)#
            self.transparentButtom[i].hide()
       # print('Total numnber of boxwa ', self.transparentButtomXDim*self.transparentButtomYDim)
        for j in range(self.transparentButtomXDim*self.transparentButtomYDim):
            #self.transparentButtom[i] = QPushButton("  ", self.container)
            self.transparentButtom[j].setGeometry(self.leftBuffer+(j%self.transparentButtomXDim)*self.tbuttonwidth, self.topBuffer+(j//self.transparentButtomXDim)*self.tbuttonHeight, self.tbuttonwidth, self.tbuttonHeight)
            self.transparentButtom[j].setStyleSheet(f"background-color: rgba(0, 0, 0, {self.certainity_numpy[self.opticalImageIndex][j]});")
            self.transparentButtom[j].show()
            
            
            
            
    def getLdaVariables(self):
        if self.labelExperimentName!='' or self.gmmExperimentName!='':
            return
        self.ldaExecuted=1
    
        value,ok=QInputDialog.getText(self.container,"Input Dialog", "Enter Experiment name")
        if ok:
            self.ldaExperimentName=value
        else:
            self.ldaExperimentName="LDA Experiment"
        self.experimentName=self.ldaExperimentName
        
        if os.path.isdir(self.toolPath+"/"+self.ldaExperimentName) and self.labelExperimentName=='':
            self.allOpticalImages=self.read_images(self.toolPath+"/"+self.ldaExperimentName+"/ImagesforLabelling")
            self.combo_box_Images.show()
            self.opticalImages=self.allOpticalImages
            self.allLdaOutputImagesUnsorted=self.read_images(self.toolPath+"/"+self.ldaExperimentName+"/BotConverted")
            self.allLdaOutputImages=sorted(self.allLdaOutputImagesUnsorted, key=lambda x: int(re.search(r'(\d+)(?=\.\w+$)', x).group()))
            self.ldaOutputImages=self.allLdaOutputImages
            self.allClassificationImages=self.read_images(self.toolPath+"/Images")
            self.classificationImages=self.allClassificationImages
            self.noOfBigPatches=len(self.allOpticalImages)//len(self.initImages)
            print('length of self.opticalImages is ',len(self.opticalImages))

            for root, dirs, files in os.walk(self.toolPath+"/"+self.ldaExperimentName):
                for file in files:
                    if file.endswith(".npy"):
                        if 'classificationList.npy'==file:
                            self.classificationList=np.load(os.path.join(root, file))
                            print('Öoaded classification list')
                            print('the path is ',os.path.join(root, file))
                            print('classification list is ',self.classificationList)
                        elif 'certainityNumpyList.npy'==file:
                            
                            self.certainity_numpy=np.load(os.path.join(root, file))
                            self.isNewLDAProject=0
                            
                    if file.endswith(".pkl"):
                        if 'classes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "rb") as file1:
                                self.classificationCodes = pickle.load(file1)
                                print(self.classificationCodes)
                            self.combo_box.clear()
                            self.comboBoxItems.clear()
                            for ik in self.classificationCodes.items():
                                self.comboBoxItems.append(ik[1]+' - '+str(ik[0]))
                                self.displayClasses[ik[0]].setText(str(ik[0])+' - '+ik[1])
                            print(self.comboBoxItems)
                            self.combo_box.addItems(self.comboBoxItems)
                        if 'patchSizes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "rb") as file2:
                                patchsizeDictionaries=pickle.load(file2)
                                self.bigPatchDim=patchsizeDictionaries['BP']
                                self.smallPatchDim=patchsizeDictionaries['SP']
                                self.gridSize=patchsizeDictionaries['GS']
                                self.imgWidth=patchsizeDictionaries['IW']
                                print('the imagewidth after loading is ',self.imgWidth)
                                self.transparentButtomXDim=self.imgWidth//self.gridSize
                                self.transparentButtomYDim=self.imgWidth//self.gridSize
       
                    
            return
        self.isNewLDAProject=1
            
            
            
        #value,ok=QInputDialog.getText(self.container,"Input Dialog", "Enter number of bands")
        #if ok:
        #    self.bandNumber=value
            
        value,ok=QInputDialog.getText(self.container,"Input Dialog", "assumptions of number of classes (optional) ")
        if ok:
            self.noOfTopics=value
        else:
            self.noOfTopics=4
            
        value,ok=QInputDialog.getText(self.container,"Input Dialog", "BigPatchDim")
        if ok:
            self.bigPatchDim =int(value)
        else:
            self.bigPatchDim =128
        value,ok=QInputDialog.getText(self.container,"Input Dialog", "SmallPatchDim")
        if ok:
            self.smallPatchDim =int(value)
        else:
            self.smallPatchDim =4   
                
        self.executeLDA()
            
            
    def executeLDA(self):
        

        
        ldaoutputlist=[]
        self.verticalCounter.setValue(self.smallPatchDim)
        naturalImage = Image.open(self.initImages[0])
        naturalImagewidth, naturalImageheight = naturalImage.size
        with rasterio.open(self.atmosPenImages[0]) as src:
            atmosPenImagewidth=src.width
            print('Width of atmosPenImage is',atmosPenImagewidth)
        
        self.natColAtmosPenRatio=naturalImagewidth//atmosPenImagewidth
        opticaloutputlist=[]        
        print('Before Exection')   
        self.combo_box_Images.show()
        print('Before callinf LDA ',self.ldaExperimentName,' ',self.noOfTopics,' ', self.bigPatchDim,' ', self.smallPatchDim)
        
        lDA_last_2025latest.runLDA(self.ldaExperimentName,self.noOfTopics,self.bigPatchDim,self.smallPatchDim)
        self.loadImagesAndLabelling()
        os.makedirs("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.ldaExperimentName+"/bigPatchesConverted", exist_ok=True)
        os.makedirs("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.ldaExperimentName+"/fullBotConverted", exist_ok=True)
        os.makedirs("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.ldaExperimentName+"/BotConverted", exist_ok=True)
        #lda_changes2023.find_changes("C:/Users/goya_sh/Desktop/Neuer Ordner/Images",5490,5490,128,4)
        print('Before Exection') 
        self.allOpticalImages=self.read_images(self.toolPath+"/"+self.ldaExperimentName+"/ImagesforLabelling")
        self.opticalImages=self.allOpticalImages
        self.noOfBigPatches=len(self.allOpticalImages)//len(self.initImages)
        print('length of self.opticalImages is ',len(self.opticalImages))
        #self.allClassificationImages=self.list_of_images("self.toolPath+"/"+self.ldaExperimentName,2)
        self.allClassificationImages=self.read_images(self.toolPath+"/Images")
        self.classificationImages=self.allClassificationImages
        self.allLdaOutputImagesUnsorted=self.list_of_images(self.toolPath+"/"+self.ldaExperimentName+"/BoT",3)
        self.allLdaOutputImages=sorted(self.allLdaOutputImagesUnsorted, key=lambda x: int(re.search(r'(\d+)(?=\.\w+$)', x).group()))
        self.ldaOutputImages=self.allLdaOutputImages
       # self.certainity_numpy =np.random.randint(0,255,size=(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim)))
        self.transparentButtomXDim=self.imgWidth//self.gridSize
        self.transparentButtomYDim=self.imgWidth//self.gridSize
        if self.isNewLDAProject==1:
            self.certainity_numpy =np.random.rand(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim))
            self.classificationList = np.random.randint(-2,-1,size=(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomYDim)))
        print('the values are ',self.bigPatchDim,' ',self.smallPatchDim,' ',self.gridSize,' ',self.imgWidth,' ', self.transparentButtomXDim,' ', self.transparentButtomYDim)
        patchsizeDictionaries={'BP':self.bigPatchDim,'SP':self.smallPatchDim,'GS':self.gridSize,'IW':self.imgWidth}
        np.save(self.toolPath+"/"+self.experimentName+"/classificationList.npy",self.classificationList)
        np.save(self.toolPath+"/"+self.experimentName+"/certainityNumpyList.npy",self.certainity_numpy )
        with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "wb") as file:
            pickle.dump(self.classificationCodes, file)
        with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "wb") as file:
            pickle.dump(patchsizeDictionaries, file)

        #return
    def getGmmVariables(self): 
        if self.labelExperimentName!='' or self.ldaExperimentName!='':
            return

        value,ok=QInputDialog.getText(self.container,"Input Dialog", "Enter Experiment name")
        if ok:
            self.gmmExperimentName=value
        else:
            self.gmmExperimentName="GMM Experiment"
        self.experimentName=self.gmmExperimentName   
        if os.path.isdir(self.toolPath+"/"+self.gmmExperimentName):
            self.allOpticalImages=self.read_images(self.toolPath+"/"+self.gmmExperimentName+"/sourcePatches")
            #self.combo_box_Images.show()
            self.opticalImages=self.allOpticalImages
            self.allClassificationImages=self.read_images(self.toolPath+"/Images")
            self.classificationImages=self.allClassificationImages
            self.noOfBigPatches=len(self.allOpticalImages)//len(self.initImages)
            for root, dirs, files in os.walk(self.toolPath+"/"+self.gmmExperimentName):
                for file in files:
                    if file.endswith(".npy"):
                        if 'classificationList.npy'==file:
                            self.classificationList=np.load(os.path.join(root, file))
                            print('Öoaded classification list')
                            print('the path is ',os.path.join(root, file))
                            print('classification list is ',self.classificationList)
                        elif 'certainityNumpyList.npy'==file:
                            
                            self.certainity_numpy=np.load(os.path.join(root, file))
                            self.isNewGMMProject=0
                            
                    if file.endswith(".pkl"):
                        if 'classes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "rb") as file1:
                                self.classificationCodes = pickle.load(file1)
                                print(self.classificationCodes)
                            self.combo_box.clear()
                            self.comboBoxItems.clear()
                            for ik in self.classificationCodes.items():
                                self.comboBoxItems.append(ik[1]+' - '+str(ik[0]))
                                self.displayClasses[ik[0]].setText(str(ik[0])+' - '+ik[1])
                            print(self.comboBoxItems)
                            self.combo_box.addItems(self.comboBoxItems)
                        if 'patchSizes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "rb") as file2:
                                patchsizeDictionaries=pickle.load(file2)
                                self.bigPatchDim=patchsizeDictionaries['BP']
                                self.smallPatchDim=patchsizeDictionaries['SP']
                                self.gridSize=patchsizeDictionaries['GS']
                                self.imgWidth=patchsizeDictionaries['IW']
                                print('the imagewidth after loading is ',self.imgWidth)
                                self.transparentButtomXDim=self.imgWidth//self.gridSize
                                self.transparentButtomYDim=self.imgWidth//self.gridSize
       
                    
            return
        gmmInputPath = "C:/Users/goya_sh/Desktop/Neuer Ordner/gmmInput"
        os.makedirs("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName, exist_ok=True)
        os.makedirs("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/sourcePatches", exist_ok=True)
        for root, dirs, files in os.walk("C:/Users/goya_sh/Desktop/Neuer Ordner/gmmInput"):
            for file in files:
                if file.endswith(".png"):
                    image = Image.open(os.path.join(root, file))
                    patch_filename = os.path.join("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/sourcePatches", f"{file[:-4]}.png")
                    image.save(patch_filename)
                
        gmmFilePaths = [os.path.join(gmmInputPath, file) for file in os.listdir(gmmInputPath) if file.endswith('.npy')]
        merged_npy = np.stack([np.load(file) for file in gmmFilePaths], axis=0)
        originalShape=merged_npy.shape
        print(type(originalShape))
        originalShapetrunc=(originalShape[0],originalShape[1],originalShape[2])
        print(originalShapetrunc)
        np.save(os.path.join("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName, "merged_npy.npy"), merged_npy)
        self.allOpticalImages=self.read_images("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/sourcePatches")
        self.opticalImages=self.allOpticalImages
        self.allClassificationImages=self.read_images(self.toolPath+"/Images")
        self.classificationImages=self.allClassificationImages
        print('opticalImages size is ',len(self.opticalImages))
        self.executeGMM(originalShapetrunc)
        
    def executeGMM(self,originalShape): 
        gmm.runGMM(self.gmmExperimentName)
        gm_labels=np.load("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/img_1_gmlabels10.npy")
        gmLabelsReshaped=gm_labels.reshape(originalShape)
        print(gmLabelsReshaped.shape)
        certainityData=np.load("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/img_1_gm_certainty_max10.npy")
        self.certainity_numpy=certainityData.reshape(originalShape[0],originalShape[1]*originalShape[2])
        #classificationData=np.load("C:/Users/goya_sh/Desktop/Neuer Ordner/"+self.gmmExperimentName+"/img_1_gmlabels10.npy")
       # print('classificationData shape is ',classificationData.shape)
        self.classificationList=gm_labels.reshape(originalShape[0],originalShape[1]*originalShape[2])
        print('self.certainity_numpy shape is ',self.certainity_numpy.shape)
        print('self.classificationList shape is ',self.classificationList.shape)
        self.transparentButtomXDim=originalShape[1]
        print('self.transparentButtomXDim value is ',self.transparentButtomXDim)
        self.transparentButtomYDim=originalShape[2]
        print('self.transparentButtomYDim value is ',self.transparentButtomYDim)
        gmmzoomratio=1200//64
        print('gmmzoomratio value is ',gmmzoomratio)
        self.gridSize=1*gmmzoomratio
        np.save(self.toolPath+"/"+self.experimentName+"/classificationList.npy",self.classificationList)
        np.save(self.toolPath+"/"+self.experimentName+"/certainityNumpyList.npy",self.certainity_numpy )
        patchsizeDictionaries={'BP':self.bigPatchDim,'SP':self.smallPatchDim,'GS':self.gridSize,'IW':self.imgWidth}
        with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "wb") as file:
            pickle.dump(self.classificationCodes, file)
        with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "wb") as file:
            pickle.dump(patchsizeDictionaries, file)
        
    def moveToPrevious(self):
        self.clearLabels()
        self.opticalImageIndex-=1
        if self.opticalImageIndex==-1:
            self.opticalImageIndex+=1
            return
        self.load_or_remove_optical_image(self.opticalImageIndex)
        self.clearSelection()
        if self.startedLabelling==1:
            self.update_labelling()
        
        
    def moveToNext(self):
        self.clearLabels()
        self.opticalImageIndex+=1
        print(len(self.opticalImages)==self.opticalImageIndex)
        if len(self.opticalImages)==self.opticalImageIndex:
            self.opticalImageIndex-=1
            return
        print('calling the function')    
        self.load_or_remove_optical_image(self.opticalImageIndex)
        self.clearSelection()
        if self.startedLabelling==1:
            self.update_labelling()
        

        
    def backGroundImage(self,image_path):
        self.setStyleSheet(f"QMainWindow {{background-image:url({image_path}); background-repeat: no-repeat;background-position:center; }}")
        
    def clearSelection(self):
        for i in self.selectedButtonList:
            self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
        self.selectedButtonList.clear()
        
    def load_map(self):

        #s='a'
        #z=2
        #x=1
        #y=1
        
        latitude=self.latitude
        longitude=self.longitude
        zoom_level=13
        self.web_view.show()
        self.indButton.hide()
        if self.mapServicesCheckbox.isChecked():
            self.web_view.show()
        else:
            self.web_view.hide()
            return
        

        osm_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
          <style>
            #map {{ height: 100%; }}
            html, body {{ height: 100%; margin: 0; padding: 0; }}
          </style>
          <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
          <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
        </head>
        <body>
          <div id="map"></div>
          <script>
            var map = L.map('map').setView([{latitude}, {longitude}], {zoom_level});
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
              maxZoom: 19,
              attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
          </script>
        </body>
        </html>
        """
        self.web_view.setHtml(osm_html)
        self.imgLabel.raise_()
        print('Coordinates loaded')
        
    def showLdaImages(self, index):
        self.noOfBigPatches=len(self.allOpticalImages)//len(self.initImages)
        self.opticalImages=self.allOpticalImages[self.noOfBigPatches*index:self.noOfBigPatches*(index+1)]
        if self.ldaExecuted==1:
            self.ldaOutputImages=self.allLdaOutputImages[self.noOfBigPatches*index:self.noOfBigPatches*(index+1)]

        print('no of big patches ',self.noOfBigPatches)
        print('index value ',index)
        print('start range  ',self.noOfBigPatches*index)
        print('end range  ',self.noOfBigPatches*(index+1))
        print('First optical imah´ge is ', self.opticalImages[0])
        #self.classificationImages=self.allClassificationImages[self.noOfBigPatches*index:self.noOfBigPatches*(index+1)]
        #self.ldaOutputImages=self.allLdaOutputImages[self.noOfBigPatches*index:self.noOfBigPatches*(index+1)]
        #print('First LDA imah´ge is ', self.ldaOutputImages[0])
        #print('index of ',self.ldaOutputImages[0],' is ', self.ldaOutputImages.index(self.ldaOutputImages[0]))

            
        self.opticalImageIndex=0
        if self.sourceImageCheckbox.isChecked():
            self.sourceImageCheckbox.setChecked(False)
            #self.load_or_remove_optical_image(-1)
            self.sourceImageCheckbox.setChecked(True)
           # self.load_or_remove_optical_image(-1)
            
        if self.classificationMapoCheckbox.isChecked():
            self.classificationMapoCheckbox.setChecked(False)
            #self.load_or_remove_optical_image(-2)
            self.classificationMapoCheckbox.setChecked(True)
            #self.load_or_remove_optical_image(-2)
            
    def initiateLabelling(self):
        
        value,ok=QInputDialog.getText(self.container,"Input Dialog", "Enter Experiment name")
        if ok:
            self.labelExperimentName=value
        else:
            self.labelExperimentName="Label Experiment"
        self.experimentName=self.labelExperimentName
        self.loadImagesAndLabelling()
        self.allOpticalImages=self.read_images(self.toolPath+"/"+self.labelExperimentName+"/ImagesforLabelling")
        #if self.isNewLabelProject==1:
        # #   self.allOpticalImages=self.read_images(self.toolPath+"/"+self.labelExperimentName+"/ImagesforLabelling")
        #  #  print('Exited list_of_images')
        #else:
        #    self.allOpticalImages=self.read_images(self.toolPath+"/"+self.labelExperimentName+"/ImagesforLabellingConverted")
        print('Assigning optical images')
        self.opticalImages=self.allOpticalImages
        self.combo_box_Images.show()
        print('Images loaded successfully')
        self.transparentButtomXDim=self.imgWidth//self.gridSize
        self.transparentButtomYDim=self.imgWidth//self.gridSize
        if self.isNewLabelProject==1:
            self.certainity_numpy =np.random.rand(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim))
            self.classificationList = np.random.randint(-2,-1,size=(len(self.opticalImages),(self.transparentButtomXDim*self.transparentButtomXDim)))
        patchsizeDictionaries={'BP':self.bigPatchDim,'SP':self.smallPatchDim,'GS':self.gridSize,'IW':self.imgWidth}
        np.save(self.toolPath+"/"+self.experimentName+"/classificationList.npy",self.classificationList)
        np.save(self.toolPath+"/"+self.experimentName+"/certainityNumpyList.npy",self.certainity_numpy )
        with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "wb") as file:
            pickle.dump(self.classificationCodes, file)
        with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "wb") as file:
            pickle.dump(patchsizeDictionaries, file)


    def loadImagesAndLabelling(self):
        

        input_folder = self.toolPath+"/NaturalColorImages"
        output_folder = self.toolPath+"/"+self.experimentName
        
        if os.path.isdir(output_folder) and self.ldaExperimentName=='':            
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file.endswith(".npy"):
                        if 'classificationList.npy'==file:
                            self.classificationList=np.load(os.path.join(root, file))
                            print('Öoaded classification list')
                        elif 'certainityNumpyList.npy'==file:
                            self.certainity_numpy=np.load(os.path.join(root, file))
                            self.isNewLabelProject=0
                            print(self.classificationList)
                    if file.endswith(".pkl"):
                        if 'classes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/classes.pkl", "rb") as file1:
                                self.classificationCodes = pickle.load(file1)
                            print(self.classificationCodes)
                            self.combo_box.clear()
                            self.comboBoxItems.clear()
                            for ik in self.classificationCodes.items():
                                self.comboBoxItems.append(ik[1]+' - '+str(ik[0]))
                                self.displayClasses[ik[0]].setText(str(ik[0])+' - '+ik[1])
                            print(self.comboBoxItems)
                            self.combo_box.addItems(self.comboBoxItems)
                        if 'patchSizes.pkl'==file:
                            with open(self.toolPath+"/"+self.experimentName+"/patchSizes.pkl", "rb") as file2:
                                patchsizeDictionaries=pickle.load(file2)
                                self.bigPatchDim=patchsizeDictionaries['BP']
                                self.smallPatchDim=patchsizeDictionaries['SP']
                                self.gridSize=patchsizeDictionaries['GS']
                                self.imgWidth=patchsizeDictionaries['IW']

       
                    
            return

        os.makedirs(output_folder+"/ImagesforLabelling", exist_ok=True)
        os.makedirs(output_folder+"/ImagesforLabellingConverted", exist_ok=True)
        os.makedirs(output_folder+"/Images", exist_ok=True)
        self.geotiff_to_png(self.toolPath+"/NaturalColorImages", output_folder+"/Images")
        self.isNewLabelProject=1

        patch_size = int(self.bigPatchDim*self.natColAtmosPenRatio)
        print('patch_size is ',patch_size)
        ldazoomratio=self.max_size//patch_size
        if ldazoomratio>10:
            self.max_size=600
            ldazoomratio=self.max_size//self.bigPatchDim
        sizeResize=ldazoomratio*patch_size
        self.imgWidth=ldazoomratio*patch_size
        self.gridSize=self.smallPatchDim*self.natColAtmosPenRatio*ldazoomratio
        scalingFactor = sizeResize / patch_size
        
        
        for root, dirs, files in os.walk(output_folder+"/Images"):
            for file in files:

                image = Image.open(os.path.join(root, file))
                width, height = image.size
                print('natural image width is ',width)
                print('natural image height is ',height)


                patch_num = 0
                for y in range(0, height-patch_size+1, patch_size):
                    for x in range(0, width-patch_size+1, patch_size):
                       # print('width-patch_size is ',width-patch_size)
                       # print('height-patch_size is ',height-patch_size)
                       # print('big patch created dimensions are ',x, ' ',y,' ', x + patch_size, ' ',y + patch_size)
                        box = (x, y, x + patch_size, y + patch_size)
                        patch = image.crop(box)

                        patch_filename = os.path.join(output_folder+"/ImagesforLabelling", f"{file[:-4]}{patch_num}.png")
                        resized_patchImage = patch.resize((int(patch_size * scalingFactor), int(patch_size * scalingFactor)), Image.LANCZOS)
                        resized_patchImage.save(patch_filename)
                        patch_num += 1
        
        
    def geotiff_to_png(self,input_folder, output_folder):
        for file in os.listdir(input_folder):
            input_tiff=os.path.join(input_folder,file)
            output_png=os.path.join(output_folder,file.replace(".tif",".png").replace(".tiff",".png"))
            
            
            with rasterio.open(input_tiff) as src:
                image_array = src.read()  
                profile = src.profile  
               
                if image_array.shape[0] >= 3:
                    image_array = image_array[:3]  
                else:
                    image_array = np.repeat(image_array, 3, axis=0)

                image_array = image_array.astype(np.float32)
                min_val = image_array.min()
                max_val = image_array.max()
               
                if max_val > min_val:  
                    image_array = 255 * (image_array - min_val) / (max_val - min_val)
               
                image_array = image_array.astype(np.uint8)

                image_array = np.moveaxis(image_array, 0, -1)  

                img = Image.fromarray(image_array)
                img.save(output_png)
                print(f"Saved {output_png}")
        
    def extract_number(self,filepath):
        filename = os.path.basename(filepath)  
        match = re.match(r'(\d+)?([a-zA-Z]+)?(\d+)?(?=\.\w+$)?', filename)  

        start_num = int(match.group(1)) if match.group(1) else float('-inf')  
        text = match.group(2) if match.group(2) else ""  
        end_num = int(match.group(3)) if match.group(3) else float('-inf')  

        return (start_num, text, end_num) 
        
    def showLabels (self):
        self.clearLabels()
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            if self.classificationList[self.opticalImageIndex][i]==0:
                self.transparentButtom[i].setStyleSheet("border: 3px solid red;")
            elif self.classificationList[self.opticalImageIndex][i]==1:
                self.transparentButtom[i].setStyleSheet("border: 3px solid green;")
            elif self.classificationList[self.opticalImageIndex][i]==2:
                self.transparentButtom[i].setStyleSheet("border: 3px solid yellow;")
            elif self.classificationList[self.opticalImageIndex][i]==3:
                self.transparentButtom[i].setStyleSheet("border: 3px solid magenta;")
            elif self.classificationList[self.opticalImageIndex][i]==4:
                self.transparentButtom[i].setStyleSheet("border: 3px solid cyan;")
            elif self.classificationList[self.opticalImageIndex][i]==5:
                self.transparentButtom[i].setStyleSheet("border: 3px solid orange;")
            elif self.classificationList[self.opticalImageIndex][i]==6:
                self.transparentButtom[i].setStyleSheet("border: 3px solid purple;")
            elif self.classificationList[self.opticalImageIndex][i]==7:
                self.transparentButtom[i].setStyleSheet("border: 3px solid teal;")
            elif self.classificationList[self.opticalImageIndex][i]==8:
                self.transparentButtom[i].setStyleSheet("border: 3px solid brown;")
            elif self.classificationList[self.opticalImageIndex][i]==9:
                self.transparentButtom[i].setStyleSheet("border: 3px solid crimson;")
                
        for key in self.classificationCodes.keys():
            print('key is ',key)
            print('len(self.displayClasses is ',len(self.displayClasses))
            if key<=len(self.displayClasses):
                self.displayClasses[key].show()
            
                
    def clearLabels (self):
        for i in range(self.transparentButtomXDim*self.transparentButtomYDim):
            #print('self.transparentButtomXDim value is ',self.transparentButtomXDim)
            #print('self.transparentButtomYDim value is ',self.transparentButtomYDim)
            #print('range is ',self.transparentButtomXDim*self.transparentButtomYDim)
            #print('self.classificationList[self.opticalImageIndex][i] value is ',self.classificationList[self.opticalImageIndex][i])
            if self.classificationList[self.opticalImageIndex][i]==0:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==1:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==2:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==3:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==4:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==5:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==6:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==7:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==8:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
            elif self.classificationList[self.opticalImageIndex][i]==9:
                self.transparentButtom[i].setStyleSheet("border: 1px solid black;")
             
        for y in range(50):
            self.displayClasses[y].hide()
         
           
        
            
def runApp():
    app = QApplication(sys.argv)
    win=ImagefocusungApp()
    win.show()
    sys.exit(app.exec_())


runApp()