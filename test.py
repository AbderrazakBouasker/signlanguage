import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")
offset=50
imgsize=400
folder="data/B"
labels=["A","B"]

while True:
    success, img=cap.read()
    imgoutput=img.copy()
    hands,img=detector.findHands(img)
    
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop=img[y-offset : y+h+offset , x-offset : x+w+offset]
        imgcropshape=imgcrop.shape
        imgwhite[0:imgcropshape[0],0:imgcropshape[1]]
        
        aspectratio=h/w
        if aspectratio>1:
            k=imgsize/h
            wcalc=math.ceil(w*k)
            imgresize=cv2.resize(imgcrop,(wcalc,imgsize))
            imgresizeshape=imgresize.shape
            wgap=math.ceil((imgsize-wcalc)/2)
            imgwhite[:,wgap:wcalc+wgap]=imgresize
            prediction,index=classifier.getPrediction(imgwhite)

        else:
            k=imgsize/w
            hcalc=math.ceil(h*k)
            imgresize=cv2.resize(imgcrop,(imgsize,hcalc))
            imgresizeshape=imgresize.shape
            hgap=math.ceil((imgsize-hcalc)/2)
            imgwhite[hgap:hcalc+hgap,:]=imgresize
            prediction,index=classifier.getPrediction(imgwhite)

        cv2.putText(imgoutput,labels[index],(x+10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        # cv2.imshow("ImageCrop",imgcrop)
        # cv2.imshow("ImageWhite",imgwhite)

    cv2.imshow("Image",imgoutput)

    cv2.waitKey(1)
