import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=50
imgsize=400
folder="data/B"

while True:
    success, img=cap.read()
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
        else:
            k=imgsize/w
            hcalc=math.ceil(h*k)
            imgresize=cv2.resize(imgcrop,(imgsize,hcalc))
            imgresizeshape=imgresize.shape
            hgap=math.ceil((imgsize-hcalc)/2)
            imgwhite[hgap:hcalc+hgap,:]=imgresize

        cv2.imshow("ImageCrop",imgcrop)
        cv2.imshow("ImageWhite",imgwhite)

    cv2.imshow("Image",img)

    key=cv2.waitKey(1)
    if key==ord('s'):
        print("Saving")
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgwhite)