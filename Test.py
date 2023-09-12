import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", 'model/labels.txt')
offset = 20
imgsize = 300

# folder = "data/d"
counter = 0
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']

while True:
    suc, img = cap.read()
    imgoutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgcropshape = imgcrop.shape

        # imgwhite[0:imgcropshape[0],0:imgcropshape[1]]=imgcrop

        aspectratio = h / w
        if aspectratio > 1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize
            prediction, index = classifier.getPrediction(imgwhite, draw=False)
            print(prediction, index)

        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize
            prediction, index = classifier.getPrediction(imgwhite, draw=False)

        cv2.rectangle(imgoutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(imgoutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (25, 56, 180), 2)
        cv2.rectangle(imgoutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 255, 255), 4)

        cv2.imshow('imgcrop', imgcrop)
        cv2.imshow('imgwhite', imgwhite)

    cv2.imshow('image', imgoutput)
    key = cv2.waitKey(1)
