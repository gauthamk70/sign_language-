import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300

folder = "data/#space"
counter = 0

while True:
    suc, img = cap.read()
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
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize

        cv2.imshow('imgcrop', imgcrop)
        cv2.imshow('imgwhite', imgwhite)

    cv2.imshow('image', img)
    key = cv2.waitKey(1)  # & 0XFF==ord('q')
    if key == ord('s'):
        counter += 1
        timestamp=time.time()
        cv2.imwrite(f'{folder}/Image_{timestamp:2f}.jpg', imgwhite)
        print(counter)
# cv2.destroyAllWindows()