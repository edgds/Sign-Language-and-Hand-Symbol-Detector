import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0) #turn on webcam
detector = HandDetector(maxHands=1) #detecting one hand

offset = 20
imgSize = 300

folder = "Data/Peace"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  #find hand
    if hands:
        hand = hands[0]  #only have one hand
        x, y, w, h, = hand['bbox'] #get bounnding box information

        #create image matrix:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) *255

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  #(returns as matrix where first param is height and second is width)

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize # put imgCrop matrix inside imgWhite matrix

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # put imgCrop matrix inside imgWhite matrix

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"): #if s key is pressed
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) #save images
        print(counter) #keep track of how many images saved