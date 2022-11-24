import cv2
from cvzone.HandTrackingModule import HandDetector #detects hand (left vs right) and draws a skeleton
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0) #turn on webcam
detector = HandDetector(maxHands=1) #detecting one hand
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") #model and the model class names
offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

labels = ["A", "B", "C", "Peace", "Love"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)



        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # put imgCrop matrix inside imgWhite matrix
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        #print(prediction, index)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
