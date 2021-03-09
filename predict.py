import numpy as np
from keras.models import load_model
import operator
import cv2
import sys, os
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import time

prediction = ''
prediction_actual = ''
loaded_model = load_model("model.h5")
print("Loaded model from disk")

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate = 0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    res = cv2.bitwise_and(frame, frame, mask = fgmask)
    return res

# Category dictionary
# signals = {0: 'let', 1: 'let_l', 2: 'nolet', 3: 'nolet_l', 4: 'none', 5: 'stroke', 6: 'stroke_l'}
signals = {0: 'let', 1: 'nolet', 2: 'none', 3: 'stroke'}
signals_translated = {
    '': '', 
    'let': 'Let',
    # 'let_l': 'Let',
    'nolet': 'No Let', 
    # 'nolet_l': 'No Let',
    'none': 'None',
    'stroke': 'Stroke',
    # 'stroke_l': 'Stroke'
}

camera = cv2.VideoCapture(0)
camera.set(10, 200)
isBgCaptured = 0

while camera.isOpened():
    ret, frame = camera.read()
    # smoothing filter
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # flip frame horizontally
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(0.5 * frame.shape[1]), 0),
                  (frame.shape[1], int(0.8 * frame.shape[0])), (255, 0, 0), 2)
    # cv2.putText(frame, f"Prediction: {signals_translated[prediction]}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)    
    cv2.putText(frame, f"Prediction: {prediction}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)    
    cv2.imshow('Original', frame)

    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(0.8 * frame.shape[0]),
              int(0.5 * frame.shape[1]):frame.shape[1]]
        
        # do the processing after capturing the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        ret, thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("ROI", thresh)
        image = resize(thresh, (64, 64))
        image = image.reshape(1, 64, 64, 1)

        # generate prediction using trained model
        result = loaded_model.predict(image)
        print(result)

        prediction = {}
        for idx, signal in signals.items():
            prediction[signal] = result[0][idx]

        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
        print(prediction)
        prediction = prediction[0][0]
        # Displaying the predictions
        # cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)    
        cv2.imshow("Original", frame)  

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        isBgCaptured = 1
        # print('Captured Background!')
    elif interrupt & 0xFF == ord('s'):
        isBgCaptured = 0
        # print('Stopping prediction capturing')
    elif interrupt & 0xFF == ord('q'): # esc key
        break

camera.release()
cv2.destroyAllWindows()