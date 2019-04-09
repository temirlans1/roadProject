import cv2
from moviepy.editor import VideoFileClip
import Main
import requests
from utils.color_recognition_module import color_recognition_api
import numpy as np
import json
from PIL import Image
import glob
import threading
import queue 

with open('./config.json') as f:
    config = json.load(f)

ip = str(config["ip"])

frames = queue.Queue()
isProcessOn = False
plates = {}
previous = ""

def vehicleDetect():
    count = 0
    cap = cv2.VideoCapture("3stage1.avi")
    n = 0

    w = cap.get(3)
    h = cap.get(4)
    frameArea = h * w
    areaTH = frameArea / 400

    #Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

    #Kernals
    kernalOp = np.ones((3,3), np.uint8)
    kernalOp2 = np.ones((5,5), np.uint8)
    kernalCl = np.ones((11,11), np.int)
    
    crop_vehicle = None
   
    while cap.isOpened():

        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)

        #Binarization
        ret,imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)

        #OPening i.e First Erode the dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernalOp)

        #Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernalCl)

        #Find Contours
        _, countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in countours0:
            area = cv2.contourArea(cnt)
            #print(area)
            #print(cnt)
            
            if area > areaTH + 10000:
                ####Tracking######
                m = cv2.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)
                crop_vehicle = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plateDetect(frame)
                """if frames.qsize()<10:
                    frames.put(crop_vehicle)
                print(frames.qsize())
                if not isProcessOn:
                    el = frames.get()
                    cv2.imshow('new', el)
                    thr = threading.Thread(target=plateDetect, args=(el,))
                    thr.start()"""
                count += 1           
        cv2.imshow('Video', frame)
            #Main.main(frame2)
        if cv2.waitKey(1)&0xff==ord('q'):
            break
        n += 1
    print("Finished.")
    cap.release()

def plateDetect(frame):
    global isProcessOn
    isProcessOn = True
    global previous
    global plates

    result, plateCoordinates = Main.main(frame)
    #cv2.imwrite("frames/new.jpg", frame)
    #cv2.imshow('new', frame)
    if result != "":
        print("License Plate: " + str(result) + "\n")
        
        color = color_recognition_api.color_recognition(frame[int(plateCoordinates[2][1] - 150) : int(plateCoordinates[0][1] - 100), int(plateCoordinates[1][0]) : int(plateCoordinates[3][0])])
        #print("Color: " + str(color) + "\n")
        #print("-----------------------------------------------------------------\n")
        if result not in plates:
            plates[result] = 1
        else :
            plates[result] += 1
        if plates[result] > 1:
            print("License Plate Approved: " + str(result) + "\n")
            print("Send...\n")
            r = requests.post("https://fathomless-plains-27484.herokuapp.com/api/v1/s3M5aCMtypyas8fs1VPHhw/passage", 
                                data = {'car_num': result, 'color': color, 'camera_id': 1})
            print(r.status_code, r.reason)
            plates[result] = -10
            #cv2.imshow("Warped", frame[int(plateCoordinates[2][1] - 200) : int(plateCoordinates[0][1] - 100), int(plateCoordinates[1][0]) : int(plateCoordinates[3][0])])
    #print("Round!")    
    isProcessOn = False

if __name__ == "__main__":
    
    vehicleDetect()   
    #plateDetect() 
    