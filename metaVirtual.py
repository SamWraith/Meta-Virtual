
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, render_template, Response, url_for
import os


def feature1():
    bpos = [deque(maxlen=1024)]
    gpos = [deque(maxlen=1024)]
    rpos = [deque(maxlen=1024)]
    ypos = [deque(maxlen=1024)]
    cirpos = [deque(maxlen=1024)]


    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0
    cir_index = 0


    kernel = np.ones((5,5),np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0


    paintWindow = np.zeros((471,636,3)) + 255

    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence = 0.6)
    mpDraw = mp.solutions.drawing_utils

    


    cap = cv2.VideoCapture(0)
    start = True

    while start:
        # img = cv2.imread('Screenshot (2).png')
        # cv2.imshow('Original Image',img)
    
        # imageLine = img.copy()
        # cv2.resize(imageLine,(0, 0), fx = 0.1, fy = 0.1)
        start, frame = cap.read()

        x, y, c = frame.shape

    
        frame = cv2.flip(frame, 1)
        
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #hsv

        frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
        frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
        frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
        frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
        frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
        frame = cv2.rectangle(frame,  (40,100), (140,70), (0,0,0), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Circle", (49, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


        result = hands.process(framergb)

    
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)

                    landmarks.append([lmx, lmy])



                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0],landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0],landmarks[4][1])
            cv2.circle(frame, center, 3, (0,255,0),-1)
            # cv2.circle(paintWindow,center,3, (0,255,0),-1)
            print(center[1]-thumb[1])
            if (thumb[1]-center[1]<30):
                bpos.append(deque(maxlen=512))
                blue_index += 1
                gpos.append(deque(maxlen=512))
                green_index += 1
                rpos.append(deque(maxlen=512))
                red_index += 1
                ypos.append(deque(maxlen=512))
                yellow_index += 1

            elif center[1] <= 65:

                if 40 <= center[0] <= 140: # Clear Button
                    bpos = [deque(maxlen=512)]
                    gpos = [deque(maxlen=512)]
                    rpos = [deque(maxlen=512)]
                    ypos = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    cir_index = 0
                    paintWindow[0:,:,:] = 255
                elif 160 <= center[0] <= 255:
                        colorIndex = 0 # Blue
                elif 275 <= center[0] <= 370:
                        colorIndex = 1 # Green
                elif 390 <= center[0] <= 485:
                        colorIndex = 2 # Red
                elif 505 <= center[0] <= 600:
                        colorIndex = 3 # Yellow
            elif center[0] <= 70:    #circle
                    if 40 <= center[0] <= 140:
                        cir_index = 1
                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0   
                
            else :
                if colorIndex == 0:
                    bpos[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpos[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpos[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypos[yellow_index].appendleft(center)
            

        else:
            bpos.append(deque(maxlen=512))
            blue_index += 1
            gpos.append(deque(maxlen=512))
            green_index += 1
            rpos.append(deque(maxlen=512))
            red_index += 1
            ypos.append(deque(maxlen=512))
            yellow_index += 1


        points = [bpos, gpos, rpos, ypos]
        
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    if cir_index == 1:
                        
                        cv2.circle(frame, (lmx, lmy), 40, (0,255,255), 2)
                        cv2.circle(paintWindow, (lmx, lmy), 40, (0,255,255), 2)
                        # cv2.circle(imageLine, (lmx, lmy), 40, (0,255,255), 2)
                        cir_index = 0
                        print("Circle")
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    # cv2.line(imageLine, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    

        cv2.imshow("Output", frame) 
        cv2.imshow("Paint", paintWindow)
        # cv2.imshow('Original Image',imageLine)

        if cv2.waitKey(1) == ord('q'):
            # render_template('meta.html')
            break


    cap.release()
    cv2.destroyAllWindows()
    # import statistics
    
    # # initializing list
    # li = [1, 2, 3, 3, 2, 2, 2, 1]
    
    # # using mean() to calculate average of list
    # # elements
    # print ("The average of list values is : ",end="")
    # print (statistics.mean(li))
def feature2():
    width, height = 1280, 720
    gestureThreshold = 300
    folderPath = "Presentation"
    
    # Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    
    # Hand Detector
    detectorHand = HandDetector(detectionCon=0.8, maxHands=1)
    
    # Variables
    imgList = []
    delay = 30
    buttonPressed = False
    counter = 0
    drawMode = False
    imgNumber = 0
    delayCounter = 0
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False
    hs, ws = int(140), int(250)  
    

    pathImages = sorted(os.listdir(folderPath), key=len)
    print(pathImages)
    
    while True:
        
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)
    
        
        hands, img = detectorHand.findHands(img)  # with draw
        
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    
        if hands and buttonPressed is False:  
    
            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]  
            fingers = detectorHand.fingersUp(hand)  
    
            
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
            indexFinger = xVal, yVal
    
            if cy <= gestureThreshold:  
                if fingers == [1, 0, 0, 0, 0]:
                    print("Left")
                    buttonPressed = True
                    if imgNumber > 0:
                        imgNumber -= 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
                if fingers == [0, 0, 0, 0, 1]:
                    print("Right")
                    buttonPressed = True
                    if imgNumber < len(pathImages) - 1:
                        imgNumber += 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
    
            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
    
            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                print(annotationNumber)
                annotations[annotationNumber].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
    
            else:
                annotationStart = False
    
            if fingers == [0, 1, 1, 1, 0]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    
        else:
            annotationStart = False
    
        if buttonPressed:
            counter += 1
            if counter > delay:
                counter = 0
                buttonPressed = False
    
        for i, annotation in enumerate(annotations):
            for j in range(len(annotation)):
                if j != 0:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)
    
        imgSmall = cv2.resize(img, (ws, hs))
        h, w, _ = imgCurrent.shape
        imgCurrent[0:hs, w - ws: w] = imgSmall
    
        cv2.imshow("Slides", imgCurrent)
        cv2.imshow("Image", img)
    
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
     
app = Flask(__name__)

@app.route('/')
def home():
     return render_template("pagee1.html")
@app.route('/signup')
def signup():
     return render_template("signup.html")
@app.route('/meta', methods = ['POST', 'GET'])
def meta():
     return render_template("meta1.html")
@app.route('/fea1')
def fea1():
    Response(feature1())
    return  render_template('fea1.html')
    # return render_template("page1.html")
@app.route('/fea2')
def fea2():
     Response(feature2())
     return render_template('fea1.html')
if __name__ == '__main__':
     app.run(debug = False, host = "0.0.0.0")
    #  feature1() 