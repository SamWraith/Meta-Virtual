
import cv2
import numpy as np
import mediapipe as mp
from collections import deque



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
    img = cv2.imread('Screenshot (2).png')
    cv2.imshow('Original Image',img)
    # cv2.waitKey(0)
    # if img is None:
    #     print('Could not read image')
    imageLine = img.copy()
    cv2.resize(imageLine,(0, 0), fx = 0.1, fy = 0.1)
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
        elif center[0] <= 70:
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
                    
                    # cv2.circle(frame, (lmx, lmy), 40, (0,255,255), 2)
                    # cv2.circle(paintWindow, (lmx, lmy), 40, (0,255,255), 2)
                    # cv2.circle(imageLine, (lmx, lmy), 40, (0,255,255), 2)
                    # cir_index = 0
                    print("Circle")
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(imageLine, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)
    cv2.imshow('Original Image',imageLine)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
