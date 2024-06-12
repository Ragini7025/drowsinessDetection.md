```python
import cv2
import os
import numpy as np
import time
import dlib
from math import hypot
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 import cv2
          2 import os
          3 import numpy as np
    

    ModuleNotFoundError: No module named 'cv2'



```python
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
```


```python
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
(W, H) = (None, None)
yawns = 0
yawn_status = False 

def e_o_r(pts,facial_landmarks):
    
    l_p = (facial_landmarks.part(pts[0]).x, facial_landmarks.part(pts[0]).y)
    r_p = (facial_landmarks.part(pts[3]).x, facial_landmarks.part(pts[3]).y)
    c_t = midpoint(facial_landmarks.part(pts[1]), facial_landmarks.part(pts[2]))
    c_b = midpoint(facial_landmarks.part(pts[5]), facial_landmarks.part(pts[4]))
    hor_line = cv2.line(frame, l_p, r_p, (0, 255, 0), 2)
    ver_line = cv2.line(frame, c_t, c_b, (0, 255, 0), 2)
    hor_line_lenght = hypot((l_p[0] - r_p[0]), (l_p[1] - r_p[1]))
    ver_line_lenght = hypot((c_t[0] - c_b[0]), (c_t[1] - c_b[1]))
    
    r = hor_line_lenght / ver_line_lenght
    
    return r

def m_o_r(pts,facial_landmarks):
    
    l_p = (facial_landmarks.part(pts[0]).x, facial_landmarks.part(pts[0]).y)
    r_p = (facial_landmarks.part(pts[1]).x, facial_landmarks.part(pts[1]).y)
    c_t = (facial_landmarks.part(pts[2]).x, facial_landmarks.part(pts[2]).y)
    c_b = (facial_landmarks.part(pts[3]).x, facial_landmarks.part(pts[3]).y)
    hor_line = cv2.line(frame, l_p, r_p, (0, 255, 0), 2)
    ver_line = cv2.line(frame, c_t, c_b, (0, 255, 0), 2)
    hor_line_lenght = hypot((l_p[0] - r_p[0]), (l_p[1] - r_p[1]))
    ver_line_lenght = hypot((c_t[0] - c_b[0]), (c_t[1] - c_b[1]))
    if ver_line_lenght!=0:
        r = hor_line_lenght / ver_line_lenght
    else: 
        r=0
    return ver_line_lenght

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
    left_eye_ratio = e_o_r([36, 37, 38, 39, 40, 41], landmarks)
    right_eye_ratio = e_o_r([42, 43, 44, 45, 46, 47], landmarks)
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
    #print(blinking_ratio)
    #cv2.putText(frame, str(blinking_ratio), (150,150),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    if blinking_ratio>4.6:
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) 
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>7):
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
        
    lip_distance=m_o_r([48,54,51,57], landmarks)
    prev_yawn_status = yawn_status  
    #cv2.putText(frame, str(lip_distance), (150,150),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    if lip_distance > 50:
        yawn_status = True 
        cv2.putText(frame, "Yawning Allert", (50,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        output_text = " Yawn Count: " + str(yawns + 1)
        cv2.putText(frame, output_text, (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

```


```python

```
