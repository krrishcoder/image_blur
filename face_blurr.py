import cv2 as cv
import numpy as np
from deepface import DeepFace
import json
from retinaface import RetinaFace

import sys




# Python version: 3.11.10 (main, Sep  7 2024, 01:03:31) [Clang 15.0.0 (clang-1500.3.9.4)]
# Version info: sys.version_info(major=3, minor=11, micro=10, releaselevel='fi
# OpenCV version: 4.10.0
# NumPy version: 1.26.4
# DeepFace version: 0.0.93
# RetinaFace version: 0.0.17

# Note: json does not have a version attribute
print(f"JSON version: Not available")


'''
img = cv.imread("./people-1979261_640.jpg")

faces = RetinaFace.detect_faces("./people-1979261_640.jpg", threshold=0.5)



# drawing geomteric shapes
# create a black image
#img = np.zeros((512,512,3), np.uint8)



# draw a diagonal blue line with thikness of 5 px

# cv.line(img, pt1=(0,0), pt2=(511,511), color=(255,0,0),thickness=5)
# cv.rectangle(img,pt1=(10,10),pt2=(100,100),color=(255,0,0),thickness=3)

# cv.circle(img,center=(100,100), radius=45,color=(0,255,0),thickness=5)
# cv.ellipse(img, (256,256),(100,50),0,0,330,255,-1)

# cv.putText(img, 'open cv', (10,500), cv.FONT_HERSHEY_COMPLEX, 4, (255,0,0),2,cv.LINE_AA)


# # mouse callback function whihc is executed when mouse click takes place
# events = [i for i in dir(cv) if 'EVENT' in i]
# print(events)

# # mouse callback function
# def draw_circle(event, x,y , flags, param):
#     if event == cv.EVENT_MOUSEMOVE:
#         cv.circle(img,(x,y),20,(255,0,0), -1)
#         print("clicked")


# cv.namedWindow("image")
# cv.setMouseCallback('image', draw_circle)



for key,value in faces.items():
    # cv.rectangle(img,(value["facial_area"][2], value["facial_area"][3]), (value["facial_area"][0], value["facial_area"][1]) , (255,0,0),2 )

    x1 = value["facial_area"][2]
    y1 =value["facial_area"][3]

    x2 = value["facial_area"][0]
    y2 =value["facial_area"][1]
    ksize = (15,15)  # Kernel size
    print(x1,y1)
    print(x2,y2)
    
    roi = img[y2:y1, x2:x1]
    blurred_region =  cv.blur(roi,(ksize))
    img[y2:y1, x2:x1] =  blurred_region 





while(1):
    cv.imshow("image", img)
    if cv.waitKey(20) & 0xFF ==27:
        break

k = cv.waitKey(0)  # wait for a keystroke in the window

if k == ord("s"):
    cv.imwrite("i am saving image.png", img)


cv.destroyAllWindows()


'''
