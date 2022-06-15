import cv2
import numpy as np

def Impresion(namme,imagen,x,y):
    cv2.namedWindow(namme)
    cv2.moveWindow(namme, x,y)
    cv2.imshow(namme, imagen)
    cv2.waitKey(0)

img1 = cv2.imread('Original.jpg',0)
img2 = cv2.imread('Sonne.jpg',0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
Impresion('Similitudes',img3,50,50)


cap = cv2.VideoCapture('Mancha.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        fgmask = fgbg.apply(frame)

        cv2.imshow('fgmask',frame)

        cv2.imshow('frame',fgmask)

        if cv2.waitKey(30) == ord('s'):
            break
    else: break
    
cap.release()
cv2.destroyAllWindows()
