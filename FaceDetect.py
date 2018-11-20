# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:10:49 2018

@author: Greenu
"""

import cv2 as cv
import sys

def main(inputfile):
    print('generating the facial model')
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    print('Reading Image File')
    img = cv.imread(inputfile[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print('Detecting Facial Image')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print('Drawing rectangle around found faces')
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
   
    print('Display result')    
    cv.imshow('img',img)
    #sys.exit("program end")
    c = cv.waitKey(0)
    if 'q' == chr(c & 255):
        cv.destroyAllWindows()
    #    sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])