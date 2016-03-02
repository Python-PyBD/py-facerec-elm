#!/usr/bin/env python

from sklearn.decomposition import PCA
import numpy as np
import cv2 as cv

#shuf -n 10 input.txt > test.txt

def doPCA(imgpath):
    imgdata = cv.imread(imgpath, 0)
    #imgdata = cv.resize(imgdata, (0,0), fx=.5, fy=.5)

    pca = PCA()
    data = pca.fit(imgdata)
    mean = data.mean_
    
    return np.interp(mean, [0, 255], [-15, 15])
    #return np.interp(mean, [0, 255], [0, 10])


data = []
classes = []
#test = []

for people in range(40):
    for index in range(10):
        imgfile = 'Cambridge_FaceDB/s%d/%d.pgm' % (people+1,index+1)
        
        print imgfile
        
        imgdata = doPCA(imgfile)
        
        #if index == 9:
            #test.append(imgdata)
        
        data.append(imgdata)
        a = [0] * 40
        a[people] = 1
        classes.append(a)
        
np.savetxt('input.txt', data, fmt='%.3f')
np.savetxt('output.txt', classes, fmt='%d')
#np.savetxt('test.txt', test, fmt='%.3f')
