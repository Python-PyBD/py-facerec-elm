#!/usr/bin/env python

import numpy as np
import time
import random
import sys
import cv2 as cv
from sklearn.decomposition import PCA
from hpelm import ELM

def doPCA(imgpath):
    imgdata = cv.imread(imgpath, 0)

    pca = PCA(n_components=92)
    data = pca.fit(imgdata)
    mean = data.mean_
    
    return np.interp(mean, [0, 255], [-15, 15])

# RANDOM --------------------------------------------------
img_test = sys.argv[1:]

test_data = []
for img_file in img_test:
    img_data = doPCA(img_file)
    test_data.append(img_data)

np.savetxt('test.txt', test_data, fmt='%.3f')
testp = np.loadtxt("test.txt")

# TRAINING AND CLASS DATA ---------------------------------
inp = np.loadtxt("input.txt")
outp = np.loadtxt("output.txt")

# BUILD ELM -----------------------------------------------
neuron = 400
elm = ELM(92, 40)
elm.add_neurons(neuron, "sigm")

# TRAIN ---------------------------------------------------
t0 = time.clock()
elm.train(inp, outp, "c")
t1 = time.clock()
tr = t1-t0

# PREDICT -------------------------------------------------
t0 = time.clock()
pred = elm.predict(testp)
t1 = time.clock()
te = t1-t0

# RESULT --------------------------------------------------
for p in pred:
    i = 0
    for v in p:
        i += 1
        if 1.0 - abs(v) < 0.00001:
            print "people id:", i
            break

print "training took", tr*1000000, "ns"
print "testing took", te*1000000, "ns"
