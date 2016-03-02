#!/usr/bin/env python

import numpy as np
import time
import random
import sys
from hpelm import ELM

inp = np.loadtxt("input.txt")
outp = np.loadtxt("output.txt")

for neuron in range(10, 1000, 10):
    elm = ELM(92, 40)
    elm.add_neurons(neuron, "sigm")

    t0 = time.clock()
    elm.train(inp, outp, "c")
    t1 = time.clock()
    t = t1-t0

    pred = elm.predict(inp)
    acc = float(np.sum(outp.argmax(1) == pred.argmax(1))) / outp.shape[0]
    print "neuron=%d error=%.1f%% time=%dns" % (neuron, 100-acc*100, t*1000000)
    
    if int(acc) == 1:
        break
