# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:57:17 2018

@author: Sam
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt

#Creates time step dt.
def time_step(T,n):
    dt = T/n
    return dt

#Creates the Wiener process walk.
def wiener_process(w0,T,n):
    dt = time_step(T,n)
    t = np.zeros(n+1,dtype=float)
    w = np.zeros(n+1,dtype=float)
    t[0] = 0.0
    w[0] = w0
    for k in range(0,n):
        z_t = random.normalvariate(0,1)
        dW = math.sqrt(dt)*z_t
        t[k+1] = t[k] + dt
        w[k+1] = w[k] + dW
    return t,w

#Function to label graph and save.
def labels(xlabel , ylabel, title, filename=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename:
        plt.savefig(filename)

def main():
    plt.figure()
    w0 = 0
    T = 2
    n = 100
    x,y = wiener_process(w0,T,n)
    plt.plot(x,y)
    labels("$t$", "$W_t$", "One realisation of a Wiener process",
           "wiener_process.pdf")
    plt.show()
    
if __name__ == '__main__':
    main()