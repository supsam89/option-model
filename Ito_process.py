# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:05:06 2018

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

#Creates the asset walk.
def risk_neutral_walk(s0,r,sigma,T,n):
    dt = time_step(T,n)
    t = np.zeros(n+1,dtype=float)
    s = np.zeros(n+1,dtype=float)
    t[0] = 0.0
    s[0] = s0
    for k in range(0,n):
        z_t = random.normalvariate(0,1)
        dW = math.sqrt(dt)*z_t
        t[k+1] = t[k] + dt
        s[k+1] = s[k]*math.exp((r-1/2*sigma**2)*dt+sigma*dW)
    return t,s

#Function to label graph and save.
def labels(xlabel , ylabel, title, filename=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename:
        plt.savefig(filename)

#Function that plots multiple walks
#on the same graph.
def plot_walks(s0,r,sigma,T,n,m):
    plt.figure()
    for k in range(0,m):
        t,s = risk_neutral_walk(s0, r, sigma, T, n)
        plt.plot(t,s)
        labels("$t$", "$S_t$", "Multiple Ito processes",
           "Ito_process.pdf")
        
def main():
    s0, r, sigma, T, n, m = 100, 0.05, 0.2, 1.0, 100, 6
    a = plot_walks(s0,r,sigma,T,n,m)
    print(a)
    
if __name__ == '__main__':
    main()