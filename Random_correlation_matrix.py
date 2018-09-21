# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:30:19 2018

@author: Sam
"""

import math
import numpy as np
import random
import matplotlib.pyplot as plt

def random_correlation_matrix(d):
    P = np.zeros((d,d))
    S = np.identity(d)
    for k in range(0,d-1):
        for i in range(k+1,d):
            P[k,i] = random.random()
            P[k,i] = P[k,i]*2-1
            p = P[k,i]
            for l in range(k-1,0,-1):
                p = p*math.sqrt((1-P[l,i]**2)*(1-P[l,k]**2)) + P[l,i]*P[l,k]
            S[k,i] = p
            S[i,k] = p
    return S

def main():
    d = 30
    C = random_correlation_matrix(d)
    print(C)
    eigvals, eigvecs_T = np.linalg.eig(C)
    assert (np.imag(eigvals) == 0).all(), 'Imaginary eigenvalues!'

    vals = list(np.array(C.ravel()))
    
    plt.figure()
    plt.plot(list(reversed(sorted(eigvals))))
    plt.savefig("Eigenvalues_correlation_matrix.pdf")

    plt.figure()
    plt.hist(vals, range=(-1,1))
    plt.savefig("Correlation_matrix_histogram.pdf")


    plt.figure()
    plt.imshow(C, interpolation=None)
    plt.savefig("Correlation_heatmap.pdf")
    
    plt.show()
   
if __name__ == '__main__':
    main()