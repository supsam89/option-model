# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:16:58 2018

@author: Sam
"""

import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import math

#Creates time step dt.
def time_step(T,n):
    dt = T/n
    return dt

#Gets the share price data.
def fetch_data():
    list1 = pdr.DataReader("GOOG", "yahoo").values
    list2 = pdr.DataReader("IBM", "yahoo").values
    list3 = pdr.DataReader("ORCL", "yahoo").values
    google = list1[:,4]
    IBM = list2[:,4]
    Oracle = list3[:,4]
    return google, IBM, Oracle

#Saves a copy of this data and loads
#instead of having to get online everytime.
import pickle

def save_data(list_of_assets, filename):
    with open(filename, 'wb') as f:
        pickle.dump(len(list_of_assets), f)
        for asset in list_of_assets:
            pickle.dump(asset, f)
    
def load_data(filename):
    with open(filename, 'rb') as f:
        n = pickle.load(f)
        list_of_assets = []
        for k in range(n):
            asset = pickle.load(f)
            list_of_assets.append(asset)
    return list_of_assets

#Calculates the correlation matrix
#between inputs.
def calculate_correlation(a,b,c):
    d = np.corrcoef([a,b,c])
    return d

#Cholesky decomposes a correlation matrix.
def cholesky_decomposition(a):
    L = np.linalg.cholesky(a)
    return L

#Creates the correlated random variables.
def correlated_random_variable(n,L):
    r = np.random.randn(3,n)
    z = np.dot(L,r)
    return z

#Creates tge Wiener process from the correlated
#random variables.
def correlated_wiener_n(T, n, L):
    dt = time_step(T,n)
    z = correlated_random_variable(n,L)
    dw = dt**0.5*z
    dw[:,0] = 0
    y = np.cumsum(dw,axis=1)
    x = y.T
    return x

#Creates the asset walks from the correlated
#Wiener process, returned as a matrix.
def correlated_assets_3(S_0, sigma, mu, T, n, L):
    t = np.linspace(0,T,n)
    S = np.zeros((3,n))
    S[:,0] = S_0
    a = sigma*correlated_wiener_n(T, n, L)
    e = (mu-1/2*sigma**2)
    g = e*t.reshape(n,1)
    S = S_0*np.exp(g+a)
    return t, S
    
#Returns the Monte Carlo value given a payoff.
def monte_carlo_3(S_0, r, mu, sigma, T, n, m, path_payoff, L):
    total = 0.0
    for j in range(m):
        t, S = correlated_assets_3(S_0, sigma, mu, T, n, L)
        v_T = path_payoff(S)
        total = total + v_T
    a = 1/m*total
    b = math.exp(-r*T)*a
    return b

#Returns the final price given an array.
def final_price(s):
    return s[-1]

#The payoff for a three asset spread option.
def asset_spread_option_payoff_3(S,E):
    S1 = S[:,0]
    S2 = S[:,1]
    S3 = S[:,2]
    s1_T = final_price(S1)
    s2_T = final_price(S2)
    s3_T = final_price(S3)
    return max((s1_T-s2_T-s3_T-E),0)

#This is a factory function allowing only the 
#exercise price to be entered.
def make_asset_spread_option_payoff_3(E):
    def c(S):
        return asset_spread_option_payoff_3(S,E)
    return c

def main():
    T, n = 1, 100000
    S_0, sigma, r, mu = np.array([200,50,90]), np.array([0.2,0.2,0.2]), 0.05, np.array([0.12,0.12,0.12])
    #data = fetch_data()
    #save_data(data, 'assets.pkl')
    data = load_data('assets.pkl')
    a, b, c = data
#    print(data)
    d = calculate_correlation(a,b,c)
    L = cholesky_decomposition(d)
    print(calculate_correlation(a,b,c))
#    print(L)
#    e = correlated_random_variable(n, L)
#    p, o, q = e[0,:], e[1,:], e[2,:]
#    print(calculate_correlation(p,o,q))
    #f = correlated_wiener_n(T,n,L)
    #x, y, z = f[0,:], f[1,:], f[2,:]
#    print(x)
    #print(calculate_correlation(x,y,z))
#    print(cholesky_decomposition(y))
    t, S = correlated_assets_3(S_0, sigma, r, T, n, L)
#    print(S)
    plt.figure()
    plt.plot(t,S)
    plt.savefig("3_correlated_assets.pdf")
    plt.show()
    E, m = 40, 10
    print(monte_carlo_3(S_0, r, mu, sigma, T, n, m, make_asset_spread_option_payoff_3(E), L))
    
    
if __name__ == '__main__':
    main() 
    
    
    