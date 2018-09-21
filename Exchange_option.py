# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:02:34 2018

@author: Sam
"""

import math
import random
import numpy as np

#Creates time step dt.
def time_step(T,n):
    dt = T/n
    return dt

#Returns the final price given an array.
def final_price(s):
    return s[-1]

#Creates a 2x2 Cholesky decomposition given
#the correlation coefficient (rho).
def cholesky_correlation_matrix_2(rho):
    a = np.zeros((2,2),dtype=float)
    j, k =np.shape(a)
    for i in range(j):
        for l in range(k):
            a[0,0]=1
            a[i,l]=rho
            a[0,l]=0
            a[i,i]=np.sqrt(1-rho**2)
    return a

#The payoff of an exchange option.
def exchange_path_payoff(s1,s2):
    s1_T = final_price(s1)
    s2_T = final_price(s2)
    return max(s1_T-s2_T,0)

#Creates the correlated asset walks.
def risk_neutral_walk(s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho, T, n):
    dt = time_step(T,n)
    t = np.zeros(n+1,dtype=float)
    s1 = np.zeros(n+1,dtype=float)
    s2 = np.zeros(n+1,dtype=float)
    g = cholesky_correlation_matrix_2(rho)
    t[0] = 0.0
    s1[0] = s1_0
    s2[0] = s2_0
    for k in range(0,n):
        a = random.normalvariate(0,1)
        b = random.normalvariate(0,1)
        z_1, z_2 = np.dot(g,[a,b])
        dW1 = math.sqrt(dt)*z_1
        t[k+1] = t[k] + dt
        s1[k+1] = s1[k]*math.exp((r-q_1-1/2*sigma_1**2)*dt+sigma_1*dW1)
        dW2 = math.sqrt(dt)*z_2
        s2[k+1] = s2[k]*math.exp((r-q_2-1/2*sigma_2**2)*dt+sigma_2*dW2)
    return t,s1,s2
    
#Returns the Monte Carlo value with the exchange 
#option payoff.
def monte_carlo(s1_0, s2_0, r,q_1,q_2,sigma_1, sigma_2, rho, T,n,m):
    total = 0.0
    for j in range(m):
        t,s1,s2 = risk_neutral_walk(s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho,T,n)
        v_T = exchange_path_payoff(s1, s2)
        total = total + v_T
    a = 1/m*total
    b = math.exp(-r*T)*a
    return b

#Function that repeats the monte carlo process
#to increase accuracy.
def monte_carlo_simulations(s1_0, s2_0, r,q_1,q_2,sigma_1, sigma_2, rho, T,n,m,l):
    total = 0.0
    for j in range(l):
        m_T = monte_carlo(s1_0, s2_0, r,q_1,q_2,sigma_1, sigma_2, rho, T,n,m)
        total = total + m_T
    a = 1/l*total
    return a

#Defines the probability density function.
def n(x):
    return math.exp(-0.5*x**2)/math.sqrt(2*math.pi)

#Defines the cumulative density function.
def N(d):
    return 0.5*(1 + math.erf(d/math.sqrt(2)))

#Defines the exchange option price formula from derivation.
def exchange_option_price(s1_0, s2_0, r,q_1,q_2,sigma_1,sigma_2, rho, T):
    sigma_new = math.sqrt(sigma_1**2-2*rho*sigma_1*sigma_2+sigma_2**2)
    d1 = (math.log(s1_0/s2_0)+(q_2-q_1+1/2*sigma_new**2)*T)/(sigma_new*math.sqrt(T))
    d2 = d1 - sigma_new*math.sqrt(T)
    V = s1_0*math.exp((-q_1)*T)*N(d1)-s2_0*math.exp((-q_2)*T)*N(d2)
    return V

def main():
    s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho, T, n, m, j = 50, 40, 0.05, 0, 0, 0.2, 0.2, 0.5, 1, 100, 100, 10
    sigma_new = math.sqrt(sigma_1**2-2*rho*sigma_1*sigma_2+sigma_2**2)
    d1 = (math.log(s1_0/s2_0)+(q_2-q_1+1/2*sigma_new**2)*(T))/(sigma_new*math.sqrt(T))
    d2 = d1 - sigma_new*math.sqrt(T)
    t, s1, s2 = risk_neutral_walk(s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho, T, n)
    a = np.corrcoef(s1,s2)
    print(a)
    print(monte_carlo_simulations(s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho,T,n,m,j))
    print(exchange_option_price(s1_0, s2_0, r, q_1, q_2, sigma_1, sigma_2, rho, T))
    
if __name__ == '__main__':
    main()