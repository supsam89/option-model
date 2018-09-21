# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:10:05 2018

@author: Sam
"""

import random
import numpy as np
import math

#Creates time step dt.
def time_step(T,n):
    dt = T/n
    return dt

#Returns the final price given an array.
def final_price(s):
    return s[-1]

#The payoff of a call option.
def call_path_payoff(s,E):
    s_T = final_price(s)
    return max(s_T-E,0)

#This is a factory function allowing only the 
#exercise price to be entered.
def make_call_path_payoff(E):
    def c(s):
        return call_path_payoff(s,E)
    return c

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

#Returns the Monte Carlo value given a payoff.
def monte_carlo(s0,r,sigma,T,n,m,path_payoff):
    total = 0.0
    for j in range(m):
        t,s = risk_neutral_walk(s0,r,sigma,T,n)
        v_T = path_payoff(s)
        total = total + v_T
    a = 1/m*total
    b = math.exp(-r*T)*a
    return b

#Function that repeats the monte carlo process
#to increase accuracy.
def monte_carlo_simulations(s0, r, sigma, T, n, m, path_payoff, l):
    total = 0.0
    for j in range(l):
        m_T = monte_carlo(s0,r,sigma,T,n,m,path_payoff)
        total = total + m_T
    a = 1/l*total
    return a

#Defines the probability density function.
def n(x):
    return math.exp(-0.5*x**2)/math.sqrt(2*math.pi)

#Defines the cumulative density function.
def N(d):
    return 0.5*(1 + math.erf(d/math.sqrt(2)))

#Defines the call option price formula from derivation.
def call_option_price(s_0, E, r, q, sigma, T):
    d1 = (math.log(s_0/E)+(r-q+1/2*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    V = s_0*math.exp((-q)*T)*N(d1)-E*math.exp((-r)*T)*N(d2)
    return V
    
def main():
    s0 ,r, sigma, T, n, m, q, E, l = 100, 0.05, 0.2, 1, 100, 100, 0, 90, 10
    print(call_option_price(s0, E, r, q, sigma, T))
    print(monte_carlo_simulations(s0, r, sigma, T, n, m, make_call_path_payoff(E), l))
    
if __name__ == '__main__':
    main() 