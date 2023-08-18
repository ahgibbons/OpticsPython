# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:49:21 2020

@author: andre
"""

## BraggStackModel

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:14:52 2018

@author: andrew
"""

"""Multi Layer Bragg Model"""

import matplotlib.pyplot as plt
import numpy as np
exp = np.exp
sqrt = np.sqrt
from math import pi as PI
from numpy.linalg import det,matrix_power
from scipy.optimize import curve_fit
import pandas as pd

import Si, Gold

""" Material Params """
ps_params = {'A': 1, 'B': 1.4435, 'C': 20216} # Sellmeier
si_params = {'A': 1, 'B': 10.668, 'C': 90912} # Sellmeier
pc_params = {'A' : 1, 'B': 1.4182, 'C' : 21304} # Sellmeier
bk7_params = {'A' : 1, 'B': 1.03961212, 'C' : 6.00069867e-3 }
n_air = 1
n_si = 3.67

class Structure:
    def __init__(self, layers, nLeft, right_params):
        self.layers = layers
        self.nLeft = nLeft
        self.right_params = right_params
        self.total_thickness = sum([a.thickness for a in layers])
        
    def reflectance(self, wavelengths):
        rs = [gen_multilayer_matrix(w,self.layers,self.nLeft,self.right_params) for w in wavelengths]
        return rs
        
    def __repr__(self):
        return "Num Layers: {:d}, Thickness: {:.03f}".format(len(self.layers),self.total_thickness) 

class SellmeierMedium:
    def __init__(self, params):
        self.params = params
        self.A = params['A']
        self.B = params['B']
        self.C = params['C']     
        
        
    def n(self,w):
        return np.sqrt(self.A + ((self.B * w**2) / (w**2 - self.C)))


def sellmeierEqn(A,B,C,w):
    n = np.sqrt(A + ((B * w**2) / (w**2 - C)))
    return n

def n2param_const(n):
    return {'A':1, 'B': n*n-1, 'C':0}

def cauchyEqn(A,B,C,ws):
    ws2 = ws/1000
    return A + B/np.power(ws2,2) + C/np.power(ws2,4)


class SellmeierLayer:
    def __init__(self, mat_params, thickness, name=None):
        """
        SellmeierLayer(mat_params, thickness, name=None)
        """
        self.A = mat_params['A']
        self.B = mat_params['B']
        self.C = mat_params['C']
        self.thickness = thickness
        self.name = name
        
    def __repr__(self):
        return "<SellmeierLayer: A = {:.03f}, B = {:0.03f}, t = {:003f}>".format(self.A,self.B,self.thickness)
        
    def ri(self,w):
        return np.sqrt(self.A + ((self.B * w**2) / (w**2 - self.C)))
    
    def matrix(self,w):
        """
        matrix(w)
        Generate transmission matrix for layer for wavelength w
        """
        n = self.ri(w)
        
        k = 2*PI*n / w
        theta = k*self.thickness
        
        M = np.array([[np.cos(theta), 1j*np.sin(theta)/n],
                        [1j*n*np.sin(theta), np.cos(theta)]])
        return M
    
class DataFrameLayer:
    def __init__(self, material_dataframe, thickness, name=None):
        self.dataframe = material_dataframe
        self.thickness = thickness
        self.name = name
        
    def matrix(self,w):
        n_real = np.interp(w,self.dataframe['Wavelength(nm)'],self.dataframe['n'])
        n_imag = np.interp(w,self.dataframe['Wavelength(nm)'],self.dataframe['k'])
        n = n_real + 1j*n_imag
        k = 2*PI*n / w
        theta = k*self.thickness
        
        M = np.array([[np.cos(theta), 1j*np.sin(theta)/n],
                        [1j*n*np.sin(theta), np.cos(theta)]])
    
        return M
    
class PorousLayer:
    def __init__(self, mat_params, porosity, thickness, 
                 name = None, npore = n_air):
        self.A = mat_params['A']
        self.B = mat_params['B']
        self.C = mat_params['C']
        self.porosity = porosity
        self.thickness = thickness
        self.name = name
        self.npore = npore
        
    def ri(self,w):
        nh = np.sqrt(self.A + ((self.B * w**2) / (w**2 - self.C)))
        #n = 1 + ((nh -1 ) * (1-self.porosity)) 
        return self.npore*self.porosity + nh*(1-self.porosity)
        

    def matrix(self,w):
        #nh = np.sqrt(self.A + ((self.B * w**2) / (w**2 - self.C)))
        #n = 1 + ((nh -1 ) * (1-self.porosity)) 
        #n = self.npore*self.porosity + nh*(1-self.porosity)
        n = self.ri(w)
        
        
        k = 2*PI*n / w
        theta = k*self.thickness
        
        M = np.array([[np.cos(theta), 1j*np.sin(theta)/n],
                        [1j*n*np.sin(theta), np.cos(theta)]])
        return M

class SimpleLayer:
    """
    SimpleLayer(refractive_index, thickness, name = None)
    Simple layer with constant refractive index.
    """
    def __init__(self, refractive_index, thickness, name = None):
        """
        SimpleLayer(refractive_index, thickness, name = None)
        Simple layer with constant refractive index.
        """
        self.refractive_index = refractive_index
        self.thickness = thickness
        self.name = name

    def __repr__(self):
        return "<SimpleLayer: n = {:.03f},  t = {:.03f}>".format(self.refractive_index, self.thickness)
        
    def matrix(self,w):
        k = 2*PI*self.refractive_index / w
        theta = k*self.thickness
        
        n = self.refractive_index
        M = np.array([[np.cos(theta), 1j*np.sin(theta)/n],
                        [1j*n*np.sin(theta), np.cos(theta)]])
        return M
    

def thin_film(ws,mat_params,thickness,substrate_df):
    layer = [SellmeierLayer(mat_params,thickness)]
    rs = [gen_multilayer_matrix_df_substrate(w,layer,n_air,substrate_df) for
          w in ws]
    return rs

def gen_multilayer_matrix(w,layers,n_Left,right_params):
    BLinv = 0.5*np.array([[1,-1/n_Left],[1,1/n_Left]])
    A = right_params['A']
    B = right_params['B']
    C = right_params['C']
    n_si_sell = np.sqrt(A + ((B * w**2) / (w**2 - C)))
    BR = np.array([[1,1],[-1*n_si_sell,n_si_sell]])
    
    MT = BLinv
    
    for layer in layers:
        m = layer.matrix(w)
        MT = MT @ m
    
    MT = MT @ BR
    
    r = MT[1][0] / MT[0][0]
    
    return (r * r.conj()).real


def fit_porous_film_thickness(ws,ys,mat_params,porosity,df_sub,
                              nlayers,t0,t1):
    def fit_func(xs,t0,t1):
       layer_ps = SellmeierLayer(mat_params,t0)
       layer_pore = PorousLayer(mat_params,porosity,t1)
       layers = [layer_ps,layer_pore]*nlayers + [layer_ps]
       rs = [gen_multilayer_matrix_df_substrate(x,layers,n_air,df_sub) for
             x in xs]
       return rs
   
    a = curve_fit(fit_func,ws,ys,[t0,t1])
    ys_fit = fit_func(ws,a[0][0],a[0][1])
    
    return (a, ys_fit)
       

def gen_multilayer_matrix_df_substrate(w,layers,n_Left,df_right):
    n_Right = np.interp(w,df_right['Wavelength(nm)'],df_right['n'])
    k_Right = np.interp(w,df_right['Wavelength(nm)'],df_right['k'])
    ncomp_Right = n_Right + 1j*k_Right
    
    BLinv = 0.5*np.array([[1,-1/n_Left],[1,1/n_Left]])
    BR = np.array([[1,1],[-1*ncomp_Right,ncomp_Right]])
    
    MT = BLinv
    
    for layer in layers:
        m = layer.matrix(w)
        MT = MT @ m
    
    MT = MT @ BR
    
    r = MT[1][0] / MT[0][0]
    
    return (r * r.conj()).real

def gen_multilayer_matrix_fixed_substrate_reflectance(w,layers,n_Left,n_Right):
    BLinv = 0.5*np.array([[1,-1/n_Left],[1,1/n_Left]])
    BR = np.array([[1,1],[-1*n_Right,n_Right]])
    
    MT = BLinv
    
    for layer in layers:
        m = layer.matrix(w)
        MT = MT @ m
    
    MT = MT @ BR
    
    r = MT[1][0] / MT[0][0]
    
    return (r * r.conj()).real

def gen_multilayer_matrix_fixed_substrate(w,layers,n_Left,n_Right):
    BLinv = 0.5*np.array([[1,-1/n_Left],[1,1/n_Left]])
    BR = np.array([[1,1],[-1*n_Right,n_Right]])
    
    MT = BLinv
    
    for layer in layers:
        m = layer.matrix(w)
        MT = MT @ m
    
    MT = MT @ BR
    
    r = MT[1][0] / MT[0][0]
    t = 1 / MT[0][0]

    R = (r * r.conj()).real
    T = (t * t.conj()).real
    
    return (R,T)

def DBR_fix(ws, layers, n_Left, n_Right):
    rs = np.array([gen_multilayer_matrix_fixed_substrate(w,layers,n_Left,n_Right)[0] for w
           in ws])
    return rs

def DBR_mat(ws, layers, n_Left, pr):
    rs = np.array([gen_multilayer_matrix(w, layers, n_Left, pr) for w
          in ws])
    return rs

def DBR_df(ws, layers, n_Left, df_right):
    rs = np.array([gen_multilayer_matrix_df_substrate(w,layers,n_Left,df_right) for 
                   w in ws])
    return rs       
    
def Rayleigh_Scattering(w,d,n,distance,scatter_angle=0):
    I1 = (1 + (np.cos(scatter_angle))**2)/(2*distance**2)
    I2 = np.power(2*PI/w,4)*(n**2 -1)**2/(n**2+2)**2*(d/2)**6
    return I1*I2

    
def bilayer_fitter(ws,t0,t1,p):
    layer0 = SellmeierLayer(ps_params,t0)
    layer1 = PorousLayer(ps_params,p,t1)
    layers = [layer0,layer1]*4 + [layer0]
    
    rs = [gen_multilayer_matrix_df_substrate(w,layers,n_air,df_si) for w in ws]
    return rs

def multilayer_matrix(n1,n2,L1,L2,nlayers,w):
    B0inv = 0.5*np.array([[1,-1],[1,1]])
    Bsub  = np.array([[1,1],[-1*n_si,n_si]])
    k1 = 2*PI*n1 / w
    k2 = 2*PI*n2 / w
    
    theta1 = k1*L1
    theta2 = k2*L2
    
    
    M1 = np.array([[np.cos(theta1), 1j*np.sin(theta1)/n1],
                    [1j*n1*np.sin(theta1), np.cos(theta1)]])
    
    M2 = np.array([[np.cos(theta2), 1j*np.sin(theta2)/n2],
                    [1j*n2*np.sin(theta2), np.cos(theta2)]])
 
    
    M1M2 = np.matmul(M1,M2)
    
    MN = matrix_power(M1M2,nlayers)
    
    MT = B0inv @ MN @ M1 @ Bsub
    
    r = MT[1][0] / MT[0][0]

    return (r*r.conj()).real    