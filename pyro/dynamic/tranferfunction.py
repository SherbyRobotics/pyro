#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:17:40 2021

@author: alex
"""


import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt

from pyro.dynamic import ContinuousDynamicSystem
from pyro.dynamic import linearize

# Embed font type in PDF
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

###############################################################################
class TransferFunction( ContinuousDynamicSystem ):
    """Time-invariant transfer function representation of a dynamic system

    Y(s) / U(s) =   [ num ] / [den]

    """
    ############################################
    def __init__(self, num, den):
        
        self.num = num
        self.den = den

        n = den.shape[0] - 1
        
        super().__init__(n, 1, 1)
        
        self.compute_ss()
        self.compute_poleszeros()
        
        # Plot params
        self.figsize = (5,3)
        self.dpi = 300
        self.fontsize = 5
        
    #############################################
    def compute_ss(self):

        A, B, C, D = signal.tf2ss(self.num, self.den)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    #############################################
    def compute_poleszeros(self):
        
        tf = signal.TransferFunction(self.num, self.den)
        
        self.poles = tf.poles
        self.zeros = tf.zeros
        
    #############################################
    def f(self, x, u, t):

        dx = np.dot(self.A, x) + np.dot(self.B, u)

        return dx
    
    #############################################
    def h(self, x, u, t):
        
        y = np.dot(self.C, x) + np.dot(self.D, u)
        
        return y
    
    ############################################
    def bode_plot(self):
        """ Plot frequency response """
        
        tf = signal.TransferFunction(self.num, self.den)
        
        w, mag, phase = tf.bode()
        
        fig , plots = plt.subplots(2, sharex=True, figsize=self.figsize, 
                                  dpi=self.dpi, frameon=True)
        
        plots[0].semilogx(w, mag)
        plots[1].semilogx(w, phase)
        
        plots[0].set_ylabel(self.output_label[0] + ' ' + self.output_units[0] +'\n-------------------\n'+ self.input_label[0] + ' ' + self.input_units[0]
                 , fontsize= self.fontsize )
        plots[1].set_ylabel( 'Phase [rad]', fontsize= self.fontsize )
        plots[1].set_xlabel( 'Freq [rad/sec]', fontsize= self.fontsize )
        
        for i in [0,1]:
            plots[i].grid(True)
            plots[i].tick_params( labelsize = self.fontsize )
        
        fig.canvas.set_window_title('Bode plot of ' + self.name)
        
        plt.show()
        
    ############################################
    def pz_map(self):
        """ Plot poles and zeros """
        
        self.compute_poleszeros()
        
        fig , plot = plt.subplots(1, sharex=True, figsize=self.figsize, 
                                  dpi=self.dpi, frameon=True)
        
        plot.plot( self.poles.real, self.poles.imag, 'x')
        plot.plot( self.zeros.real, self.zeros.imag, 'o')
        plot.grid(True)
        plot.tick_params( labelsize = self.fontsize )
        
        fig.canvas.set_window_title('Poles and zeros of ' + self.name)
        
        plt.show()
        




#################################################################
def ss2tf( ss, u_index , y_index):
    """
    Compute the transfer function of a given input and output of a state-space
    """
    nums, den = signal.ss2tf(ss.A, ss.B, ss.C, ss.D, u_index)
    
    num = nums[y_index]
    
    tf = TransferFunction(num, den)
    
    tf.name = (ss.output_label[y_index] + '/' + ss.input_label[u_index] + 
               ' transfer function of ' + ss.name )

    tf.output_label[0] = ss.output_label[y_index]
    tf.output_units[0] = ss.output_units[y_index]
    tf.input_label[0]  = ss.input_label[u_index]
    tf.input_units[0]  = ss.input_units[u_index]
    
    return tf
        

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    num = np.array([1])
    den = np.array([1,0,1])
    
    #TF = signal.TransferFunction( num, den)
    
    sys = TransferFunction( num , den)
    
    #sys.bode_plot()
    
    from pyro.dynamic import pendulum
    
    non_linear_sys = pendulum.SinglePendulum()
    
    non_linear_sys.lc1     = 1
    non_linear_sys.m1      = 2
    non_linear_sys.I1      = 2
    non_linear_sys.d1      = 2
    non_linear_sys.gravity = 9.81
    
    linearized_sys = linearize( non_linear_sys )
    siso_sys       = ss2tf( linearized_sys, 0, 0)
    
    print('Poles',siso_sys.poles)
    print('num',siso_sys.num)
    print('den',siso_sys.den)
    
    siso_sys.bode_plot()
    
    
    