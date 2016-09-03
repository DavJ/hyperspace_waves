'''
Created on Aug 21, 2016

@author: david jaros
site: www.octonion-multiverse.com

(note: for numpy usage see http://www.scipy-lectures.org/intro/numpy/operations.html)
'''

import numpy as np
import math
import cmath



def real_exp(arg):
    return cmath.exp(arg).real
    

def generate_hyperspace_wave(freq=2e6, fsample=25e6, s1=-1/math.sqrt(2), s2=1, N=1024):
    '''
    This function creates invariant (or balanced) hyperspace pulss (waves), which propagate same way in normal space and hyperspace.
    
    Note: for s1>0 the pulse would have infinite energy, therefore s1 should be always equal to -1/sqrt(2). s2 can be either +1 or -1 
    
    :Arguments
    freq ... frequency of wave (pulse)
    fsample ... sampling frequency (Nyquist criteria should be always satisfied fsample > 2* freq)
    s1 ... s1 is dumping coeficient, which should be equal to -1/sqrt(2) for finite energy invariant pulse
    s2 ... frequency (or signum) of wave (pulse), can be e.g. used for modulation
    
    :Returns
    function returns N samples of balanced hyperspace pulse/wave (list of floats). Samples are at time k/fsample
    
    '''
    index_k = np.asarray(range(0, N))
    arg_arr1 = index_k * (2 * cmath.pi * (s1+s2*1j) * (freq/fsample))
    vexp=np.vectorize(real_exp)
    output=vexp(arg_arr1)
    
    return output



