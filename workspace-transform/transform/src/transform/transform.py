'''
Created on Jul 23, 2016

@author: david jaros
site: www.octonion-multiverse.com

(note: for numpy usage see http://www.scipy-lectures.org/intro/numpy/operations.html)
'''

import numpy as np
import cmath
#from gnuradio import fft



def mfft(input, p=0):
    '''
      mfft
      calculates modified fft using modified Cooley-Tookey algorithm (recursive implementation)
      (note: algorithm is a bit slower than original Cooley-Tookey,  because generally for p<>0: E(k+N/2)<>E(k); O(k+N/2)<>O(k),
             therefor periodicity could not be exploited and additional multiplication is needed
      
      Arguments
    '''
    N = len(input)
    even = np.asarray(input[::2])
    odd =  np.asarray(input[1::2])
    #np.zeros((2,2),dtype=np.complex_)
    index_k = np.asarray(range(0, N/2))
    arg_arr1 = -index_k * (2 * cmath.pi * (p+1j) / N)
    vexp=np.vectorize(cmath.exp)

    multiplicator1 = vexp(arg_arr1)
    multiplicator2 = -cmath.exp(-p*cmath.pi)*vexp(arg_arr1)
    
    if N>2:
        emfft = mfft(even.tolist(), p)
        omfft = mfft(odd.tolist(), p)
        lower_result = emfft + omfft*multiplicator1
        upper_result = emfft + omfft*multiplicator2
    elif N==2:
        lower_result = even + odd*multiplicator1
        upper_result = even + odd*multiplicator2
    else: 
        raise
    
    return lower_result.tolist() + upper_result.tolist()
       
a=[1,2,3,4]   
print mfft(a)
#results for p=0 can be checked here http://calculator.vhex.net/post/calculator-result/1d-discrete-fourier-transform
#print gnuradio.fft