'''
Created on Aug 27, 2016

@author: david jaros
'''


#tutorial http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
import generator.generate as G
import transform.transform as T

#test generator
plt.plot(G.generate_hyperspace_wave())
plt.ylabel('hyperspace wave')
plt.show()


#test mfft
a=[1,2,3,4]   
print T.mfft(a)
#results for p=0 can be checked here http://calculator.vhex.net/post/calculator-result/1d-discrete-fourier-transform
#print gnuradio.fft
