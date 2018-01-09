#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2016 <+DAVID JAROS@OCTONION-MULTIVERSE.COM+>.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import numpy
from gnuradio import gr
from hyperspace_comm.transform.transform import mfft

class mfft_fc(gr.basic_block):
    """
    docstring for block mfft_fc

    this block performs modified transform

    :Arguments

    p .... dumping parameter for mmft (use values +1/-1 for invariant pulses analysis, 0 for regular (slower) FFT) 
    N .... number of input samples required for analysis 
    M .... scrolling parameter tells, by how many samples we shift after analysis

    :Returns
 
    list of complex values of length N, representing the results of the transform 
    
    """
    def __init__(self, p, N, M):
        gr.basic_block.__init__(self,
            name="mfft_fc",
            in_sig=[numpy.float32],
            out_sig=[(numpy.complex64)])

        self.p = p
        self.N = N
        self.M = M
        self.set_relative_rate(N)

    def forecast(self, noutput_items, ninput_items_required):
        #print "fcast noutput_items: " + str(noutput_items)

        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            #ninput_items_required[i] = max(noutput_items, self.N)
            ninput_items_required[i] = max(self.N + (noutput_items/self.N - 1), self.N)

            #print "ninput_items_required: " + str(ninput_items_required[i]) 

    def general_work(self, input_items, output_items):
        print "Input_items" + str(input_items)
        #print "XXLen output items: " + str((len(output_items[0])))
        noutput_items=len(output_items[0])
        ninput_items=len(input_items[0])


        number_of_iterations = noutput_items/self.N
        print "Number of iterations: " + str(number_of_iterations) 
        print  "Number of output items: " + str(noutput_items)

        for i in range(0, noutput_items/self.N):
            #sliced_input_items = input_items[0][ninput_items-self.N-i*self.M : ninput_items-i*self.M]
            sliced_input_items = input_items[0][ i*self.M : i*self.M + self.N]
            print "Sliced input items: " + str(sliced_input_items)

            if i==0:
                 out = mfft(sliced_input_items, self.p)
            else:
                 out = numpy.concatenate((out, mfft(sliced_input_items, self.p)), axis=0)

            self.consume_each(self.M)
       
        if number_of_iterations>0:
           output_items[0][:] = out 
           print "Out: " + str(out)
           return len(out)
        else: return(0)
     

        #return len(output_items[0])
