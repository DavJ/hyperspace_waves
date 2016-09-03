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
            in_sig=[numpy32.float],
            out_sig=[numpy.complex32])
        self.p = p
        self.N = N
        self.M = M

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            #ninput_items_required[i] = noutput_items
	    ninput_items_required[i] = self.N
             

    def general_work(self, input_items, output_items):
        output_items[0][:] = mfft(input_items[0], self.p)
        #consume(0, len(input_items[0]))
        #self.consume_each(len(input_items[0]))
        self.consume_each(self.M)

        #return len(output_items[0])
        return self.N
