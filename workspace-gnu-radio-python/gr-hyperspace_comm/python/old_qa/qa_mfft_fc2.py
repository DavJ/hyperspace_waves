#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2016 <+DAVID JAROS @ OCTONION-MULTIVERSE.COM+>.
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

from gnuradio import gr, gr_unittest
from gnuradio import blocks
from mfft_fc import mfft_fc

from scipy.fftpack import fft

class qa_mfft_fc (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_mfft_0 (self):
	"""
        this test tests, that for p=0, block returns same results as regular fft block 

        """
        # set up fg
        #self.tb.run ()
        # check data
        src_data = (1, 2, 3, 4, 5, 6)
        expected_results = fft((3,4)) 
        print "Expected data" + str(expected_results)

        #source - float _f  
        src = blocks.vector_source_f(src_data)

        #p=0, N=2, M=1
	mfft = mfft_fc(0, 2, 1)

        #destination - complex sink _c!
	dst = blocks.vector_sink_c(1)

        #connect everything together
	self.tb.connect(src, mfft)
	self.tb.connect(mfft, dst)

        #run 
        self.tb.run()
	result_data = dst.data()
        print "Result data: " + str(result_data)

        #assert
	#self.assertComplexTuplesAlmostEqual(expected_results, result_data, 2)
        print "TODO ASSERT"

if __name__ == '__main__':
    gr_unittest.run(qa_mfft_fc, "qa_mfft_fc.xml")
