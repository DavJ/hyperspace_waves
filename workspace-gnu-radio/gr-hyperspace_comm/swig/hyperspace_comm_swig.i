/* -*- c++ -*- */

#define HYPERSPACE_COMM_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "hyperspace_comm_swig_doc.i"

%{
#include "hyperspace_comm/mfft.h"
%}


%include "hyperspace_comm/mfft.h"
GR_SWIG_BLOCK_MAGIC2(hyperspace_comm, mfft);
