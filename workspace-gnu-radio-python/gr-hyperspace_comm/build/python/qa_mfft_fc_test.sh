#!/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=/home/david/hyperspace_waves/workspace-gnu-radio-python/gr-hyperspace_comm/python
export GR_CONF_CONTROLPORT_ON=False
export PATH=/home/david/hyperspace_waves/workspace-gnu-radio-python/gr-hyperspace_comm/build/python:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PYTHONPATH=/home/david/hyperspace_waves/workspace-gnu-radio-python/gr-hyperspace_comm/build/swig:$PYTHONPATH
/usr/bin/python2 /home/david/hyperspace_waves/workspace-gnu-radio-python/gr-hyperspace_comm/python/qa_mfft_fc.py 
