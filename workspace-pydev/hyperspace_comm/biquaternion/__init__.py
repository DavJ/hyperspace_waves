"""
Biquaternion module for hyperspace wave theory.

This module provides biquaternion arithmetic and representations
for hyperspace waves according to the Unified Biquaternion Theory (UBT).

Author: David Jaros
Site: www.octonion-multiverse.com
UBT: https://github.com/DavJ/unified-biquaternion-theory
"""

from .biquaternion import Biquaternion
from .wave_bq import HyperspaceWaveBQ
from .theta_functions import biquaternion_theta

__all__ = ['Biquaternion', 'HyperspaceWaveBQ', 'biquaternion_theta']
