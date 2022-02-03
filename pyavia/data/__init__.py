"""
Functions relating to data manipulation, interpolation, filtering, et cetera.
"""

from .filter import J211_2pole, J211_4pole, MovingAverage
from .interpolate import (linear_int_ext, smooth_array2d, smooth_multi,
                          subd_num_list, Interp2Level)
