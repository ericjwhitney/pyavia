"""
Functions relating to data manipulation, interpolation, filtering, et cetera.
"""

from .filter import (J211_2pole, J211_4pole, MovingAverage, pinv_simple,
                     savgol_variable)
from .function_1d import (Function1D, FitXY1D, Line1D, PCHIP1D,
                          SmoothPoly1D)
from .interpolate import (linear_int_ext, smooth_array2d, smooth_multi,
                          subd_num_list, Interp2Level)
