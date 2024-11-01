"""
Numeric (:mod:`pyavia.numeric`)
===============================

.. currentmodule:: pyavia.numeric

Core numeric functions used throughout PyAvia.

.. autosummary::
    :toctree:

    solve
    bezier
    filter
    function_1d
    interpolate
    lines
    math_ext
    operations_1d
    polynomial

"""
from .bezier import bernstein_poly, bezier, bezier_deriv
from .filter import (J211_2pole, J211_4pole, MovingAverage, pinv_simple,
                     savgol_variable)
from .function_1d import (Function1D, FitXY1D, Line1D, PCHIP1D,
                          SmoothPoly1D)
from .interpolate import (linear_int_ext, smooth_array2d, smooth_multi,
                          subd_num_list, Interp2Level)
from .lines import line_pt
from .math_ext import (chain_mult, equal_nan, is_number_seq,
                       kind_arctan2, kind_div, min_max, monotonic,
                       strict_decrease, strict_increase, vectorise,
                       sclvec_asarray, check_sclarray, return_sclarray,
                       within_range)
from .operations_1d import sine_spacing, subdivide_series
from .polynomial import (divided_difference, newton_poly_coeff, newton_poly)

