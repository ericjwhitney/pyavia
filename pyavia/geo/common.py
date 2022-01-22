# Last updated: 16 March 2021 by Eric J. Whitney.

from __future__ import annotations
import numpy as np


def line_pt(a, b, p, scale=None):
    """Find the coordinates of a point `p` anywhere along the line `a` → `b`
    where at least one component of `p` is supplied (remaining can be None).
    Each axis may be optionally scaled.  There is no limitation that `p` is
    in the interval [`a`, `b`], so this function can also be used for
    extrapolation as required.

    Parameters
    ----------
    a, b : list_like
        Two distinct points on a line i.e. :math:`[x_1, ..., x_n]`
    p : list_like
        Required point on the line with at least a single known component,
        i.e. :math:`(..., None, p_i, None, ...)`.  If more than one value is
        supplied, the first is used.
    scale : list
        If supplied, a list corresponding to each axis [opt_1, ..., opt_n],
        where each axis can use the following options:

            - None: No scaling performed.
            - 'log': This axis is linear on a log scale.  In practice
                log(x) is performed on this axis prior to doing the
                interpolation / extrapolation, then exp(x) is done prior
                to returning.
    Returns
    -------
    list :
        Required point on line :math:`[q_1, ..., q_n]` where :math:`q_i = p_i`
        from above.
    """
    if scale is None:
        scale = [None] * len(a)
    if not len(a) == len(b) == len(p) == len(scale):
        raise ValueError("a, b, p, [scale] must be the same length.")

    # Scale axes.
    scl_funs, rev_funs = [], []
    for scl_str in scale:
        if scl_str is None:
            scl_funs.append(lambda x: x)
            rev_funs.append(lambda x: x)
        elif scl_str == 'log':
            scl_funs.append(lambda x: np.log(x))
            rev_funs.append(lambda x: np.exp(x))
        else:
            raise ValueError(f"Unknown scale type: {scl_str}.")

    a_scl = [scl_i(a_i) for a_i, scl_i in zip(a, scl_funs)]
    b_scl = [scl_i(b_i) for b_i, scl_i in zip(b, scl_funs)]

    # Find t.
    for a_scl_i, b_scl_i, p_i, scl_i in zip(a_scl, b_scl, p, scl_funs):
        if p_i is not None:
            t = (scl_i(p_i) - a_scl_i) / (b_scl_i - a_scl_i)
            break
    else:
        raise ValueError("Requested point must include at least one known "
                         "value.")
    # Compute q.
    return [rev_i((1 - t) * a_scl_i + t * b_scl_i) for a_scl_i, b_scl_i, rev_i
            in zip(a_scl, b_scl, rev_funs)]
