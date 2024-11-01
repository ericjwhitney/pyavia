from math import log10

from pyavia.units import Dim
from pyavia.containers import ValueRange


# ======================================================================

def sn_raithby(s_range: ValueRange):
    r"""
    S-N data for complete wings and tailplanes by Raithby (RAeS).

    This function uses the best fit equations of Sewell and Douglas as
    given in Appendix I of [1]_; these are straight lines on a log-log
    plot.

    .. note:: The equation in [1]_ Appendix I has a typo, the `B` term
       should be subtracted.

    Parameters
    ----------
    s_range : ValueRange[scalar or Dim]
        Stress range to use, valid when :math:`\sigma_{mean} \leq 30 ksi`.
        If no units are supplied, stresses are assumed to be ksi.

        .. note:: To handle cases where :math:`\sigma_{mean} < 0` a
           conservative simplification is made by setting
           :math:`\sigma_{mean} = 0`.

    Returns
    -------
    n : integer
        Calculated life (number of cycles).

    Raises
    ------
    ValueError
        If :math:`\sigma_{mean} > 30 ksi`.

    Notes
    -----
    .. [1] R. Hangartner, 'Correlation of Fatigue Data for Aluminium
           Aircraft Wing and Tail Structures', National Research Council
           Canada, Ottawa, Aeronautical Report LR-582, Dec. 1974.
    """
    s_m = max(Dim(s_range.mean, 'ksi').value, 0)
    s_a = Dim(s_range.ampl, 'ksi').value
    if 0 <= s_m < 2:
        a = 10.0556 - 0.51285 * s_m
        b = 4.8023 - 0.44000 * s_m
    elif 2 < s_m < 15:
        a = 9.45982 - 0.237677 * s_m + 0.0118776 * s_m ** 2 - 0.00025697 * \
            s_m ** 3
        b = 3.96687 - 0.021367 * s_m - 0.0004786 * s_m ** 2 + 0.00000657 * \
            s_m ** 3
    elif 15 <= s_m < 20:
        a = 8.5278 - 0.055198 * s_m
        b = 3.9947 - 0.028920 * s_m
    elif 20 <= s_m <= 30:
        a = 8.3050 - 0.044055 * s_m
        b = 3.9237 - 0.025373 * s_m
    else:
        raise ValueError("Mean stress must be <= 30 ksi.")

    return 10 ** (a - b * log10(s_a))

# -----------------------------------------------------------------------------
