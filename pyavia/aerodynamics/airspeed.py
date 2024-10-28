"""
Functions for calculating / converting airspeeds.
"""


# Written by Eric J. Whitney, January 2023.

# ======================================================================

def EAS2TAS(Veas: float, σ: float) -> float:
    """
    Convert equivalent airspeed to true airspeed.  This is the inverse
    of `Vtas2Veas`; refer to that function for complete details.
    """
    return Veas / σ ** 0.5


# ----------------------------------------------------------------------

def TAS2EAS(VTAS: float, σ: float) -> float:
    r"""
    Convert true airspeed to equivalent airspeed per :math:`V_E = V_T
    \sqrt{σ}`.  The equivalent airspeed is one that gives the same
    dynamic pressure as the true airspeed in an ISA sea level 
    atmosphere.

    Parameters
    ----------
    VTAS : float.
        True airspeed.
    σ :
        Ratio of actual ambient air density to ISA sea level value.

    Returns
    -------
    v_equiv: scalar or dimensioned scalar.
        Equivalent airspeed.
    """
    return VTAS * σ ** 0.5
