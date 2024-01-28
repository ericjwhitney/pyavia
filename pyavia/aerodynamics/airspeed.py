"""
Various functions for calculating airspeeds.
"""

# Written by Eric J. Whitney, January 2023.


# ===========================================================================

def Veas2Vtas(Veas: float, σ: float) -> float:
    """
    Convert equivalent airspeed to true airspeed.  This is the inverse of
    `Vtas2Veas`; refer to that function for complete details.
    """
    return Veas / σ ** 0.5


# ---------------------------------------------------------------------------

def Vtas2Veas(Vtas: float, σ: float) -> float:
    """
    Convert true airspeed to equivalent airspeed per `V_E = V_T * σ⁰ᐧ⁵`.
    The equivalent airspeed is one that gives the same dynamic pressure as
    the true airspeed in an ISA sea level atmosphere.

    Parameters
    ----------
    Vtas : float.
        True airspeed.
    σ :
        Ratio of actual ambient air density to ISA sea level value.

    Returns
    -------
    v_equiv: scalar or dimensioned scalar.
        Equivalent airspeed.
    """
    return Vtas * σ ** 0.5
