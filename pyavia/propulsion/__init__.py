"""
=====================================
Propulsion  (:mod:`pyavia.populsion`)
=====================================

.. currentmodule:: pyavia.propulsion

Functions / objects relating to propulsion analysis such as reciprocating
engines, propellers and gas turbines.

.. autosummary::
    :toctree:

    Propeller
    CSPropeller
    BEPropeller
    BEMPropeller
    BEMPropellerCS

"""

from ._propeller import Propeller, const_pitch_β
from ._cs_propeller import CSPropeller
from ._be_propeller import BEPropeller
from ._bem_propeller import BEMPropeller, BEMPropellerCS

