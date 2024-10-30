"""
=========================================
Aerodynamics (:mod:`pyavia.aerodynamics`)
=========================================

.. module:: pyavia.aerodynamics

Airspeed Functions
------------------

.. autosummary::
    :toctree: _gen_pyavia_aerodynamics/

    EAS2TAS  -- Equivalent airspeed to true airspeed.
    TAS2EAS  -- True airspeed to equivalent airspeed.


Atmosphere
----------

.. autosummary::
    :toctree: _gen_pyavia_aerodynamics/

    atmosphere      -- Atmosphere module.

XXX MORE

Notes
-----
- Unless specified otherwise, all angles are assumed to be radians for
  consistency.  Angle derivatives are similarly assumed to be per-radian.

- The term `foil` is generally used as the gender-neutral term for both
  `airfoil` and `aerofoil`.  The term 'aerofoil' is used in countries
  speaking UK-style English whereas 'airfoil' is more common in US-style
  English.  We consider the two terms interchangeable. :-)
"""

# 1-D Components.
from .airspeed import EAS2TAS, TAS2EAS
from .atmosphere import (Atmosphere, geo_alt_to_pot, pot_alt_to_geo)

# 2-D Components.
from .foil.base import Foil2DAero, Foil2DBasic, plot_foil_aero, std_α
from .foil.blend import Blend2DAero
from .foil.polar import Polar2DAero, Polar2DAeroPostStall
from .foil._geo import FoilGeo  # TODO Move this
from .foil.flat import Flat2DAero
from .foil.stall import PostStall2DMixin
from .foil.map_2d import Map2DAero, Map2DAeroPostStall
from .foil.qprop import QPROP2DAero, QPROP_from_model
from .foil.xfoil import read_XFOIL_polar
from .foil.xrotor import XROTOR2DAero, XROTOR2DAeroPostStall
