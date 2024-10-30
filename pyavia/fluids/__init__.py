"""
=============================
Fluids (:mod:`pyavia.fluids`)
=============================

.. module:: pyavia.fluids

Classes
-------

.. autosummary::
    :toctree: _gen_pyavia_fluids/

    Gas
    PerfectGas
    ImperfectGas

Functions
---------

.. autosummary::
    :toctree: _gen_pyavia_fluids/

    enthalpy
    stag_press_ratio
    stag_temp_ratio
    temp_from_enthalpy

"""

from .gas import Gas
from .perfect_gas import (PerfectGas, enthalpy, stag_press_ratio,
                          stag_temp_ratio, temp_from_enthalpy)
from .imperfect_gas import ImperfectGas
