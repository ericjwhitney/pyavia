"""
Fluids (:mod:`pyavia.fluids`)
=============================

.. currentmodule:: pyavia.fluids

Classes
-------

.. autosummary::
    :toctree:

    Gas
    PerfectGas
    ImperfectGas

Functions
---------

.. autosummary::
    :toctree:

    enthalpy
    stag_press_ratio
    stag_temp_ratio
    temp_from_enthalpy

"""

from .gas import Gas
from .perfect_gas import (PerfectGas, enthalpy, stag_press_ratio,
                          stag_temp_ratio, temp_from_enthalpy)
from .imperfect_gas import ImperfectGas
