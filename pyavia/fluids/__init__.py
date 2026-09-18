"""
Fluids (:mod:`pyavia.fluids`)
=============================

.. currentmodule:: pyavia.fluids

Protocols
---------

.. autosummary::
    :toctree:

    Gas
    GasFlow
    PerfectGas
    PolyGas

Perfect Gases
-------------

.. autosummary::
    :toctree:

    PerfectGas
    PerfectAir

Real / Polynomial Gases
-----------------------

.. autosummary::
    :toctree:

    PolyGas
    PolyAir

Functions
---------

.. autosummary::
    :toctree:

    init_gas
"""

from ._gas import Gas, GasFlow
from ._make_gas import make_gas
from ._air import PerfectAir, PerfectAirFlow, PolyAir, PolyAirFlow


# FutureWork: Add modules for other specific models.


