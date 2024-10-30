"""
=====================================
Structures (:mod:`pyavia.structures`)
=====================================

.. module:: pyavia.structures

Functions relating to structures, stress analysis, fatigue and damage
tolerance.

.. autosummary::
    :toctree: _gen_pyavia_structures/

    kt_hole3d
    sn_raithby
    mohr2d

"""


from .kt_hole3d import kt_hole3d
from .sn_life import sn_raithby
from .stress import mohr2d
