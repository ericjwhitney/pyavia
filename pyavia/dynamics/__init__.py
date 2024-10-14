"""
=============================================
Dynamics (:mod:`pyavia.dynamics`)
=============================================

.. currentmodule:: pyavia.dynamics

.. toctree::
   :hidden:

PyAvia ``dynamics`` provides objects and functions relating to dynamics
and motion in flight, control, landing and related topics.

Ballistics
----------

.. autosummary::
    :toctree: generated/

    ballistic           -- Compute ballistic trajectory, fixed drag area
    ballistic_variable  -- Compute trajectory, variable drag
    BallisticTrajectory -- Ballistic trajectory results object

Landing
-------

.. autosummary::
    :toctree: generated/

    landing_energy      -- Compute aircraft landing dynamics and energy
    LandingImpact       -- Landing impact results object
"""

from .ballistic import (BallisticTrajectory, ballistic,
                        ballistic_variable)
from .landing import LandingImpact, landing_energy
