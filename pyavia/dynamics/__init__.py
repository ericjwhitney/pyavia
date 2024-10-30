"""
=============================================
Dynamics (:mod:`pyavia.dynamics`)
=============================================

.. module:: pyavia.dynamics

Classes and functions relating to dynamics and motion in flight,
control, landing and related topics.

Ballistics
----------

.. autosummary::
    :toctree: _gen_pyavia_dynamics/

    ballistic           -- Compute ballistic trajectory, fixed drag area
    ballistic_variable  -- Compute trajectory, variable drag
    BallisticTrajectory -- Ballistic trajectory results object

Landing
-------

.. autosummary::
    :toctree: _gen_pyavia_dynamics/

    landing_energy      -- Compute aircraft landing dynamics and energy
    LandingImpact       -- Landing impact results object
"""

from .ballistic import (BallisticTrajectory, ballistic,
                        ballistic_variable)
from .landing import LandingImpact, landing_energy
