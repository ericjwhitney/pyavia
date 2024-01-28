"""
This sub-package relates to dynamics motion in flight, control, landing
and other related topics.
"""

from .ballistic import (BallisticTrajectory, ballistic,
                        ballistic_variable)
from .landing import LandingImpact, landing_energy
