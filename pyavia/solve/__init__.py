"""
**pyavia.solve** provides functions for finding solutions to various types of
equations. These are included when not already covered by NumPy / SciPy or
when a different kind of algorithm is useful.
"""

from .bisect_root import bisect_root
from .dqnm import solve_dqnm
from .exception import SolverError
from .fixed_point import fixed_point
from .newton_bounded import newton_bounded
from .stepping import step_bracket_root, step_bracket_min



