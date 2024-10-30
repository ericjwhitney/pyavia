"""
=====================================
Solvers (:mod:`pyavia.numeric.solve`)
=====================================

.. module:: pyavia.numeric.solve

Functions for finding solutions to various types of equations. These are
included when not already covered by NumPy / SciPy or when a different
kind of algorithm is useful.

Functions
---------

.. autosummary::
    :toctree: _gen_pyavia_numeric_solve/

    bisect_root
    solve_dqnm
    fixed_point
    newton_bounded
    step_bracket_min
    step_bracket_root

Exceptions
----------

.. autosummary::
    :toctree: _gen_pyavia_numeric_solve/

    SolverError

"""

from .bisect_root import bisect_root
from .dqnm import solve_dqnm
from .exception import SolverError
from .fixed_point import fixed_point
from .newton_bounded import newton_bounded
from .bracket import step_bracket_min, step_bracket_root
