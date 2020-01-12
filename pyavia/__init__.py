"""
**PyAvia** is a collection of modules useful for common tasks in aerospace
engineering or engineering in general.  Some things to note:

- In all code snippets, it is assumed that PyAvia has been imported as follows:

    >>> import pyavia as pa

- Code examples are shown in the repository under ``examples/``.

.. warning::
    *CAVEAT COMPUTOR* - These modules are provided "as is", without warranty
    of any kind.  They are intended to be read and/or used by people trained
    in engineering and scientific methods who know how to verify results and
    who can recognise incorrect values when they see them... `which will
    happen frequently.`

Version
-------
The current version is **0.0.1**.  **PyAvia** is designed for Python >= 3.7
and is platform agnostic.

.. note::
    At this stage PyAvia is extremely preliminary, alpha, pre-release,
    etc.  Structural changes may be made to the code at any time that will
    almost definitely break third party code.  Please don't get cross.

Coming Soon
-----------
- Modular jet engine performance model.
"""

# Import core functions into pyavia.* namespace.  Note that all sub-packages
# define __all__.
from pyavia.core import *

# Sub-modules / sub-packages.
import pyavia.aero
import pyavia.solve

import sys

assert sys.version_info >= (3, 7)
