"""
=============================================
Utilities (:mod:`pyavia.util`)
=============================================

.. currentmodule:: pyavia.util

Collection of less-utilities / functions / operations used in various
contexts.

XXXXXX
------

TODO
"""

from .compare_functions import compare_functions
from .display import Indenter
from .docstrings import doc
from .fortran import fortran_array, fortran_do, FortranArray

from .iter_ops import (
    all_in, all_none, all_not_none, any_in, any_none,
    bounded_by,
    count_op,
    find_bracket, first, flatten, flatten_list,
    split_dict, singlify)

from .type_ops import (
    coax_type, force_type,
    dataclass_fromlist, dataclass_names,
    make_sentinel)
