"""
==============================
Utilities (:mod:`pyavia.util`)
==============================

.. currentmodule:: pyavia.util

Collection of less-utilities / functions / operations used in various
contexts.  TODO

Checking / Comparison
---------------------

.. autosummary::
    :toctree:

    compare_funcs

Display / Documentation
-----------------------

.. autosummary::
    :toctree:

    Indenter
    doc

Iterable Operations
-------------------

.. autosummary::
    :toctree:

    all_in
    all_none
    all_not_none
    any_in
    any_none
    bounded_by
    count_op
    find_bracket
    first
    flatten
    flatten_list
    split_dict
    singlify

Type Operations
---------------

.. autosummary::
    :toctree:

    coax_type
    force_type
    dataclass_fromlist
    dataclass_names
    make_sentinel

"""

from .compare_funcs import compare_funcs
from .display import Indenter
from .docstrings import doc

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
