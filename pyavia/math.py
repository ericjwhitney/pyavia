"""
Useful mathematical functions, particularly if these are not available in
NumPy.
"""
import operator as op
from typing import Union, Sequence, List, Callable, Any

import math

from pyavia.iter import _ignore_key


# =============================================================================

def chain_mult(start_val, li: Sequence) -> List:
    """Multiply ``start_val`` by each element in ``li`` in turn, producing a
    List of values the same length of ``li``.  The starting value is not
    included.

    Examples
    --------
    >>> print(chain_mult(5.0, [2.0, 3.0, 0.5]))
    [10.0, 30.0, 15.0]
    """
    res, val = [], start_val
    for mult in li:
        val *= mult
        res.append(val)
    return res


def is_number_seq(obj) -> bool:
    """
    Returns ``True`` if ``obj`` is of ``Sequence`` type and is not a string /
    bytes / bytearray.
    """
    if isinstance(obj, Sequence):
        if not isinstance(obj, (str, bytes, bytearray)):
            return True
    return False


def kind_atan2(y, x) -> float:
    """Implementation of atan2 that allows any object as an argument provided
    it supports __div__ and __float__, allowing units-aware usage."""
    if float(x) == 0.0:
        if float(y) > 0.0:
            return +0.5 * math.pi
        elif float(y) < 0.0:
            return -0.5 * math.pi
        else:
            return math.nan
    else:
        res = math.atan(y / x)
        if float(x) < 0:
            if float(y) >= 0:
                res += math.pi
            else:
                res -= math.pi
        return res


def kind_div(x, y) -> Union[int, float]:
    """Tries integer division of x/y before resorting to float.  If integer
    division gives no remainder, returns this result otherwise it returns the
    float result. From https://stackoverflow.com/a/36637240."""
    # Integer division is tried first because it is lossless.
    quo, rem = divmod(x, y)
    if not rem:
        return quo
    else:
        return x / y


def min_max(iterable, key=None):
    """Returns (min, max) of elements in `iterable`.  Comparison provided
    using `key` if supplied.
    """
    if key is None:
        key = _ignore_key
    return min(iterable, key=key), max(iterable, key=key)


def monotonic(iterable, dirn, strict=True, key=None):
    """Returns True if elements of `iterable` are monotonic otherwise false.
    For `dirn` >= 0 the sequence is strictly increasing i.e.
    :math:`x_{i+1} > x_i`, otherwise it is strictly decreasing i.e.
    :math:`x_i+1 < x_i`. If `strict` = False then equality is sufficient
    (i.e. >=, <= instead of >, <). Comparison provided using `key` if supplied.
    """
    if key is None:
        key = _ignore_key
    if dirn >= 0:
        cmp = op.gt if strict else op.ge
    else:
        cmp = op.lt if strict else op.le
    return all([cmp(key(y), key(x)) for x, y in zip(iterable, iterable[1:])])


def strict_decrease(iterable, key=None):
    """Shorthand for ``monotonic(iterable, -1, strict=True, key=key)``.
    Returns True *i.f.f.* all :math:`x_{i+1} < x_i`.  Comparison provided
    using `key` if supplied."""
    return monotonic(iterable, -1, strict=True, key=key)


def strict_increase(iterable, key=None):
    """Shorthand for ``monotonic(iterable, +1, strict=True, key=key)``.
    Returns True *i.f.f.* all :math:`x_{i+1} > x_i`.  Comparison provided
    using `key` if supplied."""
    return monotonic(iterable, +1, strict=True, key=key)


def vectorise(func: Callable, *values) -> Union[list, Any]:
    """
    Applies function ``func`` to one or more ``*values`` that can be
    either scalar or vector.  The function must take the same number of
    arguments as there are ``*values``:

    - If all values are numeric sequences, ``map()`` is used and a list is
      returned applying ``func`` to each of the value/s in turn.  This
      applies even if the ``len() == 1`` for the values.
    - If any value is not iterable: A scalar function and result is assumed.
    """
    if all([is_number_seq(x) for x in values]):
        return list(map(func, *values))

    return func(*values)
