"""
Useful mathematical functions, particularly if these are not available in
NumPy.
"""
from __future__ import annotations
import operator as op
from collections.abc import Sequence, Iterable, Callable
from typing import Union, Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from pyavia.iter import _ignore_key

# Written by Eric J. Whitney, 2020.

_T = TypeVar('_T')


# =============================================================================

# TODO This is a bit abstract.
def chain_mult(start_val: _T, seq: Sequence[_T]) -> [_T]:
    """
    Multiply ``start_val`` by each element in ``li`` in turn, producing a
    List of values the same length of ``li``.  The starting value is not
    included.

    Examples
    --------
    >>> print(chain_mult(5.0, [2.0, 3.0, 0.5]))
    [10.0, 30.0, 15.0]
    """
    res, val = [], start_val
    for mult in seq:
        val *= mult
        res.append(val)
    return res


def equal_nan(a, b):
    """
    Extended elementwise == comparison, which also returns true for
    `NaN` elements.
    """
    return (a == b) | (np.isnan(a) & np.isnan(b))


def is_number_seq(obj) -> bool:  # TODO mostly replace with np.ndim(...) > 0
    """
    Returns ``True`` if ``obj`` is of ``Sequence`` type and is not a string /
    bytes / bytearray.
    """
    if isinstance(obj, Sequence):
        if not isinstance(obj, (str, bytes, bytearray)):
            return True
    return False


# TODO I am removing usage to this, so if it is still required, dim checks
#  will have to be done *inside here*.
def kind_arctan2(y, x) -> float:
    """
    Implementation of atan2 that allows any object as an argument provided
    it supports __div__ and __float__.
    """
    if float(x) == 0.0:
        if float(y) > 0.0:
            return +0.5 * np.pi
        elif float(y) < 0.0:
            return -0.5 * np.pi
        else:
            return np.nan
    else:
        res = np.arctan2(y, x)
        if float(x) < 0:
            if float(y) >= 0:
                res += np.pi
            else:
                res -= np.pi
        return res


def kind_div(x, y) -> int | float:
    """
    Tries integer division of x/y before resorting to float.  If integer
    division gives no remainder, returns this result otherwise it returns the
    float result.

    References
    ----------
    .. [1] Original implementation: https://stackoverflow.com/a/36637240.
    """
    # Integer division is tried first because it is lossless.
    quo, rem = divmod(x, y)
    if not rem:
        return quo
    else:
        return x / y


def min_max(it: Iterable, key=None):
    """
    Returns (min, max) of elements in `iterable`.  Comparison provided
    using `key` if supplied.
    """
    if key is None:
        key = _ignore_key
    return min(it, key=key), max(it, key=key)


def monotonic(it: Iterable, dirn, strict=True, key=None):
    """
    Returns True if elements of `iterable` are monotonic otherwise false.
    For `dirn` >= 0 the sequence is strictly increasing i.e. :math:`x_{i+1}
    > x_i`, otherwise it is strictly decreasing i.e. :math:`x_i+1 < x_i`. If
    `strict` = False then equality is sufficient (i.e. >=, <= instead of >,
    <). Comparison provided using `key` if supplied.
    """
    if key is None:
        key = _ignore_key
    if dirn >= 0:
        cmp = op.gt if strict else op.ge
    else:
        cmp = op.lt if strict else op.le
    return all([cmp(key(y), key(x)) for x, y in zip(it, it[1:])])


# TODO may be redundant if numpy use widespread; e.g. np.all(np.diff(a) > 0)
def strict_decrease(it: Iterable, key=None):
    """
    Shorthand for ``monotonic(iterable, -1, strict=True, key=key)``.
    Returns True *i.f.f.* all :math:`x_{i+1} < x_i`.  Comparison provided
    using `key` if supplied.
    """
    return monotonic(it, -1, strict=True, key=key)


def strict_increase(it: Iterable, key=None):
    """
    Shorthand for ``monotonic(iterable, +1, strict=True, key=key)``.
    Returns True *i.f.f.* all :math:`x_{i+1} > x_i`.  Comparison provided
    using `key` if supplied.
    """
    return monotonic(it, +1, strict=True, key=key)


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


# ============================================================================
def sclvec_asarray(x: ArrayLike, *args, **kwargs) -> tuple[np.ndarray, bool]:
    """
    Convenience function to prepare a numpy array from a scalar or vector
    argument, as well as `True` if it is a scalar (or `False` otherwise).
    Equivalent to::

        res = np.asarray(x, *args, **kwargs)
        return res, res.ndim == 0
    """
    res = np.asarray(x, *args, **kwargs)
    return res, res.ndim == 0


def sclvec_return(x: np.ndarray, scalar: bool) -> ArrayLike:
    """If ``scalar == True``, applies returns ``x.squeeze()[()]``, otherwise
    returns `x` unchanged.
    """
    return x.squeeze()[()] if scalar else x


# ============================================================================

def within_range(seq: [_T], x_range: (_T, _T)) -> [_T]:
    """
    Returns a list consisting only the `seq` elements that are within the
    given closed interval `x_range` [`x_min`, `x_max`].
    """
    return [x for x in seq if x_range[0] <= x <= x_range[1]]
