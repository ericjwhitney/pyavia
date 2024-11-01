"""
Math Extensions (:mod:`pyavia.numeric.math_ext`)
================================================

.. currentmodule:: pyavia.numeric.math_ext

Useful mathematical functions, particularly if these are not available in
NumPy.
"""
from __future__ import annotations

import operator as op
from collections.abc import Sequence, Iterable, Callable
from typing import Union, Any, TypeVar, Type

import numpy as np
import numpy.typing as npt

from pyavia.util.iter_ops import _ignore_key

# Written by Eric J. Whitney, 2020.

_T = TypeVar('_T')


# ======================================================================


# TODO This is a bit abstract.
def chain_mult(start_val: _T, it: Iterable[_T]) -> list[_T]:
    """
    Multiply ``start_val`` by each element in ``it`` in turn, producing
    a list of values the same length of ``it``.  The starting value is
    not included.

    Examples
    --------
    >>> print(chain_mult(5.0, [2.0, 3.0, 0.5]))
    [10.0, 30.0, 15.0]
    """
    res, val = [], start_val
    for mult in it:
        val *= mult
        res.append(val)
    return res


def equal_nan(a, b):  # TODO type hint.
    """
    Extended elementwise == comparison, which also returns true for
    `NaN` elements.
    """
    return (a == b) | (np.isnan(a) & np.isnan(b))


# TODO Delete ... mostly replace with np.ndim(...) > 0
def is_number_seq(obj) -> bool:
    """
    Returns ``True`` if ``obj`` is of ``Sequence`` type and is not a string /
    bytes / bytearray.
    """
    if isinstance(obj, Sequence):
        if not isinstance(obj, (str, bytes, bytearray)):
            return True
    return False


# TODO I am removing usages of this, so if it is still required, dim checks
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


def min_max(it: Iterable[_T], key=None) -> tuple[_T, _T]:
    """
    Returns (min, max) of elements in `iterable`.  Comparison provided
    using `key` if supplied.
    """
    if key is None:
        key = _ignore_key
    return min(it, key=key), max(it, key=key)


def monotonic(it: Iterable, dirn: int, strict: bool = True,
              key=None) -> bool:
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
def strict_decrease(it: Iterable, key=None) -> bool:
    """
    Shorthand for ``monotonic(iterable, -1, strict=True, key=key)``.
    Returns True *i.f.f.* all :math:`x_{i+1} < x_i`.  Comparison provided
    using `key` if supplied.
    """
    return monotonic(it, -1, strict=True, key=key)


def strict_increase(it: Iterable, key=None) -> bool:
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


# ======================================================================
# TODO Replace these older versions.

def sclvec_asarray(x: npt.ArrayLike[_T], *args, **kwargs
                   ) -> tuple[npt.NDArray[_T], bool]:
    """
    Convenience function to prepare a numpy array from a scalar or vector
    argument, as well as `True` if it is a scalar (or `False` otherwise).
    Equivalent to::

        res = np.asarray(x, *args, **kwargs)
        return res, res.ndim == 0
    """
    res = np.asarray(x, *args, **kwargs)
    return res, res.ndim == 0


def sclvec_return(x: npt.NDArray[_T], scalar: bool
                  ) -> npt.NDArray[_T] | _T:
    """If ``scalar == True``, applies returns ``x.squeeze()[()]``, otherwise
    returns `x` unchanged.
    """
    return x.squeeze()[()] if scalar else x


# ----------------------------------------------------------------------

# TODO These are the newer versions.

# TODO
# ScalarLike = int | float | complex | str | bytes | np.generic
# ScalarArrayLike = ScalarLike | npt.ArrayLike


def check_sclarray(x: npt.ArrayLike, ndim: int = 1,
                   dtype: Type[_T] = None,
                   copy: bool = False) -> tuple[npt.NDArray[_T], bool]:
    """
    Check parameter `x` and reformat as an `ndarray` of specified layout
    and type as required.  Whether `x` was a scalar or array is also
    returned for later use.

    Parameters
    ----------
    x : array_like or scalar
        Input argument as either a scalar or array-like sequence.
    ndim : int, default = 1
        Exact required number of dimensions in the output array.
    dtype : dtype, optional
        Type of the output array.
    copy : bool, default = False
        If `False` then a copy is only made if required for type
        conversion or reshaping.

    Returns
    -------
    result, single : tuple[ndarray, bool]
        `result` is an array of given `dtype` with `ndim` dimensions.
        `single` is:
            - `True` if `x` was a single value supplied as a scalar,
            - `False` in all other cases, even if `x` contained a single
              value contained in an n-D sequence / array.

    Raises
    ------
    ValueError
        If `x` cannot be converted to an array of `ndim` dimensions.
    """
    single = True if np.ndim(x) == 0 else False
    result = np.array(x, dtype=dtype, copy=copy, ndmin=ndim)
    if result.ndim != ndim:
        raise ValueError(f"Result array dimensions ndim = "
                         f"{result.ndim} incorrect, expected ndim = "
                         f"{ndim}.")
    return result, single


# ----------------------------------------------------------------------

def return_sclarray(x: npt.NDArray[_T], single: bool
                    ) -> npt.NDArray[_T] | _T:
    """
    Reformat the parameter `x` as a scalar or array, generally to
    match the form of a previously supplied input parameter (see
    `check_sclarray`).

    Parameters
    ----------
    x : NDArray
        Input array.
    single : bool
        Return `x` as-is if `False`, otherwise convert to a scalar
        value.

    Returns
    -------
    NDArray or scalar
        - If ``single=False``: Return `x` as-is.
        - If ``single=True``: Convert `x` to a scalar value before
          returning.

    Raises
    ------
    ValueError
        If `single=True`` but ``x.size != 1``.
   """
    if single:
        if x.size != 1:
            raise ValueError(f"Single value expected, got {x.size}.")
        return x.item(0)

    else:
        return x


# ----------------------------------------------------------------------

def within_range(it: Iterable[_T], x_range: tuple[_T, _T]
                 ) -> list[_T]:
    """
    Returns a list consisting of only the elements in `it` that are
    within the given closed interval `x_range` [`x_min`, `x_max`].
    """
    return [x for x in it if x_range[0] <= x <= x_range[1]]

# ======================================================================
