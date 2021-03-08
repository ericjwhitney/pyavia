"""
Useful small, general purpose utilities not offered in the standard library,
such as:

- Extended operators for generic types (e.g. kind_div, force_type, coax_type).
- Shorthand functions for commonly used tasks (e.g. all_none, all_not_none).
- Functions for searching, interpolating and extrapolating sequences (e.g.
  bracket_list, linear_int_ext, monotonic).
"""

# Last updated: 28 September 2020 by Eric J. Whitney.

import operator as op
import uuid
from dataclasses import fields, is_dataclass
from math import log, exp, pi, nan, atan
from typing import Union, Iterable, Sequence, List, Type

__all__ = ['kind_atan2', 'kind_div', 'force_type', 'coax_type', 'all_none',
           'all_not_none', 'any_in', 'any_none', 'first', 'count_op',
           'dataclass_fromlist', 'dataclass_names', 'bounded_by',
           'bracket_list', 'line_pt', 'linear_int_ext', 'min_max',
           'monotonic', 'strict_decrease', 'strict_increase', 'temp_filename']


# ----------------------------------------------------------------------------
# Operators.

def kind_atan2(y, x) -> float:
    """Implementation of atan2 that allows any object as an argument provided
    it supports __div__ and __float__, allowing units-aware usage."""
    if float(x) == 0.0:
        if float(y) > 0.0:
            return +0.5 * pi
        elif float(y) < 0.0:
            return -0.5 * pi
        else:
            return nan
    else:
        res = atan(y / x)
        if float(x) < 0:
            if float(y) >= 0:
                res += pi
            else:
                res -= pi
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


# -----------------------------------------------------------------------------
# Conversions.

def force_type(x, *types):
    """
    Try converting `x` into a series of types, with no check on the
    conversion validity.

    Examples
    --------
    >>> force_type(3.5, int, float)
    ... # Results in an int, because int(x) accepts float and truncates.
    3
    >>> force_type("3.5+4j", float, complex)
    (3.5+4j)
    >>> force_type(3.5+4j, int, float, str)
    '(3.5+4j)'

    Parameters
    ----------
    x :
        Argument to be converted.
    types : list_like
        Target types to use when trying conversion.

    Returns
    -------
    x_converted :
        `x` converted to the first successful type, if possible.

    Raises
    ------
    ValueError
        If no conversion was possible.
    """
    for this_type in types:
        try:
            return this_type(x)
        except (TypeError, ValueError):
            pass
    raise ValueError(f"Couldn't force {repr(x)} to "
                     f"{' or '.join(str(t) for t in types)}.")


def coax_type(x, *types, default=None):
    """
    Try converting `x` into a series of types, returning first result where
    next_type(`x`) == x.

    Examples
    --------
    >>> coax_type(3.5, int, float)  # float result.
    3.5
    >>> coax_type(3.0, int, str)  # int result.
    3
    >>> coax_type ("3.0", int, float)  # Error: 3.0 != "3.0".
    Traceback (most recent call last):
    ...
    ValueError: Couldn't coax '3.0' to <class 'int'> or <class 'float'>.
    >>> xa = 3 + 2j
    >>> coax_type(xa, int, float, default=xa)  # Can't conv., gives default.
    (3+2j)

    Parameters
    ----------
    x :
        Argument to be converted.
    types : list_like
        Target types to use when trying conversion.
    default :
        Value to return if conversion was unsuccessful.

    Returns
    -------
    x_converted :
        `x` converted to the first successful type (if possible) or default.

    Raises
    ------
    ValueError
        If default is None and conversion was unsuccessful.
    """
    for this_type in types:
        try:
            res = this_type(x)
            if res == x:
                return res
        except (TypeError, ValueError):
            pass
    if default is not None:
        return default
    else:
        raise ValueError(f"Couldn't coax {repr(x)} to "
                         f"{' or '.join(str(t) for t in types)}.")


# ----------------------------------------------------------------------------
# Shorthand functions for lists.

def all_none(*args):
    """Shorthand function for ``all(x is None for x in args)``.  Returns
    True if all `*args` are None, otherwise False."""
    return all(x is None for x in args)


def all_not_none(*args):
    """Shorthand function for ``all(x is not None for x in args)``.  Returns
    True if all `*args` are not None, otherwise False."""
    return all(x is not None for x in args)


def any_in(it: Iterable, target):
    """Shorthand function for ``any(x in target for x in iterable)``.
    Returns True if found, otherwise False."""
    return any(x in target for x in it)


def any_none(*args):
    """Shorthand function for ``any(x is None for x in args)``.  Returns
    True if any of `*args` are None, otherwise False."""
    return any(x is None for x in args)


def first(it: Iterable, condition=lambda x: True):
    # noinspection PyUnresolvedReferences
    """Function returning the first item in the iterable that satisfies the
    condition.  This function is taken almost directly from:
    https://stackoverflow.com/a/35513376

    >>> first((1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first(())
    Traceback (most recent call last):
    ...
    StopIteration

    Parameters
    ----------
    it : Iterable
        Iterable to search.
    condition : boolean function (optional)
        Boolean condition applied to each iterable.  If the condition is not
        given, the first item is returned.

    Returns
    -------
    first_item :
        First item in the iterable `it` that satisfying the condition.

    Raises:
        StopIteration if no item satysfing the condition is found.
    """
    return next(x for x in it if condition(x))


def count_op(it: Iterable, oper, value):
    """Return a count of the number of items in `it` where **oper** **value**
    == True.  This allows user-defined objects to be included and is subtly
    different to ``[...].count(...)`` which uses the __eq__ operator."""
    return [oper(x, value) for x in it].count(True)


# ----------------------------------------------------------------------------
# Useful loop constructs / generator functions.

def dataclass_fromlist(dc_type: Type, init_vals: Sequence):
    """Initialise a dataclass of type ``dc_type`` using a list of values
    ``init_vals`` ordered to match the class fields (i.e. as returned by
    ``dataclass_names``).  The length of the ``init_vals`` cannot exceed the
    number of dataclass field names.  If shorter, remaining fields are
    unassigned."""
    return dc_type(**{k: v for k, v in zip(dataclass_names(dc_type),
                                           init_vals)})


def dataclass_names(dc) -> List[str]:
    """When passed a type or specific instance of a dataclass, returns an
    ordered list containing the names of the individual fields."""
    if not is_dataclass(dc):
        raise AttributeError("Can't give field names for non-dataclass object.")
    if not isinstance(dc, type):
        dc = type(dc)
    return [f.name for f in fields(dc)]


# ----------------------------------------------------------------------------
# Simple search and interpolation functions.

def bounded_by(x, iterable, key=None):
    """Returns True if `x` is bounded by the given iterable.
    I.E. ``min(iterable) <= x <= max(iterable)``.  If `key` function is
    provided this is applied to `x` and the iterable values before comparison.
    """
    if key is None:
        key = _ignore_key
    key_list = [key(x) for x in iterable]
    return min(key_list) <= key(x) <= max(key_list)


def bracket_list(li, x, key=None):
    """
    Returns left and right indicies (`l_idx`, `r_idx`) of a sorted list that
    bracket `x`.
    I.E. where `li`[`l_idx`] <= `x` <= `li`[`r_idx`].

    .. Note:: This is not the same as the usual one-sided methods which ensure
       strict inequality on one side (e.g. low <= x < high).  This means
       that for boundary values two brackets may satisfy the condition.

    Parameters
    ----------
    li : List
        Sorted list / tuple.  Sorting can be in either direction.
    x :
        Value to bracket.
    key : function
        Comparison function if supplied.  Default = None.

    Returns
    -------
    l_idx, r_idx : Tuple
        Note that r_idx = l_idx + 1 on return.   For `x` equal to a middle
        list value, the left side bracket is returned.
    """
    l_idx = 0
    r_idx = len(li) - 1

    if l_idx >= r_idx:
        raise ValueError("List must contain at least two values.")
    if not bounded_by(x, [li[l_idx], li[r_idx]], key):
        raise ValueError("Value not in list range.")

    while r_idx - l_idx > 1:
        mid_idx = (l_idx + r_idx) // 2
        if bounded_by(x, [li[l_idx], li[mid_idx]], key):
            r_idx = mid_idx
        else:
            l_idx = mid_idx

    return l_idx, r_idx


def _ignore_key(x):
    """Shorthand function for use with some inbuilt functions that do
    not accept None as a key."""
    return x


def line_pt(a, b, p, scale=None):
    """Find the coordinates of a point `p` anywhere along the line `a` → `b`
    where at least one component of `p` is supplied (remaining can be None).
    Each axis may be optionally scaled.  There is no limitation that `p` is
    in the interval [`a`, `b`], so this function can also be used for
    extrapolation as required.

    Parameters
    ----------
    a, b : list_like
        Two distinct points on a line i.e. :math:`[x_1, ..., x_n]`
    p : list_like
        Required point on the line with at least a single known component,
        i.e. :math:`(..., None, p_i, None, ...)`.  If more than one value is
        supplied, the first is used.
    scale : list
        If supplied, a list corresponding to each axis [opt_1, ..., opt_n],
        where each axis can use the following options:

            - None: No scaling performed.
            - 'log': This axis is linear on a log scale.  In practice
                log(x) is performed on this axis prior to doing the
                interpolation / extrapolation, then exp(x) is done prior
                to returning.
    Returns
    -------
    list :
        Required point on line :math:`[q_1, ..., q_n]` where :math:`q_i = p_i`
        from above.
    """
    if scale is None:
        scale = [None] * len(a)
    if not len(a) == len(b) == len(p) == len(scale):
        raise ValueError("a, b, p, [scale] must be the same length.")

    # Scale axes.
    scl_funs, rev_funs = [], []
    for scl_str in scale:
        if scl_str is None:
            scl_funs.append(lambda x: x)
            rev_funs.append(lambda x: x)
        elif scl_str == 'log':
            scl_funs.append(lambda x: log(x))
            rev_funs.append(lambda x: exp(x))
        else:
            raise ValueError(f"Unknown scale type: {scl_str}.")

    a_scl = [scl_i(a_i) for a_i, scl_i in zip(a, scl_funs)]
    b_scl = [scl_i(b_i) for b_i, scl_i in zip(b, scl_funs)]

    # Find t.
    for a_scl_i, b_scl_i, p_i, scl_i in zip(a_scl, b_scl, p, scl_funs):
        if p_i is not None:
            t = (scl_i(p_i) - a_scl_i) / (b_scl_i - a_scl_i)
            break
    else:
        raise ValueError("Requested point must include at least one known "
                         "value.")
    # Compute q.
    return [rev_i((1 - t) * a_scl_i + t * b_scl_i) for a_scl_i, b_scl_i, rev_i
            in zip(a_scl, b_scl, rev_funs)]


def linear_int_ext(data_pts, p, scale=None, allow_extrap=False):
    """
    Interpolate data points to find remaining unknown values absent from
    `p` with optionally scaled axes. If `p` is not in the range and
    `allow_extra` == True, a linear extrapolation is done using the two data
    points at the end corresponding to the `p`.

    Parameters
    ----------
    data_pts : list_like(tuple)
        [(a_1, ... a_n), ...] sorted on the required axis (either direction).
    p : list_like
        Required point to interpolate / extrapolate with at least a single
        known component, i.e. :math:`(..., None, p_i, None, ...)`. If
        more than one is supplied, the first is used.
    scale :
        Same as ``line_pt`` scale.
    allow_extrap : bool, optional
        If True linear extrapolation from the two adjacent endpoints is
        permitted. Default = False.

    Returns
    -------
    list :
        Interpolated / extrapolated point :math:`[q_1, ..., q_n]` where
        :math:`q_i = p_i` from above.
    """
    if len(data_pts) < 2:
        raise ValueError("At least two data points required.")
    if scale is None:
        scale = [None] * len(data_pts[0])

    # Get working axis.
    for ax, x in enumerate(p):
        if x is not None:
            break
    else:
        raise ValueError("Requested point must include at least one known "
                         "value.")

    def on_axis(li):  # Return value along required axis.
        return li[ax]

    # Get two adjacent points for straight line.
    try:
        # Try interpolation.
        l_idx, r_idx = bracket_list(data_pts, p, key=on_axis)
    except ValueError:
        if not allow_extrap:
            raise ValueError(f"Point not within data range.")

        if ((on_axis(data_pts[0]) < on_axis(data_pts[-1])) != (
                on_axis(p) < on_axis(data_pts[0]))):
            l_idx, r_idx = -2, -1  # RHS extrapolation.
        else:
            l_idx, r_idx = 0, 1  # LHS extrapolation.

    return line_pt(data_pts[l_idx], data_pts[r_idx], p, scale)


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


# -----------------------------------------------------------------------------
# Useful file functions.

def temp_filename(prefix: str = '', suffix: str = '', rand_length: int = None):
    """Generates a (nearly unique) temporary file name with given `preifx` and
    `suffix` using a hex UUID, truncated to `rand_length` if required.  This is
    useful for interfacing with older DOS and FORTRAN style codes with
    specific rules about filename length."""
    return prefix + uuid.uuid4().hex[:rand_length] + suffix
