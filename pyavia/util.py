"""
Useful small utilities for strings and numbers not offered in the standard
library.

Contains:
    kind_div            Function for division that checks for integer result.
    force_type          Function to force the result to the required type.
    coax_type           Function to convert the result to the required type if
                        possible.
    UCODE_SS_CHARS      List of unicode superscript characters.
    from_ucode_super    Convert unicode sueprscripts to an ASCII string.
    to_ucode_super      Convert an ASCII string to unicode superscripts.
    bisect_root         Simple root finding function by bisection.
    bounded_by          Function checking if a value is bounded by a range.
    bracket_list        Function to find the sorted list entries either side
                        of the given value.
    fixed_point         Simple iteration of a given function until the
                        output converges x = f(x).
    fixed_point_scalar  Ident. to fixed_point() for scalar args.
    line_pt             Function to give a point on a line of two other points
                        with any one coordinate supplied.
    linear_int_ext      Function for linear interpolation (and optional
                        extrapolation).
    min_max             Function giving the min and max of an object
                        simultaneously.
    monotonic           Function checking that sequence values are monotonic.
    strict_decrease     Specialisation of monotonic.
    strict_increase     Specialisation of monotonic.

"""

# Last updated: 29 November 2019 by Eric J. Whitney.

import operator as op
from math import log, exp
from typing import Union, Callable, Any, List

__all__ = ['kind_div', 'force_type', 'coax_type', 'UCODE_SS_CHARS',
           'from_ucode_super', 'to_ucode_super', 'bisect_root',
           'bounded_by', 'bracket_list', 'fixed_point', 'fixed_point_scalar',
           'line_pt', 'linear_int_ext', 'min_max', 'monotonic',
           'strict_decrease', 'strict_increase']


# -----------------------------------------------------------------------------
# Operators.

def kind_div(x, y) -> Union[int, float]:
    """First attempts integer division of x/y (float or int). If it gives no
    remainder, returns the result. Otherwise return the ordinary float
    result. From:  https://stackoverflow.com/a/36637240"""
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
    Try to convert x into each type in turn using the type constructor i.e.
    next_type(x). No check is made to determine if the conversion is
    meaningful. Raises ValueError if no conversion was possible.  Examples:
    >>> force_type(3.5, int, float)  # int.  int(x) accepts float and truncs.
    3
    >>> force_type("3.5+4j", float, complex)
    (3.5+4j)
    >>> force_type(3.5+4j, int, float, str)
    '(3.5+4j)'
    """
    for this_type in types:
        try:
            return this_type(x)
        except (TypeError, ValueError):
            pass
    raise ValueError(f"Couldn't force {repr(x)} to "
                     f"{' or '.join(str(t) for t in types)}.")


def coax_type(x, *types, default=None):
    """Try to convert x into each type in turn using the type constructor
    i.e. next_type(x).  Return the first successful conversion that is equal
    i.e. next_type(x) == x.

    If unsuccessful return default if provided.  If default is None,
    raise ValueError.

    Examples:
    >>> coax_type(3.5, int, float)  # float result.
    3.5
    >>> coax_type(3.0, int, str)  # int result.
    3
    >>> coax_type ("3.0", int, float)  # Error: 3.0 != "3.0".
    Traceback (most recent call last):
    ...
    ValueError: Couldn't coax '3.0' to <class 'int'> or <class 'float'>.
    >>> x = 3 + 2j
    >>> coax_type(x, int, float, default=x)  # Can't convert, returns default.
    (3+2j)
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


# -----------------------------------------------------------------------------
# Unicode numbers.

UCODE_SS_CHARS = ('⁺⁻ᐧ⁰¹²³⁴⁵⁶⁷⁸⁹', '+-.0123456789')


def from_ucode_super(ss: str) -> str:
    """
    Convert any unicode numeric superscipt characters in the string to
    normal ascii text.
    Args:
        ss: String with unicode superscript characters.

    Returns:
        Converted string.
    """
    result = ''
    for c in ss:
        idx = UCODE_SS_CHARS[0].find(c)
        if idx >= 0:
            result += UCODE_SS_CHARS[1][idx]
        else:
            result += c
    return result


def to_ucode_super(ss: str) -> str:
    """
    Convert numeric characters in the string to unicode superscript.
    Args:
        ss: String with ascii numeric characters.

    Returns:
        Converted string.
    """
    result = ''
    for c in ss:
        idx = UCODE_SS_CHARS[1].find(c)
        if idx >= 0:
            result += UCODE_SS_CHARS[0][idx]
        else:
            result += c
    return result


# -----------------------------------------------------------------------------
# Simple search and interpolation functions.

def bisect_root(f: Callable[[Any], Any], x_a, x_b, max_its: int = 50,
                f_tol=1e-6, display=False) -> Any:
    """
    Approximate solution of f(x)=0 on interval [x_a, x_b] by bisection
    method. For bisection to work f(x) must change sign across the interval,
    i.e. f(x_a) and f(x_b) must have opposite sign.

    Note: This function is able to be used with arbirary units.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisect_root(f, 1, 2, 25)  # This will take 17 iterations.
    1.6180343627929688
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisect_root(f, 0, 1, 10)  # Only takes 1 iteration (in middle).
    0.5

    Args:
        f: Function to find root f(x_m) -> 0.
        x_a, x_b: Each end of the search interval, in any order.
        max_its:  Maximum number of iterations.
        f_tol: End search when abs(f(x)) < f_tol.
        display: (bool) If True, print progress statements.

    Returns:
        xm: Best estimate of root found i.e. f(x_m) approx = 0.0.

    Raises:
        RuntimeError is max_its is reached before a solution is found.
    """
    if display:
        print(f"bisect_root:")
    x_a_next, x_b_next = x_a, x_b
    f_a_next, f_b_next = f(x_a_next), f(x_b_next)
    try:
        # We use division approach to compare signs instead of multiplication as
        # it neatly cancels units if present.  But this requires a check for
        # f(x_b) == 0 first.
        if f_a_next/f_b_next >= 0:   # Sign comp. via division.
            raise ValueError(f"f(x_a) and f(x_b) must have opposite sign.")
    except ZeroDivisionError:
        raise ValueError(f"One of the start points is already zero.")

    it = 0
    while True:
        # Compute midpoint.
        x_m = (x_a_next + x_b_next)/2
        f_m = f(x_m)
        it += 1

        if display:
            print(f"\tIteration {it}: x = [{x_a_next}, {x_m}, {x_b_next}] "
                  f"--> f = [{f_a_next}, {f_m}, {f_b_next}]")

        # Check stopping criteria.
        if abs(f_m) < f_tol:
            return x_m

        if it >= max_its:
            raise RuntimeError(f"Reached {max_its} iteration limit.")

        # Check which side root is on, narrow interval.
        if f_a_next / f_m < 0:  # Sign comp. via division.
            x_b_next = x_m
            f_b_next = f_m
        else:
            x_a_next = x_m
            f_a_next = f_m



def bounded_by(x, iterable, key=None):
    """Returns True the value x is bounded by the given iterable it, i.e.
    min(seq) <= x <= max(seq).  If a key function is provided this is applied
    to x and the iterable before comparison.
    """
    if key is None:
        key = _ignore_key
    key_list = [key(x) for x in iterable]
    return min(key_list) <= key(x) <= max(key_list)


def bracket_list(li, x, key=None):
    """Returns (l_idx, r_idx) indices of sorted list/tuple 'li' which brackets
    x i.e. li[l_idx] <= x <= li[r_idx].  Note that this is not the same as
    the usual one-sided method with strict inequality i.e. low <= x < high

    r_idx = l_idx + 1 on return. Sorting can be in either direction.  For x
    equal to a middle list value, the left side bracket is returned.
    Comparison uses key function if supplied.
    """
    # EJW Removed 25/11 duck typing policy
    # if not isinstance(li, list):
    #     raise TypeError("Requires list.")

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
    """Satisfies some inbuilt functions that do not accept None as a key."""
    return x


def fixed_point(func: Callable[[Any], Any], x_start: List[Any],
                x_tol: List[Any], relax=1.0, max_its: int = 50, display=False):
    """
    Find the fixed point of a function x = f(x) by iteratively passing an
    initial estimate through the function.  The fixed point is a vector
    (list).

    Args:
        func: Function that returns a better estimate of vector 'x'.
        x_start: Starting value for 'x'. List of any numeric type (including
        user types) may be used.  Individual elements need not be the
        same type.
        x_tol: Stop when largest component abs(x_new_i - x_i) < x_tol_i.
        This is a list (vector) of any numeric type aligned with x.
        Components of x_tol that are None are ignored.
        relax: Relaxation factor.  After the next trial value 'x_step' is
        computed, it is revised to x' as follows:
            x' = x + relax * (x_step - x)
        If relax = 1 this is equivalent to standard iteration using x_step.
        When relax < 1 (e.g. 0.5) this is an under-relaxation which can add
        stability.
        max_its: (int) Iteration limit.
        display: (bool) If True, print iterations.

    Returns:
        Converged x value.
    """
    def nice_str(xli: List):
        return ', '.join([str(x_i) for x_i in xli])

    if display:
        print(f"fixed_point: Start x = {nice_str(x_start)}")
    x, its = x_start, 0
    n = len(x)  # More compact than zips.
    while True:
        x_step = func(x)
        x_next = [x[i] + relax * (x_step[i] - x[i]) for i in range(n)]
        x_err = [abs(x_next[i] - x[i]) for i in range(n)]
        its += 1

        if display:
            print(f"\tIteration {its}: x = {nice_str(x_next)}, "
                  f"Error = {nice_str(x_err)}")

        if all([x_err[i] <= x_tol[i] for i in range(n)]):
            break

        x = x_next
        if its >= max_its:
            raise RuntimeError(f"Limit of {max_its} iterations exceeded.")
    return x_next


def fixed_point_scalar(func: Callable[[Any], Any], x_start, x_tol,
                       relax=1.0, max_its: int = 15, display=False):
    """Identical to fixed_point() except it wraps scalar arguments for
    convenience."""

    def func_wrap(x):
        return [func(x[0])]

    return fixed_point(func_wrap, x_start=[x_start], x_tol=[x_tol],
                       relax=relax, max_its=max_its, display=display)


def line_pt(a, b, p, scale=None):
    """Find the coordinates of a point p anywhere along the line a --> b
    where at least one component of p is supplied (remaining can be None).
    Each axis may be optionally scaled.

    Note: There is no limitation that p is in the interval [a, b], so this
    function can also be used for extrapolation as required.

    Args:
        a: Two distinct points on the line (x_1, ... x_n)
        b: Ditto.
        p: Required point on the line with at least a single known
            component, i.e. (..., None, p_i, None, ...).  If more
            than one is supplied, the first is used.
        scale: If supplied, a list corresponding to each axis. Options:
            - None: No scaling performed.
            - 'log': This axis is linear on a log scale.  In practice
                log(x_i) is performed on this axis prior to doing the
                interpolation / extrapolation, then exp(x) is done prior to
                returning.
    Returns:
        - Required point on line [q_1, ..., q_n] where q_i = p_i from args.
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
    Interpolate data_pts to find remaining unknown values in p with
    optionally scaled axes. If p is not in the range and allow_extra ==
    True, a linear extrapolation is done using the two data points at the
    end corresponding to the p.

    Args:
        data_pts: [(a_1, ... a_n), ...] sorted on the required axis
            (either direction).
        p: Required point to interpolate / extrapolate with at least
            a single known component, i.e. (..., None, p_i, None, ...).
            If more than one is supplied, the first is used.
        scale: Same as line_pt scale.
        allow_extrap: True if linear extrapolation from the two adjacent
        endpoints is permitted.

    Returns:
        - Interpolated / extrapolated point [q_1, ..., q_n] where
            q_i = p_i from args.
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
    """Return (min, max) of elements in seq.  Comparison provided
    by key if supplied.
    """
    if key is None:
        key = _ignore_key
    return min(iterable, key=key), max(iterable, key=key)


def monotonic(iterable, dirn, strict=True, key=None):
    """Return True if elements of seq are monotonic otherwise false.  For
    dirn >= 0 the sequence is increasing i.e. x_i+1 > x_i, otherwise
    strictly decreasing i.e. x_i+1 < x_i. If strict = False then equality
    is sufficient (i.e. >=, <= instead of >, <).
    """
    if key is None:
        key = _ignore_key
    if dirn >= 0:
        cmp = op.gt if strict else op.ge
    else:
        cmp = op.lt if strict else op.le
    return all([cmp(key(y), key(x)) for x, y in zip(iterable, iterable[1:])])


def strict_decrease(iterable, key=None):
    """True iff all x_i+1 < x_i."""
    return monotonic(iterable, -1, strict=True, key=key)


def strict_increase(iterable, key=None):
    """True iff all x_i+1 > x_i."""
    return monotonic(iterable, +1, strict=True, key=key)
