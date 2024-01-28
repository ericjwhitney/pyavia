"""
Handy functions for dealing with various iterables.
"""
from collections.abc import Callable, Container, Iterable, Sequence
from typing import TypeVar

# Written by Eric J. Whitney, November 2019.

T = TypeVar('T')


# ======================================================================

def all_in(it: Iterable, target) -> bool:
    """
    Shorthand function for ``all(x in target for x in it)``. Returns
    `True` if every member of `it` is in `target`, otherwise `False`.
    """
    return all(x in target for x in it)


# ----------------------------------------------------------------------

def all_none(*args):
    """Returns True if all arguments (or elements of a single iterable
    argument) are None."""
    if len(args) == 1 and isinstance(args[0], Iterable):
        args = args[0]
    if all(x is None for x in args):
        return True
    return False


# ----------------------------------------------------------------------
def all_not_none(*args) -> bool:
    """
    Shorthand function for ``all(x is not None for x in args)``.  Returns
    True if all `*args` are not None, otherwise False.
    """
    return all(x is not None for x in args)



# ----------------------------------------------------------------------
def any_in(it: Iterable, target) -> bool:
    """Shorthand function for ``any(x in target for x in it)``.
    Returns True if found, otherwise False."""
    return any(x in target for x in it)


# ----------------------------------------------------------------------
def any_none(*args) -> bool:
    """Returns True if any argument (or element of a single iterable
    argument) are None."""
    if len(args) == 1 and isinstance(args[0], Iterable):
        args = args[0]
    if any(x is None for x in args):
        return True
    return False


# ----------------------------------------------------------------------


def bounded_by(it: Iterable, x, key: Callable = None) -> bool:
    """
    Returns `True` if `x` is bounded by the given iterable, i.e.
    ``min(iterable) <= x <= max(iterable)``.  If `key` function is
    provided this is applied to `x` and the iterable values before
    comparison.

    Parameters
    ----------
    x : Any
        Test value.
    it : Iterable
        Comparison values.
    key : Callable, default = None
        Comparison function accepting `x` values.
    Returns
    -------
    bool
        Returns `True` if `x` is bounded by the given iterable,
        otherwise `False`.
    """
    if key is None:
        key = _ignore_key
    key_list = [key(x) for x in it]
    return min(key_list) <= key(x) <= max(key_list)


# ----------------------------------------------------------------------


def count_op(it: Iterable, oper, value) -> int:
    """
    Return a count of the number of items in `it` where
    ``oper(value) == True``.  This allows user-defined objects to be
    included and is subtly different to ``[...].count(...)`` which uses
    the ``__eq__`` operator.
    """
    return [oper(x, value) for x in it].count(True)


# ======================================================================


def find_bracket(seq: Sequence, x, key: Callable = None) -> (int, int):
    """
    Returns left and right indicies of a sorted list (or iterable
    container) that bracket `x`, i.e. so that `seq[left_idx] <= x <=
    seq[right_idx]`.

    Parameters
    ----------
    x : Any
        Value to bracket.
    seq : Sequence
        Sorted list / sequence.  Sorting can be in either direction.
    key : Callable, default = None
        Comparison function accepting `x` values.

    Returns
    -------
    left_idx, right_idx : (int, int)
        Note that ``right_idx = left_idx + 1``.   If `x` is exactly
        equal to a value within the list (not at either end), the left
        side bracket is returned.

    Notes
    -----
    This is not the same as the usual one-sided methods which ensure
    strict inequality on one side (e.g. low <= x < high).  This means
    that for boundary values multiple brackets may satisfy the
    condition.
    """
    left_idx, right_idx = 0, len(seq) - 1
    if left_idx >= right_idx:
        raise ValueError("List must contain at least two values.")

    if not bounded_by(x, [seq[left_idx], seq[right_idx]], key):
        raise ValueError("Value not bounded by sequence.")

    while right_idx - left_idx > 1:
        mid_idx = (left_idx + right_idx) // 2
        if bounded_by([seq[left_idx], seq[mid_idx]], x, key):
            right_idx = mid_idx
        else:
            left_idx = mid_idx

    return left_idx, right_idx


# ======================================================================


def first(it: Iterable, condition=lambda x: True):
    # noinspection PyUnresolvedReferences
    """
    Function returning the first item in the iterable that satisfies
    the condition.  This function is taken almost directly from:
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
        Boolean condition applied to each iterable.  If the condition
        is not given, the first item is returned.

    Returns
    -------
    first_item :
        First item in the iterable `it` that satisfying the condition.

    Raises:
        StopIteration if no item satysfing the condition is found.
    """
    return next(x for x in it if condition(x))


# ----------------------------------------------------------------------

def flatten(seq):
    """
    Generator returning entries from a flattened representation of any
    sequence container (except strings).  Taken from
    https://stackoverflow.com/a/2158532

    Examples
    --------
    >>> for x in flatten((2, (3, (4, 5), 6), 7)):
    ...     print(x, end='')
    234567

    Parameters
    ----------
    seq : list_like
        Sequence container

    Yields
    ------
    :
        Each entry in turn.
    """
    for elem in seq:
        if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            yield from flatten(elem)
        else:
            yield elem


# ----------------------------------------------------------------------

def flatten_list(li):
    """
    Generator similar to ``flatten``, however only flattens lists until
    a non-list element is found.  Note that non-lists may contain
    sub-lists and these are not flattened.

    Parameters
    ----------
    li : List
        List to flatten.
    Yields
    ------
    :
        Each non-list entry in turn.
    """
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


# TODO flatten_type(it, it_type=Any):  Generalisation of flatten
#  and flatten_list.  Specifying type restricts sequential flattening.


# ----------------------------------------------------------------------


def split_dict(d: dict, keys: Container) -> (dict, dict):
    """
    Split dict `d` by `keys` and return two dicts `x` and `y`.

    Returns
    -------
    x, y : dict, dict
        - `x`: Items in `a` having a key in `keys`.
        - `y`: All other items in `a`.
    """
    x, y = {}, {}
    for k, v in d.items():
        if k in keys:
            x[k] = v
        else:
            y[k] = v

    return x, y


# ----------------------------------------------------------------------

# TODO This may not be required; instead return results that match the
#  input.
def singlify(x):
    """
    If `x` only has exactly one element, return that element only.
    Otherwise return `x` unaltered.
    """
    return x[0] if (isinstance(x, Sequence) and len(x) == 1) else x


# ======================================================================


def _ignore_key(x):
    """
    Shorthand function for use with some inbuilt functions that do not
    accept None as a key.
    """
    return x
