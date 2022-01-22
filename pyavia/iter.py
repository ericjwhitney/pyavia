"""
Handy functions for dealing with various iterables.
"""

from typing import Iterable, Sequence, Dict


# =============================================================================

def all_none(*args):
    """Returns True if all arguments (or elements of a single iterable
    argument) are None."""
    if len(args) == 1 and isinstance(args[0], Iterable):
        args = args[0]
    if all(x is None for x in args):
        return True
    return False


def all_not_none(*args):
    """Shorthand function for ``all(x is not None for x in args)``.  Returns
    True if all `*args` are not None, otherwise False."""
    return all(x is not None for x in args)


def any_in(it: Iterable, target):
    """Shorthand function for ``any(x in target for x in iterable)``.
    Returns True if found, otherwise False."""
    return any(x in target for x in it)


def any_none(*args):
    """Returns True if any argument (or element of a single iterable
    argument) are None."""
    if len(args) == 1 and isinstance(args[0], Iterable):
        args = args[0]
    if any(x is None for x in args):
        return True
    return False


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


def count_op(it: Iterable, oper, value):
    """Return a count of the number of items in `it` where
    ``oper(value) == True``.  This allows user-defined objects to be included
    and is subtly different to ``[...].count(...)`` which uses the ``__eq__``
    operator."""
    return [oper(x, value) for x in it].count(True)


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


def flatten_list(li):
    """
    Generator similar to ``flatten``, however only flattens lists until a
    non-list element is found.  Note that non-lists may contain sub-lists
    and these are not flattened.

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


def split_dict(a: Dict, keys: Sequence) -> (Dict, Dict):
    """
    Split dict `a` by `keys` and return two dicts `x` and `y`:
        - `x`: Items in `a` having a key in `keys`.
        - `y`: All other items in `a`.
    """
    x, y = {}, {}
    for k, v in a.items():
        if k in keys:
            x[k] = v
        else:
            y[k] = v

    return x, y


def singlify(x):
    """If `x` only has exactly one element, return that element only.
    Otherwise return `x` unaltered. """
    return x[0] if (isinstance(x, Sequence) and len(x) == 1) else x


# -----------------------------------------------------------------------------


def _ignore_key(x):
    """Shorthand function for use with some inbuilt functions that do
    not accept None as a key."""
    return x
