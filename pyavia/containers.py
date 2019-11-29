"""
Useful, less common containers.

Contains (hehe, get it?):
    MultiBiDict       Bi-directional dict class class.
    WeightedDirGraph  Weighted directional graph class.
"""

# Last updated: 29 November 2019 by Eric J. Whitney

from collections import deque, OrderedDict
from functools import reduce
from typing import (Dict, Any, Iterable, Sequence, Callable, Hashable,
                    Optional, List)

__all__ = ['MultiBiDict', 'WeightedDirGraph', 'link', 'flatten']


# -----------------------------------------------------------------------------

class MultiBiDict(dict):
    """
    Bi-directional dict class taken from this StackOverflow answer:
    https://stackoverflow.com/a/21894086A  A forward and inverse dictionary are
    synchronised to allow searching by either key or value to get
    the corresponding value / key.  Note:
        - The inverse dict bidict_multi.inverse auto-updates itself
        when the normal dict bidict_multi is modified.
        - Inverse directory entries bidict_multi.inverse[value] are always
        lists of key such that bidict_multi[key] == value.
        - Multiple keys can have the same value.

    Examples:
        >>> bd = MultiBiDict({'a': 1, 'b': 2})
        >>> print(bd)
        {'a': 1, 'b': 2}
        >>> print(bd.inverse)
        {1: ['a'], 2: ['b']}
        >>> bd['c'] = 1         # Now two keys have the same value (= 1)
        >>> print(bd)
        {'a': 1, 'b': 2, 'c': 1}
        >>> print(bd.inverse)
        {1: ['a', 'c'], 2: ['b']}
        >>> del bd['c']
        >>> print(bd)
        {'a': 1, 'b': 2}
        >>> print(bd.inverse)
        {1: ['a'], 2: ['b']}
        >>> del bd['a']
        >>> print(bd)
        {'b': 2}
        >>> print(bd.inverse)
        {2: ['b']}
        >>> bd['b'] = 3
        >>> print(bd)
        {'b': 3}
        >>> print(bd.inverse)
        {2: [], 3: ['b']}
    """

    def __init__(self, *args, **kwargs):
        super(MultiBiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(MultiBiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self_key = self[key]
        self.inverse.setdefault(self_key, []).remove(key)
        if self_key in self.inverse and not self.inverse[self_key]:
            del self.inverse[self_key]
        super(MultiBiDict, self).__delitem__(key)


# -----------------------------------------------------------------------------

# Note: At this stage, methods that change internal state simply clear the
# cache in lieu of a more sophisticated algorithm.

class WeightedDirGraph:
    """
    Implements a weighted directional graph where a value / weight can be
    assigned to any link / edge between any two hashable nodes / keys. Link
    values can be different in each direction (or completely absent), i.e. x
    -[value]-> y and x <-[reverse_value2]- y.  Intended for sparse graphs.
    Implemented as a dict(x-keys)-of-dicts(y-keys), both forward and reverse
    values are therefore stored independently for each pair of keys.

    Access to all links from a given key is the same as a dict, but access
    to a specific link between two keys uses the slice [:] notation.
    e.g.: wdg['a':'b'] returns the value between a -> b, whereas wdg['a']
    returns a dict of all link / edge values starting at 'a' i.e. {'b':
    'somevalue', 'c': 'anotherval'}.  Where slice notation cannot be used,
    module function link(a, b) is provided, e.g.: if link(a, b) in wdg:

    Examples:
        >>> wdg = WeightedDirGraph()
        >>> wdg['a':'b'] = 'Here'
        >>> print(wdg)
        WeightedDirGraph({'a': {'b': 'Here'}, 'b': {}})
        >>> print(wdg['a':'b'])
        Here
        >>> print(wdg['b':'a'])
        Traceback (most recent call last):
        ...
        KeyError: 'a'
        >>> wdg['a':3.14159] = (22, 7)
        >>> wdg['b':'c'] = 'it'
        >>> wdg['c':'d'] = 'is.'
        >>> path, joined = wdg.trace('a', 'd', op=lambda x, y: ' '.
        ...                         join([x, y]))
        >>> print(path, joined)
        ['a', 'b', 'c', 'd'] Here it is.
        >>> del wdg['a']
        >>> print(wdg)  # doctest:+ELLIPSIS
        WeightedDirGraph({'b': {'c': 'it'}, 3.14159: {}, ..., 'd': {}})
    """
    __slots__ = ('_links', '_cached_paths', '_cache_max_size_factor')

    def __init__(self, links: Dict[Hashable, Dict[Hashable, Any]] = None):
        """
        Construct a weighted directional graph.
        Args:
            links: If provided this is to be a dict-of-dicts representing
            forward links.  Each key corresponds to a dict holding other
            keys and the link value.  The following example creates a graph
            with linkages of
                a ---> b  link value = 2
                  |--> c  link value = 5
                b ---> (no links)
                c ---> a  link value = 4

            >>> wdg = WeightedDirGraph({'a': {'b': 2, 'c': 5}, 'c': {'a': 4}})
            >>> print(wdg['c':'a'])
            4
        """
        if links is not None:
            self._links = links
        else:
            self._links = {}
        
        # Add blank y-keys where reqd.
        self._add_blanks(self._links.keys())

        self._cached_paths = OrderedDict()
        self._cache_max_size_factor = 0.1  # Guess value = 10% of keys.

    def __getitem__(self, arg: slice):
        """
        Get link value for x -> y.
        Args:
            arg: Slice [from:to].

        Returns:
            Link value.

        Raises:
            KeyError if invalid slice argument or link does not exist.
        """
        _check_slice_arg(arg)
        return self._links[arg.start][arg.stop]

    def __setitem__(self, arg: slice, value):
        """
        Set / overwrite link value for x -> y.
        Args:
            arg: Slice [from:to].
            value: Link value.

        Returns:
            None

        Raises:
            KeyError for invalid slice argument (including path to
            nowhere x == y).
        """
        _check_slice_arg(arg)

        # Create x-key and y-key (if reqd) and set x -> y.
        self._add_blanks((arg.start, arg.stop))  # Add x and y key if reqd.
        self._links[arg.start][arg.stop] = value
        self._cached_paths.clear()

    def __delitem__(self, arg):
        """
        Delete link value for x -> y (slice arg) or delete key
        x and all associated links (single arg).
        Args:
            arg: Slice [from:to] or single value.

        Returns:
            None.

        Raises:
            KeyError for invalid slice argument or if link/s do not exist.
        """
        if isinstance(arg, slice):
            _check_slice_arg(arg)
            del self._links[arg.start][arg.stop]
        else:
            del self._links[arg]
            for xkey in self._links:
                if arg in self._links[xkey]:
                    del self._links[xkey][arg]
        self._cached_paths.clear()

    def __contains__(self, arg):
        """
        Returns True if a link value x -> y exists (slice  arg),
        or True if key exists (single arg).  Because standalone slice
        notation is  not available on the LHS, this syntax can be used:
        link(x, y) in wdg.
        Args:
            arg: Slice [from:to] or single value.

        Returns:
            Slice arg - True if a link value x -> y exists, else False.
            Single value arg -  True if key exists (single arg),
            else False.
        """
        if isinstance(arg, slice):
            _check_slice_arg(arg)
            return arg.stop in self._links[arg.start]
        else:
            return arg in self._links

    def __repr__(self):
        return f'WeightedDirGraph({repr(self._links)})'

    def _add_blanks(self, it: Iterable) -> None:
        """Insert independent empty x-key entries {key: {}} using keys from
        it. Does not overwrite any existing key."""
        for xkey in it:
            self._links.setdefault(xkey, {})

    def trace(self, x: Hashable, y: Hashable,
              op: Optional[Callable[[Any, Any], Any]] = None):
        """
        Trace shortest path between two existing keys x and y using a
        breath-first search (refer
        https://en.wikipedia.org/wiki/Breadth-first_search and
        https://www.python.org/doc/essays/graphs/).

        If 'op' is supplied, calculate the combined link / edge value by
        successively applying operator op to intermediate link values. E.G.:
        To determine xz_val x -[ xz_val]-> z we can compute xz_val = op(
        xy_val, yz_val) if we have x -[xy_val]-> y and y -[yz_val]-> z.

        Args:
            x: Starting node / vertex / key.
            y: Ending node / vertex / key.
            op: Operator that produce a combined value valid for any two
            link / edge values, i.e. result = op(val1, val2).

        Returns:
            path or (path, value):
                path: List of all points visited along the path
                including the end nodes, i.e. from x -> y: [x, i, j, k,
                y].  If no path is found it is None.
                value:  If op is given, the path value is computed as
                explained above if it covers more than one link. For direct
                links, the link value is returned. If 'op' is given but no
                path is found this is also None.

        Raises:
            KeyError: If x or y are not verticies, or if x == y.
        """
        def path_ret_val(flat: List):
            # If 'op' provided, reduce path to N-1 link values and apply
            # operator to combine them L-to-R_air. Tack onto the return value.
            if op is not None:
                if flat is not None:
                    assert len(flat) > 1
                    path_vals = [self._links[i][j]
                                 for i, j in zip(flat, flat[1:])]
                    return flat, reduce(op, path_vals)
                else:
                    return None, None
            else:
                return flat

        if x == y:
            raise KeyError(f"Can't trace path to nowhere: '{x}' -> '{y}'.")

        # Check the obvious cases first.
        if y in self._links[x]:  # Direct link already exists.
            return path_ret_val([x, y])
        if (x, y) in self._cached_paths:  # Previously cached path.
            return path_ret_val(self._cached_paths[x, y])

        # Breadth first search.
        discovered = {x: [x]}
        q = deque([x])  # Search frontier.
        while q:
            at = q.popleft()
            for x_new in self._links[at]:
                if x_new not in discovered:
                    discovered[x_new] = [discovered[at], x_new]
                    if x_new == y:  # Found, bail out.
                        q.clear()
                        break
                    else:
                        q.append(x_new)
        try:
            flat_path = [elem for elem in flatten_list(discovered[y])]
        except KeyError:
            flat_path = None

        # Cache the new path.  Allow at least 10 cache entries for small
        # graphs.
        max_cache_size = max(10, int(self._cache_max_size_factor *
                                     len(self._links)))
        if flat_path is not None:
            while len(self._cached_paths) >= max_cache_size:
                self._cached_paths.popitem()
            self._cached_paths[x, y] = flat_path

        return path_ret_val(flat_path)


def link(a, b) -> slice:
    """Return a standalone slice object when square bracket syntax is not
    available."""
    return slice(a, b)


def _check_slice_arg(arg: slice):
    if (not isinstance(arg, slice) or (arg.start == arg.stop) or
            (arg.start is None) or (arg.stop is None) or
            (arg.step is not None)):
        raise KeyError(f"Expected [from:to] or link(from, to).")


# -----------------------------------------------------------------------------

def flatten(seq):
    """
    Generator returning entries from a flattened representation of any
    sequence container (except strings).  Taken from
    https://stackoverflow.com/a/2158532

    Args:
        seq: Sequence container (list, tuple, etc)

    Yields:
        Each entry in turn.

    Examples:
        >>> for x in flatten((2, (3, (4, 5), 6), 7)):
        ...     print(x, end='')
        234567
    """
    for elem in seq:
        if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            yield from flatten(elem)
        else:
            yield elem


def flatten_list(li):
    """
    Generator similar to flatten(), however only flattens lists until a
    non-list element is found.  Note that non-lsits may contain sub-lists
    and these are not flattened.

    Args:
        li: List

    Yields:
        Each non-list entry in turn.
    """
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem
