"""
==================================================
Containers (:mod:`pyavia.containers`)
==================================================

.. currentmodule:: pyavia.containers

Adds useful, less common containers not available in the standard
library.


.. autosummary::
    :toctree:

    AttrDict
    MultiBiDict
    ValueRange
    WtDirGraph
    WriteOnceDict
"""

from collections import deque, OrderedDict
from collections.abc import MutableMapping
from functools import reduce
from operator import is_
from typing import (Dict, Any, Iterable, Callable, Hashable,
                    Optional, List)

from pyavia.util.iter_ops import count_op, flatten_list  # Direct ref.

# Written by Eric J. Whitney, January 2020.


# ----------------------------------------------------------------------

# TODO Remove? SimpleNamespace covers this?
class AttrDict(dict):
    """
    AttrDict is a dictionary class that also allows access using
    attribute notation.

    Examples
    --------
    >>> my_dict = AttrDict([('make', 'Mazda'), ('model', '3')])
    >>> my_dict['yom'] = 2007
    >>> print(f"My car is a {my_dict.make} {my_dict.model} made in "
    ...       f"{my_dict.yom}.")
    My car is a Mazda 3 made in 2007.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


# -----------------------------------------------------------------------------

class MultiBiDict(dict):
    """
    Bi-directional dict class.

    A forward and inverse dictionary are synchronised to allow searching by
    either key or value to get the corresponding value / key.
    Implementation from this StackOverflow answer:
    https://stackoverflow.com/a/21894086.

    Notes
    -----

    - The inverse dict bidict_multi.inverse auto-updates itself when the
      normal dict bidict_multi is modified.
    - Inverse directory entries bidict_multi.inverse[value] are always lists
      of key such that bidict_multi[key] == value.
    - Multiple keys can have the same value.

    Examples
    --------
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
        """
        Parameters
        ----------
        args,kwargs : dict, optional
            Initialise the forward dict as dict(`*args`, `**kwargs`) if
            supplied.  The inverse dict is then constructed.
        """
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

# TODO probably remove this
class ValueRange:
    """Represents any range of scalar values (units agnostic)."""

    def __init__(self, *, x_min=None, x_max=None, x_mean=None, x_ampl=None):
        """Establish a range using any valid combination of two of the input
        arguments from `x_min`, `x_max`, `x_mean`, `x_ampl`.

        .. Note:: If minimum and maximum are reversed (or amplitude is
           negative) these will be automatically switched which may
           produce unexpected results.

        Parameters
        ----------
        x_min : scalar
            Range minimum.
        x_max : scalar
            Range maximum.
        x_mean : scalar
            Range mean.
        x_ampl : scalar
            Range amplitude of variation (i.e. +/- value to superimpose on
            mean).

        """
        if count_op([x_min, x_max, x_mean, x_ampl], is_, None) != 2:
            raise ValueError('Exactly two arguments required.')
        if x_min is None:
            if x_mean is not None:
                x_min = x_mean - x_ampl
            else:
                x_min = x_max - 2 * x_ampl
        if x_max is None:
            if x_mean is not None:
                x_max = x_mean + x_ampl
            else:
                x_max = x_min + 2 * x_ampl
        self.min, self.max = min([x_min, x_max]), max([x_min, x_max])

    @property
    def mean(self):
        """Computed mean value."""
        return 0.5 * (self.max + self.min)

    @property
    def ampl(self):
        """Computed amplitude."""
        return 0.5 * (self.max - self.min)


# -----------------------------------------------------------------------------


wdg_edge = slice
"""`wdg_edge` is a defined as an alias of the builtin slice function.  This is 
used to indicate a graph edge when square bracket syntax is not available.
"""


# Note: At this stage, methods that change internal state simply clear the
# cache in lieu of a more sophisticated algorithm.

class WtDirGraph:
    # noinspection PyUnresolvedReferences
    """
    A weighted directed graph where a value / weight can be assigned to any
    link / edge between any two hashable nodes / keys.
    Link values can be different in each direction (or completely absent),
    i.e. x →[value]→ y and x ←[reverse_value2]← y.  Intended for sparse
    graphs.  Implemented as a dict(x-keys)-of-dicts(y-keys), both forward
    and reverse values are therefore stored independently for each pair of
    keys.

    Access to all links from a given key is the same as a dict, but access to
    a specific link between two keys uses the slice [:] notation.
    e.g.: ``wdg['a':'b']`` returns the value between `a` → `b`, whereas
    ``wdg['a']`` returns a dict of all link / edge values starting at `a`
    i.e. ``{'b': 'somevalue', 'c': 'anotherval'}``.

    .. note:: Where slice / square bracket syntax cannot be used, alias
       wdg_edge(a, b) is defined as an alias of the builtin slice function.
       This is used to define a graph edge.  e.g.: ``if wdg_edge(a,
       b) in wdg:``.

    ..

    Examples
    --------
    >>> wdg = WtDirGraph()
    >>> wdg['a':'b'] = 'Here'
    >>> print(wdg)
    WtDirgraph({'a': {'b': 'Here'}, 'b': {}})
    >>> print(wdg['a':'b'])
    Here
    >>> print(wdg['b':'a'])
    Traceback (most recent call last):
    ...
    KeyError: 'a'
    >>> wdg['a':3.14159] = (22, 7)
    >>> wdg['b':'c'] = 'it'
    >>> wdg['c':'d'] = 'is.'
    >>> path, joined = wdg.trace('a', 'd', op=lambda x, y: ' '.join([x, y]))
    >>> print(path, joined)
    ['a', 'b', 'c', 'd'] Here it is.
    >>> del wdg['a']
    >>> print(wdg)
    WtDirgraph({'b': {'c': 'it'}, 3.14159: {}, 'c': {'d': 'is.'}, 'd': {}})
    """
    __slots__ = ('_links', '_cached_paths', '_cache_max_size_factor')

    def __init__(self, links: Dict[Hashable, Dict[Hashable, Any]] = None):
        """
        Construct a weighted directional graph.

        Parameters
        ----------
        links :
            If provided this is to be a dict-of-dicts representing forward
            links.  Each key corresponds to a dict holding other keys and
            the link value.  Example:

            >>> wdg = WtDirGraph({'a': {'b': 2, 'c': 5}, 'c': {'a': 4}})
            >>> print(wdg['c':'a'])
            4

            Creates a graph with linkages of:
            ::

                a -→ b  link value = 2
                  |→ c  link value = 5
                b -→    (no links)
                c -→ a  link value = 4
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
        Get link value for x → y.

        Parameters
        ----------
        arg : slice
            [from:to].

        Returns
        -------
        :
            Link value.

        Raises
        ------
        KeyError
            Invalid slice argument or link does not exist.
        """
        _check_slice_arg(arg)
        return self._links[arg.start][arg.stop]

    def __setitem__(self, arg: slice, value):
        """
        Set / overwrite link value for x → y.

        Parameters
        ----------
        arg : slice
            [from:to]
        value :
            Link value.

        Raises
        ------
        KeyError
            Invalid slice argument (including path to nowhere `x` == `y`).
        """
        _check_slice_arg(arg)

        # Create x-key and y-key (if reqd) and set x -> y.
        self._add_blanks((arg.start, arg.stop))  # Add x and y key if reqd.
        self._links[arg.start][arg.stop] = value
        self._cached_paths.clear()

    def __delitem__(self, arg):
        """
        Delete link value for x → y (slice arg) or delete key `x` and all
        associated links (single arg).

        Parameters
        ----------
        arg : slice or key
            If slice, [from:to] otherwise key value `x`.

        Raises
        ------
        KeyError
            Invalid slice argument or if link/s do not exist.
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
        Returns True if a link value x → y exists (slice arg), or True if
        key exists (single arg).  Because standalone slice notation is  not
        available on the LHS, use the following syntax:
        ``wdg_edge(x, y) in wdg``.

        Parameters
        ----------
        arg : slice or key
            If slice, [from:to] otherwise key value `x`.

        Returns
        -------
        result : bool
            - Slice arg: True if a link value `x` → `y` exists, else False.
            - Key arg: True if key exists (single arg), else False.
        """
        if isinstance(arg, slice):
            _check_slice_arg(arg)

            try:
                start_conns = self._links[arg.start]
                return arg.stop in start_conns
            except KeyError:
                return False  # Starting point not present.

        else:
            return arg in self._links

    def __repr__(self):
        return f'WtDirgraph({repr(self._links)})'

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

        If `op` is supplied, calculate the combined link / edge value by
        successively applying operator op to intermediate link values. E.G.
        To determine xz_val x →[xz_val]→ z we can compute xz_val =
        op(xy_val, yz_val) if we have x →[xy_val]→ y and y →[yz_val]→ z.

        Parameters
        ----------
        x,y : Hashable
            Starting and ending node / vertex / key.
        op : Callable[[Any, Any], Any]], optional
            Operator that produces a combined value valid for any two link /
            edge values, i.e. ``result = op(val1, val2)``.

        Returns
        -------
        path : List
            List of all points visited along the path including the end
            nodes, i.e. from x → y: [x, i, j, k, y].  If no path is found it
            is None.
        path,value: Tuple[List, Any]
            If `op` is given, the path value is computed and returned along
            with the path list (as above), if the path covers more than
            one link. For direct links, the link value is returned. If `op`
            is given but no path is found this is also None.

        Raises
        ------
        KeyError
            If `x` or `y` are not verticies, or if `x` == `y`.
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
            raise KeyError(f"Can't trace path to nowhere: '{x}' → '{y}'.")

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


def _check_slice_arg(arg: slice):
    if (not isinstance(arg, slice) or (arg.start == arg.stop) or
            (arg.start is None) or (arg.stop is None) or
            (arg.step is not None)):
        raise KeyError(f"Expected [from:to] or wdg_edge(from, to).")


# -----------------------------------------------------------------------------


class WriteOnceDict(MutableMapping, dict):
    """
    Dictionary subclass that prevents overwriting of values when the key is
    already in use.  Keys are allowed to be deleted, and after delete the
    key can be used again.

    Adapted from https://stackoverflow.com/a/21601690

    Examples
    --------
    >>> some_dict = WriteOnceDict({'a': 1, 'b' : 2})
    >>> some_dict
    {'a': 1, 'b': 2}
    >>> some_dict['a'] = 3  # Raises KeyError - already in use.
    Traceback (most recent call last):
    ...
    KeyError: "Overwriting key 'a' not allowed."
    >>> del some_dict['a']
    >>> some_dict['a'] = 3  # OK. Key was deleted beforehand.
    >>> some_dict
    {'b': 2, 'a': 3}
    >>> len(some_dict)
    2
    >>> bool(some_dict)
    True
    >>> some_dict.clear()
    >>> bool(some_dict)
    False
    """
    # These dict methods override the MutableMapping versions.
    __delitem__ = dict.__delitem__
    __getitem__ = dict.__getitem__
    __iter__ = dict.__iter__
    __len__ = dict.__len__

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Overwriting key '{key}' not allowed.")
        dict.__setitem__(self, key, value)
