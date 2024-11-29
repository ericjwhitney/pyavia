"""
- Utilities for changing data between different / unusual types.
- Additional common types.
"""
from __future__ import annotations

import numbers
from dataclasses import is_dataclass, fields
from typing import Type, Sequence, List, TypeVar

Scalar = TypeVar('Scalar', int, float, complex)


# TODO This isn't really right, it should support Dim() as well... also
#  move this to units.py?


# ======================================================================


def coax_type(x, *types, default=None):
    """
    Try converting `x` into a series of types, returning first result which
    passes Test:  next_type(`x`) - x == 0.

    Examples
    --------
    >>> coax_type(3.5, int, float)  # float result.
    3.5
    >>> coax_type(3.0, int, str)  # int result.
    3
    >>> coax_type("3.0", int, float)  # Error: 3.0 != "3.0".
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
            if isinstance(x, numbers.Number):
                # Equality test applies to numeric values.
                if res - x == 0:  # More robust when 'x' is a Fraction, etc.
                    return res

        except (TypeError, ValueError):
            pass
    if default is not None:
        return default
    else:
        raise ValueError(f"Couldn't coax {repr(x)} to "
                         f"{' or '.join(str(t) for t in types)}.")


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


def dataclass_fromlist(dc_type: Type, init_vals: Sequence):
    """
    Initialise a dataclass of type `dc_type` using a list of values
    `init_vals` ordered to match the class fields (i.e. as returned by
    ``dataclass_names``).  The length of the `init_vals` cannot exceed the
    number of dataclass field names.  If shorter, remaining fields are
    unassigned.
    """
    return dc_type(**{k: v for k, v in zip(dataclass_names(dc_type),
                                           init_vals)})


def dataclass_names(dc) -> List[str]:
    """
    When passed a type or specific instance of a dataclass, returns an
    ordered list containing the names of the individual fields.
    """
    if not is_dataclass(dc):
        raise AttributeError("Can't give field names for non-dataclass "
                             "object.")
    if not isinstance(dc, type):
        dc = type(dc)
    return [f.name for f in fields(dc)]


# ============================================================================


def make_sentinel(name='_MISSING', var_name=None):
    """
    **This factory function is taken directly from ``boltons.typeutils``
    and has only cosmetic changes.**

    Creates and returns a new **instance** of a new class, suitable for
    usage as a "sentinel", a kind of singleton often used to indicate
    a value is missing when ``None`` is a valid input.

    Examples
    --------
    >>> make_sentinel(var_name='_MISSING')
    _MISSING

    Sentinels can be used as default values for optional function
    arguments, partly because of its less-confusing appearance in
    automatically generated documentation. Sentinels also function well
    as placeholders in queues and linked lists.

    .. note::By design, additional calls to ``make_sentinel`` with the
       same values will not produce equivalent objects.

    >>> make_sentinel('TEST') == make_sentinel('TEST')
    False
    >>> type(make_sentinel('TEST')) == type(make_sentinel('TEST'))
    False

    Parameters
    ----------
    name : str
        Name of the Sentinel.
    var_name : str (optional)
        Set this name to the name of the variable in its respective
        module to enable pickleability.
    """

    class Sentinel(object):
        def __init__(self):
            self.name = name
            self.var_name = var_name

        def __call__(self, *args, **kwargs):
            # Added __call__ in case an attempt is made.
            raise TypeError(f"Cannot call '{self.__repr__()}'.")

        def __repr__(self):
            if self.var_name:
                return self.var_name
            return '%s(%r)' % (self.__class__.__name__, self.name)

        if var_name:
            def __reduce__(self):
                return self.var_name

        @staticmethod
        def __nonzero__():
            return False

        __bool__ = __nonzero__

    return Sentinel()


# ======================================================================



def frozen(cls: type) -> type:
    # noinspection PyUnresolvedReferences
    """
    A decorator to freeze a class, i.e. make it effectively immutable.


    Examples
    --------
    First, let's define a simple class that will be frozen after
    initialisation:
    >>> @frozen
    ... class MyClass:
    ...     \"""MyClass docstring.\"""
    ...     def __init__(self, x):
    ...         \"""MyClass.__init__ docstring.\"""
    ...         self.x = x

    Attributes can be set during initialisation:
    >>> obj = MyClass(42)
    >>> obj.x
    42

    Once initialised, attributes can't be reassigned:
    >>> obj.x = 24
    Traceback (most recent call last):
    ...
    AttributeError: Can't set attribute 'x' on frozen object.

    Docstrings are preserved:
    >>> obj.__doc__
    'MyClass docstring.'
    >>> obj.__init__.__doc__
    'MyClass.__init__ docstring.'

    Derived classes can be added, but attributes must be set using
    ``object.__setattr__(...)`` (or the parent's superclass
    ``__setattr__``) method as all atributes are frozen:
    >>> class DerivedClass(MyClass):
    ...     def __init__(self, x, y):
    ...         super().__init__(x)
    ...         object.__setattr__(self, 'y', y)  # Note!

    >>> child_obj = DerivedClass(42, 84)
    >>> child_obj.x, child_obj.y
    (42, 84)

    >>> child_obj.y = 314
    Traceback (most recent call last):
    ...
    AttributeError: Can't set attribute 'y' on frozen object.
    """
    # Move and replace original __init__.
    cls.__frozen_init__ = cls.__init__

    def init_wrapper(self, *args, **kwargs):
        self.__frozen_init__(*args, **kwargs)
        self.__frozen = True

    init_wrapper.__doc__ = cls.__frozen_init__.__doc__
    cls.__init__ = init_wrapper

    # Add new __setattr__ method.
    def setattr_wrapper(self, name, value):
        if getattr(self, '__frozen', False):
            raise AttributeError(f"Can't set attribute '{name}' on "
                                 f"frozen object.")
        else:
            super(cls, self).__setattr__(name, value)

    cls.__setattr__ = setattr_wrapper

    return cls