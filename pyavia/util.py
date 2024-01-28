"""
Small, general purpose utility functions.
"""
from __future__ import annotations

# Written by: Eric J. Whitney  Last updated: 16 May 2022.


import uuid
from typing import Callable


# == I/O Functions ==========================================================

class Indenter:
    """
    Indenter is used to print information from multi-level nested
    functions.  Depending on level / function depth the output message is
    indented or supressed.  This is done by tracking the number of
    `Indenter` objects that exist when the print statement is called.

    Examples
    --------
    First defined an 'inner' working function:
    >>> def inner_func(display: int = 0):
    ...     indent = Indenter(display)  # Output for this level.
    ...     # ... does some other things ...
    ...     indent("In inner_func()...")

    Finally define a top-level function definition that calls the 'inner'
    function:
    >>> def top_level(display: int | bool = False):
    ...     indent = Indenter(display)  # Output for this level.
    ...     # Does some things.
    ...     indent("In top_level()...")
    ...     inner_func(display=indent.next_level)

    Running `top_level` with default arguments (equivalent to
    ``display=False`` or ``display=0``) produces no output:
    >>> top_level()

    Running `top_level` with ``display=True`` (equivalent to ``display=1``):
    >>> top_level(display=True)
    In top_level()...

    Running `top_level` with ``display=2`` produces indented output:
    >>> top_level(display=2) # doctest: +NORMALIZE_WHITESPACE
    In top_level()...
        In inner_func()...
    """
    _active = 0

    def __init__(self, level: bool | int = 0):
        """
        Parameters
        ----------
        level : int | bool, optional
            The display level of the function where the `Indenter` is used.

            - `int`:  Values > 0 mean information will be printed.
               Typically subsequent levels are given the next lower level
               (level - 1, see `next_level` property).
            - `bool`: Converted to `int`, `True` = 1 and `False` = 0.
        """
        Indenter._active += 1
        self._level = int(level)

    def __call__(self, *args, **kwargs):
        """
        Print information, indented using tabs when required.  All
        arguments are passed directly to the underlying Python ``print``
        function.
        """
        if self._level > 0:
            # Separate print for tabs to avoid extraneous whitespace caused
            # by empty string.
            if self._active > 1:
                print('\t' * (Indenter._active - 1), end='')

            print(*args, **kwargs)

    def __del__(self):
        Indenter._active -= 1

    # -- Public Methods ---------------------------------------------------

    @property
    def next_level(self) -> int:
        """Shorthand property returning level - 1 (see `__init__`)."""
        return self._level - 1


# ======================================================================
# Alternative approach to Indenter.

_DISP_CURRENT_LEVEL = 1
_DISP_MAX_LEVEL: int | None = None  # Disabled = None, Enabled = 1, 2, ...


def disp_enter(disp: int | bool = None):
    """
    Called when entering a function scope where nested display is
    desired.  See ``disp_print(...)`` for examples.

    Parameters
    ----------
    disp : int or bool, optional
        - `None` (default): The current nested function level is
          automatically updated.
        - `int`: Sets the maximum function depth to print.  Values >= 1 mean
          information will be printed.  For example, if ``disp=3`` then
          ``disp_print(...)`` will be active for this function and the next
          two nested levels (where applicable).
        - `bool`: Converted to `int`, `True` = 1 and `False` = 0.
    """
    global _DISP_CURRENT_LEVEL, _DISP_MAX_LEVEL

    if disp is not None:
        # New max level declared; start at level one.  Values <= 0 or False
        # will disable display.
        _DISP_MAX_LEVEL = max(int(disp), 0) or None
        _DISP_CURRENT_LEVEL = 1

    else:
        # Move up a display level.
        _DISP_CURRENT_LEVEL += 1


# ----------------------------------------------------------------------

def disp_exit():
    """
    Called when exiting a function scope that uses nested display.  See
    ``disp_print(...)`` for examples.
    """
    global _DISP_CURRENT_LEVEL, _DISP_MAX_LEVEL

    if _DISP_CURRENT_LEVEL <= 1:
        # If exiting a function at level 1, display will be terminated.
        _DISP_MAX_LEVEL = None
        _DISP_CURRENT_LEVEL = 1

    else:
        # Drop back a display level.
        _DISP_CURRENT_LEVEL -= 1


# ----------------------------------------------------------------------

def disp_print(s: str, *args, **kwargs):
    """
    ``disp_print(...)`` is used to print information from multi-level
    nested functions and works in conjunction with ``disp_enter(...)`` and
    ``disp_exit()``. Depending on the current level / function depth the
    output message is either indented or supressed.  `*args` and `**kwargs`
    are identical to the ``print(...)`` statement.

    Examples
    --------
    First define an 'inner' working function:

    >>> def inner_func(disp: int = None):
    ...     disp_enter(disp)  # Setup output at this level.
    ...     # ... does some other things ...
    ...     disp_print("In inner_func()...")
    ...     disp_exit()  # Upon leaving this scope.

    Finally define a top-level function definition that calls the 'inner'
    function.  The default of `None` is used to enable automatic nesting,
    otherwise a display level can be assigned:

    >>> def top_level(disp: int | bool = None):
    ...     disp_enter(disp)  # Setup output at this level.
    ...     # Does some things.
    ...     disp_print("In top_level()...")
    ...     inner_func()  # Default 'disp' argument automatically indents.

    Running `top_level` with default arguments (equivalent to
    ``display=False`` or ``display=0``) produces no output:

    >>> top_level()

    Running the top level function with ``disp=True`` (equivalent to
    ``disp=1``):

    >>> top_level(disp=True)
    In top_level()...

    Running the top level function with ``disp=2`` produces indented output:

    >>> top_level(disp=2) # doctest: +NORMALIZE_WHITESPACE
    In top_level()...
        In inner_func()...
    """
    if (_DISP_MAX_LEVEL is not None and
            _DISP_CURRENT_LEVEL <= _DISP_MAX_LEVEL):
        print('\t' * (_DISP_CURRENT_LEVEL - 1) + s, *args, **kwargs)


# ===========================================================================

def temp_filename(prefix: str = '', suffix: str = '',
                  rand_length: int = None):
    """
    Generates a (nearly unique) temporary file name with given `prefix` and
    `suffix` surrounding a random hex UUID (using library function
    ``uuid.uuid4()``.  The UUID is nominally 32 hex values as chars (i.e.
    128 bit), and this is truncated to `rand_length` if this specified.
    This is useful for interfacing with older DOS and FORTRAN style codes
    which may have specific rules about filename length.
    """
    return prefix + uuid.uuid4().hex[:rand_length] + suffix


# == Profiling Functions ====================================================


# noinspection PyPackageRequirements
def profile_snake(profile_func: Callable):
    """
    This function will profile the function passed as an argument, writing
    the profiling data to a file with the name of the current module
    ``__file__`` and extension changed to ``.prof``.  It then calls
    `snakeviz` to visualise the result which opens a browser view.

    .. note:: The `snakeviz` package is not installed by default as part of
       *PyAvia* and package must be explicitly installed, e.g. ``pip install
       snakeviz``.

    Parameters
    ----------
    profile_func : Callable
        Function to be profiled.
    """
    import cProfile
    import pstats
    # noinspection PyUnresolvedReferences
    import snakeviz.cli as cli
    import os

    with cProfile.Profile() as pr:
        profile_func()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)

    filename = os.path.basename(__file__)
    filename = os.path.splitext(filename)[0] + '.prof'
    stats.dump_stats(filename=filename)
    cli.main([filename])
