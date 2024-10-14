"""
TODO
"""

from typing import Callable

# Written by Eric J. Whitney, May 2022.

# ======================================================================


def profile_snake(profile_func: Callable):
    """
    This function will profile the function passed as an argument,
    writing the profiling data to a file with the name of the current
    module ``__file__`` and extension changed to ``.prof``.  It then
    calls `snakeviz` to visualise the result which opens a browser view.

    .. note:: The `snakeviz` package is not installed by default as part
       of the base *PyAvia* package.  It can be installed using either
       ``pip install pyavia[profile]`` or ``pip install pyavia[all]`` or
       directly e.g. ``pip install snakeviz``.

    Parameters
    ----------
    profile_func : Callable
        Function to be profiled.
    """
    import cProfile
    import pstats
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
