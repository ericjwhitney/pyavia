
from collections.abc import Callable, Iterable
from time import time
from typing import Any


# Written by Eric J. Whitney, March 2024.

# ======================================================================


def compare_functions(
        funcs: Iterable[Callable],
        args: tuple | Callable[[], tuple] = None,
        kwargs: dict[str, Any] | Callable[[], dict[str, Any]] = None,
        num_times: int = 1) -> tuple[list[list[Any]], list[float]]:
    """
    Compare the results and execution time of two functions.

    Parameters
    ----------
    funcs : Iterable[Callable[[args, kwargs], Any]]
        One or more functions to compare.

    args : tuple or Callable[[], tuple]), optional
        Identical positional arguments to pass to all functions.  See
        `Notes` if this is a `callable`.

    kwargs : dict[str, Any] or Callable[[], dict[str, Any]]), optional
        Identical keyword arguments to pass to all functions. See
        `Notes` if this is a `callable`.

    num_times:  int, default = 1
        The number of times to run each function.

    Returns
    -------
    results : list[list[...]]
        The results for each function as a list.  Each entry is also a
        list of length `num_times` containing the results for each run
        of the function.

    times : list[float]
        The total execution times of `each function..

    Notes
    -----
    If a callable is provided for `args` or `kwargs`, it will be called
    `num_times` in advance to generate a list of identical inputs for
    all functions.
    """
    if not isinstance(funcs, Iterable):
        raise ValueError("First argument must be an iterable of "
                         "functions.")

    # Generate lists of arguments in advance.  This allows for random
    # generators, etc, and ensures both functions have the same inputs.

    # Generate 'args' for each call.
    if args is None:
        args = ()

    arg_list = []
    if callable(args):
        for i in range(num_times):
            arg_list.append(args())  # noqa
    else:
        arg_list = [args] * num_times

    # Generate 'kwargs' for each call.
    if kwargs is None:
        kwargs = {}

    kwarg_list = []
    if callable(kwargs):
        for i in range(num_times):
            kwarg_list.append(kwargs())  # noqa
    else:
        kwarg_list = [kwargs] * num_times

    # Run each function in turn.  Call time() twice every pass to
    # eliminate any overhead due to append, etc.
    results, times = [], []
    for func in funcs:
        func_results = []
        t_start = time()
        for args_i, kwargs_i in zip(arg_list, kwarg_list):
            func_results.append(func(*args_i, **kwargs_i))

        times.append(time() - t_start)
        results.append(func_results)

    return results, times
