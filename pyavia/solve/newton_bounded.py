"""
Find a zero of a real or complex function using the Newton-Raphson
method, with the following additional features:

    - All trial points generated will lie within the given bounded
      interval.  This may help in the situation where the given function
      is not well defined outside the given interval.
    - If the secant method is being used (a derivative is not provided, i.e.
      `fprime=None`) and no second initial trial point is given (`x1 =
      None`) then `x1` is computed by placing it nearer to the middle of the
      bounding interval than `x0`.

This is a modification of SciPy's ``newton()`` at
https://github.com/scipy/scipy/blob/master/scipy/optimize/_zeros_py.py
"""

# Written by: Eric J. Whitney  Last updated: 21 December 2021

import operator
import warnings
from collections import namedtuple
from collections.abc import Sequence
from numbers import Number
from typing import Union, Callable, TypeVar

# noinspection PyProtectedMember
from scipy.optimize.zeros import (_results_select, _ECONVERGED, _ECONVERR)
import numpy as np


_SclOrVec = TypeVar('_SclOrVec', Number, Sequence[Number])


# ---------------------------------------------------------------------------

# TODO:  The rule about bounds not containing infinity can be relaxed if we
#  offer a starting point option.  Also see _array_newton_bounded.

def newton_bounded(func: Callable, x0: _SclOrVec, *,
                   fprime: Union[bool, Callable] = None,
                   args=(), tol: float = 1.48e-8,
                   maxiter: int = 50, fprime2: Union[bool, Callable] = None,
                   x1: _SclOrVec = None, rtol: float = 0.0,
                   full_output: bool = False, disp: bool = True,
                   bounds: (_SclOrVec, _SclOrVec)):
    """
    Find a zero of a real or complex function using the Newton-Raphson (or
    secant or Halley's) method, with the additional requirement that all
    trial points are generated to lie within the given bounded interval.
    This may help in the situation where the given function is not well
    defined outside the given interval.

    Parameters
    ----------
    x1 : float, optional
        Another estimate of the zero that should be somewhere near the
        actual zero, this is used if `fprime` is not provided.  If `x1` is
        also not provided, its position is computed by placing it nearer to
        the middle of the bounding interval than `x0`.

        .. note:: For vector-valued functions, `x1` is not provided and is
           always computed automatically.

    bounds : (array-like, array-like)
        if n > 1 vectors are tkaen to form opposite corners of a bounding
        hypercube.  Order is not important, bounds are always rearranged into
        min -> max order.  Bounds cannot contain infinity when the secant
        method is used, because they are used to determine a starting point.

    func, x0, fprime, args, tol, maxiter, fprime2, rtol, full_output, disp :
        Only parameters with different behaviour from SciPy `newton()` and
        `root_scalar()` are covered here.  For other parameters, refer to
        SciPy documentation.

    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r, converged, zero_der :
        Additional information depending on the argument `full_output` and
        whether `x0` is a scalar or array.  See SciPy `newton()` for more
        details.
    """
    # Note:  Sections that are modified from the refence SciPy
    # implementation are marked **.

    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)

    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    if np.size(x0) > 1:
        return _array_newton_bounded(func, x0, fprime=fprime, args=args,
                                     tol=tol, maxiter=maxiter,
                                     fprime2=fprime2, full_output=full_output,
                                     bounds=bounds)

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    p_min, p_max = 1.0 * min(bounds), 1.0 * max(bounds)  # ** Setup bounds.
    funcalls = 0

    # ** Check initial point.
    if not (p_min < p0 < p_max):
        raise ValueError("Starting point cannot be on or outside boundary.")

    if fprime is not None:
        # Newton-Raphson method.

        for itr in range(maxiter):
            # Evaluate f(p0).
            fval = func(p0, *args)
            funcalls += 1

            # If fval is 0, a root has been found; terminate.
            if fval == 0:
                return _results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED))

            fder = fprime(p0, *args)
            funcalls += 1

            if fder == 0:
                # Reached a level state -> df/dp = 0.
                msg = "Derivative was zero."
                if disp:
                    msg += (" Failed to converge after %d iterations, "
                            "value is %s." % (itr + 1, p0))
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)
                return _results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR))

            # Compute step using derivative.
            newton_step = fval / fder

            if fprime2:
                # Halley's method.  Adjust the Newton step:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                #
                # Only do it if denominator stays close enough to 1.
                # Rationale: If 1-adj < 0, then Halley's method sends x in
                # the opposite direction to Newton. This doesn't happen if x
                # is close enough to the root.

                fder2 = fprime2(p0, *args)
                funcalls += 1

                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj

            # Compute new point position using step.
            p = p0 - newton_step

            # ** Enforce boundaries (both Newton and Halley's method).
            p_clip = np.clip(p, p_min, p_max)
            hit_boundary = (p_clip - p).any()
            p = p_clip

            # ** Only check for convergence if the last step was not
            # clipped to a boundary.
            if np.isclose(p, p0, rtol=rtol, atol=tol) and not hit_boundary:
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0 = p

    else:
        # Secant method.
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1

        else:
            # ** Determine in which direction to place p1; move towards the
            # centre of the bounds.  Also make the offset a fraction of the
            # problem size.
            p_mid = 0.5 * (p_max + p_min)
            p_scl = p_max - p_mid
            dirn = -1 if p0 > p_mid else +1
            p1 = p0 + 1e-2 * dirn * p_scl

        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0

        for itr in range(maxiter):

            if q1 == q0:
                # Reached a level state: f(p0) = f(p1) -> df/dp = 0.
                if p1 != p0:
                    # Warn if collapsed to a single point.
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += (" Failed to converge after %d iterations, "
                                "value is %s." % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning)

                # Return the midpoint.
                p = (p1 + p0) / 2.0
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))

            else:
                # Compute new point position using secant.
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)

            # ** Enforce boundaries.
            p_clip = np.clip(p, p_min, p_max)
            hit_boundary = (p_clip - p).any()
            p = p_clip

            # ** Only check for convergence if the last step was not
            # clipped to a boundary.
            if np.isclose(p, p1, rtol=rtol, atol=tol) and not hit_boundary:
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))

            # ** Update working points and continue.  For secant method,
            # do not replace the old point if the current step took us to a
            # boundary (re-use that data).
            if not hit_boundary:
                p0, q0 = p1, q1

            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        # noinspection PyUnboundLocalVariable
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    # noinspection PyUnboundLocalVariable
    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR))


# ---------------------------------------------------------------------------

def _array_newton_bounded(func: Callable, x0: Sequence[Number], *,
                          fprime: Union[bool, Callable, None], args,
                          tol: float, maxiter: int,
                          fprime2: Union[bool, Callable, None],
                          full_output: bool,
                          bounds: (Sequence[Number], Sequence[Number])):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays,
    adding a bounds limitation.  This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``.
    """
    # Note:  Sections that are modified from the refence SciPy
    # implementation are marked **.

    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = 1.0 * np.array(x0, copy=True)  # x 1.0 raises to at least float.

    # ** Check bounds array sizes.
    if (np.size(bounds[0]) != np.size(x0) or
            np.size(bounds[1]) != np.size(x0)):
        raise ValueError("Bounds lengths must match x0.")

    # ** Rearrange bounds to min -> max.
    p_min, p_max = np.empty_like(p), np.empty_like(p)
    for i, (p_b1, p_b2) in enumerate(zip(*bounds)):
        p_min[i], p_max[i] = (p_b1, p_b2) if p_b1 < p_b2 else (p_b2, p_b1)

    # ** Check initial point.
    if not ((p_min < p) & (p < p_max)).all():
        raise ValueError("Starting point cannot be on or outside boundary.")

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)

    if fprime is not None:
        # Newton-Raphson method.

        for iteration in range(maxiter):
            # Evaluate f(p0).
            fval = np.asarray(func(p, *args))

            # If all fval are 0, all roots have been found - terminate.
            if not fval.any():
                failures = fval.astype(bool)
                break

            fder = np.asarray(fprime(p, *args))
            nz_der = (fder != 0)

            # Stop iterating if all derivatives are zero.
            if not nz_der.any():
                break

            # Newton step.
            dp = fval[nz_der] / fder[nz_der]

            if fprime2 is not None:
                # Halley's method, adjust step.
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])

            # Only update components where there are nonzero derivatives.
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp

            # ** Enforce boundaries (both Newton and Halley's method).
            p = np.clip(p, p_min, p_max)

            # Find components that are not yet converged.
            failures[nz_der] = np.abs(dp) >= tol

            # Stop iterating if there aren't any failures, not including zero
            # derivatives.
            if not failures[nz_der].any():
                break

    else:
        # Secant method.

        # ** Determine in which direction to place p1; move towards the
        # centre of the bounds.  Also make the offset a fraction of the
        # problem size.
        p_mid = 0.5 * (p_max + p_min)
        p_scl = p_max - p_mid
        dirn = np.where(p_mid - p < 0, -1, +1)  # For zero, default is +ve.
        p1 = p + 1e-2 * dirn * p_scl

        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)

        # ** Added check on f(x) output size.
        if q0.shape != p.shape:
            raise ValueError(f"Wrong shape output from f(x): Expected "
                             f"{p.shape} got {q0.shape}.")

        for iteration in range(maxiter):
            nz_der = (q1 != q0)

            if not nz_der.any():
                # Stop iterating if all derivatives are zero.
                p = (p1 + p) / 2.0
                break

            # Secant step.
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]

            # Only update components where there are nonzero derivatives.
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp

            # ** Enforce boundaries.
            p = np.clip(p, p_min, p_max)

            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # Don't assign zero derivatives again.
            failures[nz_der] = np.abs(dp) >= tol  # Not yet converged.

            # Stop iterating if there aren't any failures, not including zero
            # derivatives.
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args))

    zero_der = ~nz_der & failures  # Don't include converged with zero-derivs.

    if zero_der.any():

        if fprime is None:
            # Secant warnings.
            # noinspection PyUnboundLocalVariable
            nonzero_dp = (p1 != p)
            # Non-zero dp, but infinite Newton step.
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2))
                warnings.warn(
                    'RMS of {:g} reached'.format(rms), RuntimeWarning)

        else:
            # Newton or Halley warnings.
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = '{:s} derivatives were zero'.format(all_or_some)
            warnings.warn(msg, RuntimeWarning)

    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = '{0:s} failed to converge after {1:d} iterations'.format(
            all_or_some, maxiter)
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)

    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)

    return p


# ---------------------------------------------------------------------------

# Short test function - to be moved.
if __name__ == '__main__':
    def f(x):
        return [3 * x[0] ** 2 + x[0] - 2, x[1]]

    def f_der(x):
        return [6 * x[0] + 1, 1]

    # noinspection PyUnusedLocal
    def f_der2(x):
        return [6, 0]

    sol = newton_bounded(f, [-.999, 0], bounds=[[-1, +15], [-1, +1.5]],
                         fprime=f_der, fprime2=f_der2,
                         disp=True, full_output=True)

    print()
