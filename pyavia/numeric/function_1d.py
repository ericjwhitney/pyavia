"""
1-D Functions (:mod:`pyavia.numeric.function_1d`)
=================================================

.. currentmodule:: pyavia.numeric.function_1d

Classes for working with 1-D / scalar functions.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union, Final

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from pyavia.numeric.math_ext import (sclvec_asarray, sclvec_return,
                                     within_range)
from pyavia.numeric.solve import SolverError

# Written by Eric J. Whitney, May 2023.

_ExtOpt = Union[None, str, float, 'Function1D']  # Extrapolation options.


# TODO Add roots() as solve(0, ...) for compatibility with SciPy.  Move
#  differentiation / anti-differentiation to __call__.

# TODO replace sclvec_asarray with check/make_array, return_array, etc.

# ======================================================================

class Function1D(ABC):
    """
    Abstract definition of a 1-D scalar function
    :math:`y = f_{model}(x)`.

    Notes
    -----
    Only real / float types are supported at this stage for `x` and `y`
    values, as well as for solutions / roots.
    """

    def __init__(self, x_domain: tuple[float, float],
                 ext_lo: _ExtOpt = None, ext_hi: _ExtOpt = None):
        r"""
        Parameters
        ----------
        x_domain : tuple[float, float]

            Applicable closed interval (range) of the function and its
            derivatives, i.e. :math:`x \in [x_{ min}, x_{max}]`.  If an
            `x` value is supplied outside this range, how it is handled
            depends on ``ext_lo`` and ``ext_hi`` (see below).

        ext_lo, ext_hi : various, default = None

            Extend / extrapolate function outside `x_domain` on either
            the ``lo`` (or left ``x < x_domain[0]``) side or ``hi`` (or
            right ``x > x_domain[1]``) side.  The type of extension
            depends on the argument:

            - `None`: Attempting to compute a value outside the range
              raises a `ValueError` exception.

            - ``'constant'``:  Returns the  `y` value corresponding to
              the function endpoint, i.e. :math:`f_{model}(x_{min})` or
              :math:`f_{model}(x_{max})`.

            - ``'linear'``: Extend a straight line from the adjacent
              range endpoint, using the function value and derivative
              computed at the adjacent endpoint.

            - `value`: Similar to ``'constant'`` however the value
              given is used as the constant.  This is akin to a `fill
              value`.

            - `Function1D`: Use the given `Function1D` object for
              extension / extrapolation.  In this way, functions can
              be chained together as desired.
        """
        # -- Setup Base Functions --------------------------------------

        if (not isinstance(x_domain, Sequence) or len(x_domain) != 2 or
                x_domain[0] > x_domain[1] or np.isnan(x_domain).any()):
            raise TypeError("'x_domain' invalid.")

        self._x_domain = tuple(x_domain)
        self._setup_func()  # Required before extrapolation setup.

        # -- Setup Extrapolations --------------------------------------

        for side in (0, 1):
            # Get argument, endpoint and establish 'exterior' half-
            # domain ('constant', 'value' and 'linear' options all
            # extend to infinity).
            if side == 0:
                ext_arg = ext_lo
                x_end = self._x_domain[0]
                ext_domain = (-np.inf, x_end)
            else:
                ext_arg = ext_hi
                x_end = self._x_domain[1]
                ext_domain = (x_end, +np.inf)

            # -- Extrapolation Options ---------------------------------

            if ext_arg is None:
                ext_func = None

            elif isinstance(ext_arg, str):
                # Convert string extrapolation arguments to functions.
                ext_arg = ext_arg.lower()  # To lowercase.

                # Calculate endpoint 'y' and set slope.
                y_end = self._func(np.asarray(x_end), n=0).flat[0]
                if ext_arg == 'constant':
                    end_gradient = 0.0

                elif ext_arg == 'linear':
                    end_gradient = self._func(np.asarray(x_end), n=1).flat[0]

                else:
                    raise ValueError(f"Unknown extrapoation type '{ext_arg}'.")

                ext_func = Line1D(x_end, y_end, slope=end_gradient,
                                  x_domain=ext_domain)

            elif isinstance(ext_arg, (int, float)):
                # 'Fill' value, similar to 'constant'.
                ext_func = Line1D(x_end, ext_arg, slope=0.0,
                                  x_domain=ext_domain)

            elif isinstance(ext_arg, Function1D):
                # User defined extrapolation or chained-up function.
                ext_func = ext_arg

            else:
                raise TypeError(f"Unknown extrapolation argument: "
                                f"{repr(ext_arg)}")

            # -- Finalise ----------------------------------------------

            if side == 0:
                self._ext_lo = ext_func
            else:
                self._ext_hi = ext_func

    def __call__(self, x: ArrayLike) -> ArrayLike:
        r"""
        Return the value of the approximating function
        :math:`y_{approx}` at the given abscissa/s `x`.

        Parameters
        ----------
        x : array_like
            `x` value/s.

        Returns
        -------
        f : array_like
            Value/s of the approximating function f(`x`).

        Raises
        ------
        ValueError
            If `x` is outside `x_domain` and there is no extension /
            extrapolation.

        Notes
        -----
        - The model function is assumed to apply on the closed interval
          that includes the endpoints i.e. :math:`x \in [x_{min},
          x_{max}]`.  If the function is discontinuous at the
          endpoint/s, the endpoint value is taken from this model
          function (i.e. the interior).
        - Any `x` values of `NaN` automatically return a corresponding
          `NaN`.
        """
        return self._eval(x, n=0)

    # -- Public Methods ------------------------------------------------

    def derivative(self, x: ArrayLike, n: int = 1) -> ArrayLike:
        """
        Return the derivative of the model function
        :math:`d^{n}f_{model}/dx^n` at the given abscissa/s `x` where
        `n` is the order.

        Parameters
        ----------
        x : array_like
            `x` value (or sequence of values).
        n : int, default = 1
            Order of derivative to evaluate, where `n` >= 1. For
            example, if ``n=1`` the first derivative `dy/dx` is
            returned.

        Returns
        -------
        array_like
            Value/s of the derivative :math:`df_{model}/dx`
            corresponding to `x`.

        Raises
        ------
        ValueError
            If `x` out of bounds and `extrapolate` is `None`.

        Notes
        -----
        - Behaviour when `x` corresponds to an endpoint or `NaN` is the
          same as `__call__`.
        """
        if n < 1:
            raise ValueError("Derivative requires n >= 1.")

        # TODO also check n is integer

        return self._eval(x, n)

    @property
    def ext_hi(self) -> None | Function1D:
        """
        Return the function used to extrapolate beyond the high end of
        the domain (`x_max`), or `None`.
        """
        return self._ext_hi

    @property
    def ext_lo(self) -> None | Function1D:
        """
        Return the function used to extrapolate beyond the low end of
        the domain (`x_min`), or `None`.
        """
        return self._ext_lo

    def solve(self, y: float = 0.0,
              x_range: (float | None, float | None) = (None, None)
              ) -> [float]:
        """
        Find one (or more) real `x` values that satisfy the equation
        :math:`y = f_{model}(x)` in the given `x_range = [min, max]`.

        Parameters
        ----------
        y : float, default = 0.0
            Target `y` value of the equation.

        x_range : tuple[float | None, float | None], default = (None, None)
            Closed interval `x_range = [min, max]` to search for
            solutions:

            - `x_range` may extend beyond the function domain (even to
              (-∞, +∞)) depending on function extrapolation available:

              * If `ext_lo` or `ext_hi` is defined on the applicable
                side, the extension / extrapolation function is also
                interrogated for additional solutions in the extended
                region.

              * If `ext_lo` or `ext_hi` is `None` on the applicable
                side, `ValueError` is raised.

            - If either side of `x_range` is `None` then that side is
              set to the current function extents (which may extend
              beyond the domain, depending on extrapolation options).

        Returns
        -------
        x : [float]
            A sorted list containing one or more unique solutions (real
            roots) to the equation :math:`y = f_{model}(x)`. If no
            solutions are found an empty list is returned.

        Notes
        -----
        Whether or not discontinuities or gaps are reported as proper
        solutions is up to each specific implementation / derived class.
        """
        if not isinstance(x_range, Sequence) or len(x_range) != 2:
            raise TypeError("Invalid x_range.")

        # Substitute any x_range 'None' values with the x_extents value.
        x_range = [x_r if x_r is not None else x_d
                   for x_r, x_d in zip(x_range, self.x_extents)]

        # Get unique 'interior' solutions from within the current domain.
        x_range_intl = np.clip(x_range, *self._x_domain)
        x_sols = set(self._solve(y, x_range_intl))

        # Add 'exterior' unique solutions from LHS outside the normal
        # domain.
        if x_range[0] < self._x_domain[0]:
            if self._ext_lo is None:
                raise ValueError("x_range extends beyond function "
                                 "domain on LH (low) side.")
            else:
                x_range_lo = (x_range[0], self._x_domain[0])  # Shrunk.
                x_sols.update(self._ext_lo.solve(y, x_range_lo))

        # Add RH 'exterior' unique solutions from outside the normal domain.
        if x_range[1] > self._x_domain[1]:
            if self._ext_hi is None:
                raise ValueError("x_range extends beyond function "
                                 "domain on RH (high) side.")
            else:
                x_range_hi = (self._x_domain[1], x_range[1])  # Shrunk.
                x_sols.update(self._ext_hi.solve(y, x_range_hi))

        # Return sorted list of unique points.
        return list(sorted(x_sols))

    @property
    def x_domain(self) -> (float, float):
        """
        Returns the `x` domain of the function as (`min`, `max`).  This
        does not including any left or right extension / extrapolation
        regions.
        """
        return self._x_domain

    @property
    def x_extents(self) -> (float, float):
        """
        Return the complete `x` extents where function can be evaluated.
        This may be larger than `x_domain` as it includes the extents of
        any low and/or high extrapolation regions (possibly up to ±∞).
        """
        if self._ext_lo is None:
            extents_min = self._x_domain[0]
        else:
            extents_min = self._ext_lo.x_extents[0]

        if self._ext_hi is None:
            extents_max = self._x_domain[1]
        else:
            extents_max = self._ext_hi.x_extents[1]

        return extents_min, extents_max

    # -- Private Methods -----------------------------------------------

    def _eval(self, x: ArrayLike, n: int) -> ArrayLike:
        """
        Evaluates the model function (n == 0) or it's derivative
        (n > 0) at the given abscissa/s `x`.  This is common code used
        by `__call__` and `derivative`.  Note:

        - Any `NaN` `x` values automatically generate a `NaN` result.
        - `x` values are passed to extended / extrapolated regions using
          `_eval` not `_func`, as these may need to further delegate calls
          to other extensions in turn.
        """
        assert n >= 0
        x, scalar = sclvec_asarray(x)
        res = np.full_like(x, np.nan)  # Setup results, default NaN.

        # Handle NaN, interior and exterior points separately.
        valid = ~np.isnan(x)
        lo = (x < self._x_domain[0]) & valid
        hi = (x > self._x_domain[1]) & valid
        interior = ~lo & ~hi & valid

        # Compute interior points.
        res[interior] = self._func(x[interior], n)

        # Compute LHS extrapolation.
        if self._ext_lo is not None:
            res[lo] = self._ext_lo._eval(x[lo], n)

        elif lo.any():
            raise ValueError(f"Extrapolation for x < x_domain not defined.")

        # Compute RHS extrapolation.
        if self._ext_hi is not None:
            res[hi] = self._ext_hi._eval(x[hi], n)

        elif hi.any():
            raise ValueError(f"Extrapolation for x > x_domain not defined.")

        return sclvec_return(res, scalar)

    @abstractmethod
    def _func(self, x: NDArray, n: int) -> NDArray:
        """
        The internal implementation of the model function and derivative -
        derived classes must provided this method.

        Parameters
        ----------
        x : ndarray, shape (n,)
            1-D array of normal (non-`NaN`) values within the function
            domain.  Note: `NaN` values do not need to be handled as
            they are covered separately in `_eval`).

        Returns
        -------
        y : numpy.ndarray, shape (n,)
            1-D array of function results `y = f(x)` corresponding to
            `x`.
        """
        raise NotImplementedError

    def _setup_func(self):
        """
        Derived classes can override this function to perform any
        initial setup required on the model function/s.  This is called
        during `__init__` after any initial actions, but *prior* to
        setting up extrapolations - because extrapolations may need to
        call the model function itself.
        """
        pass

    @abstractmethod
    def _solve(self, y: float, x_range: (float, float)) -> [float]:
        """
        Return one (or more) real `x` values that satisfy the model
        function equation :math:`y = f_{model}(x)`.  These must lie
        only in the given closed interval x_range = [min, max], which
        may be equal in size to or smaller than the current `x_domain`.
        """
        raise NotImplementedError


# ======================================================================


class FitXY1D(Function1D, ABC):
    """
    Abstract class incorporating standard methods for functions that are
    fit to `x`-`y` data.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike,
                 x_domain: tuple[float | None, float | None] = (None, None),
                 **kwargs):
        r"""
        Parameters
        ----------
        x, y : array_like, shape (n,)
            (`x`, `y`) values of known points where `n` >= 2.  Duplicate
            `x` values are not permitted.  These values are copied for
            internal storage.

        x_domain : tuple[float | None, float | None], default =
                   (None, None)
            Applicable closed interval (range) of the function and its
            derivatives, i.e. :math:`x \in [x_{min}, x_{max}]`.  If
            either entry is `None`, then the domain limit on that side
            is taken to encompass just the given `x` values (i.e. either
            the minimum or maximum).

        kwargs :
            See `Function1D.__init__` for additional arguments (e.g.
            extrapolation)
        """
        # -- Setup and Sort Points -------------------------------------

        self._x = np.array(x, copy=True, ndmin=1)
        self._y = np.array(y, copy=True, ndmin=1)
        if self._x.ndim != 1 or self._y.ndim != 1:
            raise ValueError("'x' and 'y' must be 1-D / array like.")
        if self._x.size != self._y.size:
            raise ValueError("'x' and 'y' arrays must be the same size.")
        if self._x.size == 0:
            raise ValueError("'x', 'y' empty.")

        sort_idx = np.argsort(self._x)
        self._x = self._x[sort_idx]
        self._y = self._y[sort_idx]

        # -- Create Domain ---------------------------------------------

        # Check x_domain has two elements in advance.
        if not isinstance(x_domain, Sequence) or len(x_domain) != 2:
            raise TypeError("Invalid x_domain.")

        # Substitute `None` domain values with the points range.
        x_pts = (np.min(self._x), np.max(self._x))
        x_domain = [x_d if x_d is not None else x_p
                    for x_d, x_p in zip(x_domain, x_pts)]

        # -- Init Base -------------------------------------------------

        super().__init__(x_domain, **kwargs)

    # -- Public Methods ------------------------------------------------

    @property
    def n_pts(self) -> int:
        """
        Number of fit points (`x`, `y`) stored for this function.
        """
        return self._x.size

    @property
    def R2(self) -> float:
        """
        Coefficient of determination :math:`R^2 = 1 - SS_{res} /
        SS_{tot}`.  This is calculated unless overridden by a derived
        class.
        """
        return 1.0 - self.sumsq_residual / self.sumsq_total

    @property
    def sumsq_residual(self) -> float:
        r"""
        Residual sum of squares :math:`SS_{res} = \sum{(y_i -
        f_{model}(x_i))^2}`.  This is always re-calculated unless
        overridden by a derived class.
        """
        return float(np.sum((self._y - self._func(self._x, n=0)) ** 2))

    @property
    def sumsq_total(self) -> float:
        r"""
        Total sum of squares :math:`SS_{tot} = \sum{(y_i - \bar{y})^2}`.
        This is always calculated unless overridden by a derived class.
        """
        mean = np.sum(self._y) / len(self._y)
        return float(np.sum((self._y - mean) ** 2))

    @property
    def x(self) -> NDArray:
        """
        Absiccas `x` of known function values, in sorted order.
        """
        return self._x

    @property
    def y(self) -> NDArray:
        """
        Ordinates 'y' of known function values, corresponding to
        abscissas returned by property `x`.
        """
        return self._y

    # -- Private Methods -----------------------------------------------


# ======================================================================

class Line1D(Function1D):
    """
    A 1-D function modelled as a straight line.  The line may be finite
    or infinite; see `__init__` for more details.

    Notes
    -----
    For a horizontal line (`m` = 0) the `solve` method will return no
    intersections even in the case where `y` equals the intercept (which
    would imply all `x` values are solutions).
    """

    def __init__(self, x: ArrayLike = None, y: ArrayLike = None,
                 x_domain: (float, float) = (-np.inf, +np.inf), *,
                 slope: float = None, intercept: float = None, **kwargs):
        """
        Possible argument combinations are as follows:

        - If two `x` values and two `y` values are given corresponding
          to two points, this defines the line.
        - If a single `x` and `y` value are provided, then `slope` or
          `intercept` must also be provided to define the line.
        - If no `x` and `y` value is provided, then both `slope` and
          `intercept` must be provided.

        Parameters
        ----------
        x, y : array_like
            Refer to text for argument combinations.
        x_domain : tuple[float, float] or None, default = (-∞, +∞)
            Applicable closed interval (domain) for the line.
        slope, intercept : float
            Refer to text for argument combinations.
        kwargs :
            See `Function1D.__init__` for additional arguments (e.g.
            extrapolation)
        """
        super().__init__(x_domain=x_domain, **kwargs)

        # Standardise x, y points.
        x = np.atleast_1d(x) if x is not None else np.array([])
        y = np.atleast_1d(y) if y is not None else np.array([])
        if x.shape != y.shape:
            raise ValueError("'x' and 'y' arrays must have the same shape.")

        # Construct line based on combination of arguments provided.
        if x.size > 2:
            raise ValueError("Only one or two points can be provided.")

        elif x.size == 2:
            # Two points provided.
            if slope is not None or intercept is not None:
                raise ValueError("Two points provided, slope and "
                                 "intercept must be None.")

            # Line defined by two points.
            if x[1] == x[0]:
                raise ValueError("Vertical lines not permitted.")

            self._line_m = (y[1] - y[0]) / (x[1] - x[0])
            self._line_c = y[0] - self._line_m * x[0]

        elif x.size == 1:
            # Single point provided along with with slope or intercept.
            if slope is not None and intercept is not None:
                raise ValueError("One point provided, either slope or "
                                 "intercept must be None.")

            if slope is not None:
                # Line defined by one point and slope.
                self._line_m = slope
                self._line_c = y[0] - self._line_m * x[0]

            else:
                # Line defined by one point and intercept.
                if x[0] == 0:
                    raise ValueError("Intercept definition requires "
                                     "point not on y-axis.")
                self._line_c = intercept
                self._line_m = (y[0] - self._line_c) / x[0]

        else:
            if slope is None or intercept is None:
                raise TypeError("No points provided, both slope and "
                                "intercept are required.")

            self._line_m = slope
            self._line_c = intercept

        # Check line equation values.
        if np.isnan(self._line_m) or np.isnan(self._line_c):
            raise ValueError("Resulting line is invalid.")

    # -- Private Methods -----------------------------------------------

    def _func(self, x: NDArray, n: int) -> NDArray:
        assert n >= 0
        if n == 0:
            # Basic line equation.
            return self._line_m * x + self._line_c

        elif n == 1:
            # First derivative (constant).
            return np.full_like(x, self._line_m)

        else:
            # Higher derivatives (zero).
            return np.zeros_like(x)

    def _solve(self, y: float, x_range: (float, float)) -> [float]:
        # Horizontal lines give no solutions.
        if self._line_m == 0.0:
            return []

        # Find the single solution, only return if within the range.
        x_sol = (y - self._line_c) / self._line_m
        if x_range[0] <= x_sol <= x_range[1]:
            return [x_sol]
        else:
            return []

# ======================================================================


class PCHIP1D(FitXY1D):
    r"""
    A 1D function model using a Piecewise Cubic Hermite Interpolating 
    Polynomial (`PCHIP`).  
    
    This interpolation passes through all given (`x`, `y`) points exactly, 
    i.e. ``sumsq_residal == 0`` and `R`² = 1. See [1]_ for more detail.

    .. note:: For ``solve(...)`` `PCHIPApprox1D` does not include
       discontinuity jumps across `y` as possible solutions.

    Notes
    -----
    .. [1] PCHIP Interpolation using SciPy:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    """  # noqa

    # -- Public Methods ------------------------------------------------------

    @property
    def R2(self) -> float:
        return 1.0  # Always passes thru all points.

    @property
    def sumsq_residual(self) -> float:
        return 0.0  # Always passes thru all points.

    # -- Private Methods -----------------------------------------------

    def _func(self, x: NDArray, n: int) -> NDArray:
        assert n >= 0
        if n <= 3:
            # Up to third derivative.
            return self._pchip_funcs[n](x)

        else:
            # Higher derivatives are zero.
            return np.zeros_like(x)

    def _setup_func(self):
        # Setup base PCHIP function (n = 0) and three derivatives,
        # before extrapolations area added.  Note: Extrapolate is always
        # 'True' for the internal PCHIP objects as the domain is handled
        # separately.
        self._pchip_funcs = [PchipInterpolator(self.x, self.y,
                                               extrapolate=True)]
        for n in (1, 2, 3):
            self._pchip_funcs.append(self._pchip_funcs[0].derivative(n))

    def _solve(self, y: float, x_range: (float, float)) -> [float]:
        # Use PCHIP 'solve', then trim to given range.
        x_sols = self._pchip_funcs[0].solve(y=y, discontinuity=False)
        return within_range(x_sols, x_range)

# ======================================================================


class SmoothPoly1D(FitXY1D):
    """
    A 1-D polynomial of specified degree/s that smoothly fits the data
    points in a least-squares sense.
    """
    # Set a maximum on how many polys will be precomputed (base + derivatives).
    # The remainder are computed on demand.
    _PRECOMPUTE_N_POLYS: Final = 3

    def __init__(self, x: ArrayLike, y: ArrayLike,
                 x_domain: (float | None, float | None) = (None, None),
                 *, degree: int | [int],
                 R2_min: float = None, **kwargs):
        """
        Parameters
        ----------
        x, y, x_domain : array_like, shape (n,)
            See `FitXY1D.__init__` for more details.
        degree : int or [int]
            One or more positive integers (>=0) specifying the degree of
            the smoothed polynomial:

            - If one value is given, this is the (fixed) degree of the
              fitting polynomial.  Parameter `R2_min` is ignored.

            - If a sequence of values is given, an attempt is made to
              build the polynomial using each degree in turn, in the
              order given.  A given degree is rejected if it cannot be
              constructed (i.e. insufficient / invalid points).  In
              addition, if the parameter `R2_min` is provided, the
              polynomial is rejected if it is not sufficiently accurate,
              i.e. :math:`R^2 < R^2_{min}`.

              .. note:: This (re-)building can occur whenever the stored
                 `x` and `y` values change (e.g. after
                 ``add_points(...)``), which means that if a sequence of
                 degrees is provided the order of the polynomial can be
                 dynamic.

        kwargs :
            See `Function1D.__init__` for additional arguments (e.g.
            extrapolation)
        """
        # Setup polynomial degree/s.
        self._poly_funcs: list[Polynomial] = []

        if not isinstance(degree, Sequence):
            degree = [degree]
        self._degrees = list(degree)
        if any(d < 0 for d in self._degrees):
            raise ValueError(f"Invalid polynomial degree.")

        # Setup error criteria; ignore for single / non-sequence degree.
        self._R2_min = R2_min
        if len(self._degrees) == 1:
            self._R2_min = None

        if self._R2_min is not None and not (0.0 < self._R2_min <= 1.0):
            raise ValueError(f"Requires 0 < R2_min <= 1.")

        self._sumsq_resid, self._sumsq_total = np.nan, np.nan

        # Remainder of base can now be setup.
        super().__init__(x, y, x_domain, **kwargs)

    # -- Public Methods ------------------------------------------------

    @property
    def sumsq_residual(self) -> float:
        return self._sumsq_resid  # Precomputed value.

    @property
    def sumsq_total(self) -> float:
        return self._sumsq_total  # Precomputed value.

    # -- Private Methods -----------------------------------------------

    def _func(self, x: NDArray, n: int) -> NDArray:
        assert n >= 0
        assert len(self._poly_funcs) > 0

        if n < len(self._poly_funcs):
            # Polynomial of this order is available.
            return self._poly_funcs[n](x)

        else:
            # Compute polynomial for this order on the fly.
            poly_n = self._poly_funcs[0].deriv(n)
            return poly_n(x)
        
    def _setup_func(self):
        # Try each of the available degrees in turn.
        for deg in self._degrees:
            # Skip any degrees with insufficient number of points.
            if self.n_pts < deg + 1:
                continue

            # Build and assign the base polynomial - this allows us to
            # check the error measures.
            poly_0, (self._sumsq_resid, *_) = Polynomial.fit(
                self.x, self.y, deg, full=True)
            self._poly_funcs = [poly_0]
            self._sumsq_total = super().sumsq_total

            # If there is no error criteria, or if the accuracy is
            # sufficient, we're done.
            if self._R2_min is None or self.R2 >= self._R2_min:
                break

        else:
            # We exhausted the available options, bail out.
            self._poly_funcs = []  # Reset.
            self._sumsq_resid, self._sumsq_total = np.nan, np.nan
            raise SolverError(
                "Failed to construct smooth polynomial.",
                details="Could not construct polynomial of any degree "
                        "with the required accuracy.")

        # Get the max number of polynomials to precompute, then
        # precompute n = (1, ...) as derivatives of the base.
        n_pre = min(max(self._degrees) + 1, self._PRECOMPUTE_N_POLYS)
        for n in range(1, n_pre):
            self._poly_funcs.append(self._poly_funcs[0].deriv(n))

    def _solve(self, y: float, x_range: (float, float)) -> [float]:
        # Get all polynomial roots using NumPy subtraction and roots().
        roots = np.array((self._poly_funcs[0] - y).roots(),
                         dtype=np.complex_)

        # Extract only real roots.  Tolerance on imaginary part is same
        # as SciPy _zeros_py.py.
        real_roots = np.real(roots[np.abs(np.imag(roots)) <
                                   4 * np.finfo(float).eps])

        return within_range(real_roots, x_range)


# ======================================================================
