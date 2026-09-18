from typing import Type, cast, TypeVar, Callable

import numpy as np
from numpy import typing as npt
from scipy.optimize import least_squares

from pyavia.fluids._gas import Gas
from pyavia.numeric.solve import SolverError


# Original by Eric J. Whitney, December 2020.

# ======================================================================

# Floating point precision.
_ε = cast(float, np.finfo(float).tiny)  # Cast avoids spurious warning.

# General bounds for gas properties.  If not specified, +/- np.inf is
# used. Note that enthalpy and entropy bases are theoretically
# arbitrary, so these can be negative.

# FutureWork: I'm not sure if these separate properties are *really*
# still needed if we organise the imports properly.

# Setup generic initial values for gas properties to start convergence,
# if a reference gas is not supplied. At the moment these are based on
# PerfectAir at sea level, 15°C, Mach 0.5 (SI units).
_PROP_INIT = {
    'M'   : 0.5,
    'R'   : 287.05287,
    'T'   : 288.15,
    'T0'  : 302.5575,
    'T0_T': 1.05,
    'V'   : 170.146994,
    'a'   : 340.293988,
    'c_p' : 1004.6850,
    'c_v' : 717.632175,
    'h'   : 710713.149,
    'h0'  : 725188.149,
    'p'   : 101325,
    'p0'  : 120192.995,
    'p0_p': 1.18621263,
    's'   : 5644.20615,
    'γ'   : 1.4,
    'μ'   : 1.78938028e-05,
    'ρ'   : 1.225
}

_PROP_LO_BOUND = {
    'p': _ε,
    'p0': _ε,
    'T': _ε,
    'T0': _ε,  # P, T strict > 0
    'M': 0.0,  # M >= 0
}

_PROP_HI_BOUND = {
    'p': np.inf,
    'p0': np.inf,
    'T': np.inf,
    'T0': np.inf,
    'M': np.inf,
}

# Type for covariant generic arguments.
T_co = TypeVar('T_co', covariant=True)

# ----------------------------------------------------------------------

# TODO This needs to be replaced with a more generic 'fit_model_lsq' as we
#  di this for more than just gases.

# TODO Unit support - presently SI units.
def make_gas[T_co: Gas](gas_type: Type[T_co],
                     init_props: tuple[str, ...] = ('p', 'T', 'M'),
                     *, ref_gas: Gas | None = None,
                     tol: float = 1e-9,
                     **target_props: float | str) -> T_co:
    r"""
    Create and initialise a gas object of type `gas_type`, by
    calculating values for the (possibly unknown) initialisation
    properties in `init_props`.  This is acheived by iteratively
    converging the properties until the known properties in
    `**reqd_props` are matched. If a property in `init_props` is already
    present in `**target_props` then the value is simply substituted and
    that property is treated as known.

    Parameters
    ----------
    gas_type : Type[Gas]
        Specific type of gas to instantiate.

    init_props : tuple[str, ...]
        Comma-separated list of keywords that will be passed to
        `gas_type.__init__(...)`.

    ref_gas : Type[Gas], optional
        If provided, convergence can be accelerated by supplying a gas
        to use as an initialisation reference point.  This should be
        close / similar to the final result.

    tol : float, optional
        Convergence tolerance for each value in `**reqd_props`.
        Criteria is::

        .. math:: |x - x^*| \leq max(|x^*| \times tol, tol)

    **target_props : dict[str, float]
        When converged, the resulting gas must have these properties
        (within the requested tolerance).

    Returns
    -------
    Gas
        Resulting gas object.

    Examples
    --------

    For example, create and initialise a `PolyAir` object by finding
    values for `p`, `T` and `M`, where pressure, Mach number and
    enthalpy are already known::

        ``result = init_gas(PolyAir, ('p','T','M'), p=..., M=..., h=...)``

    In this case since `p` and `M` are already known, only the
    temperature (`T`) needs to be calculated.
    """
    # Build known / unknown __init__ properties, bounds, initial guess.
    known_props, known_val = [], []
    unknown_props, unknown_lb, unknown_ub = [], [], []
    x0, x_scale = [], []

    for prop in init_props:
        if prop in target_props:
            # Already know this value.
            known_props.append(prop)
            known_val.append(target_props[prop])

        else:
            # Need to find this value.  Make an initial guess.
            unknown_props.append(prop)

            if ref_gas is not None:
                x0.append(getattr(ref_gas, prop))
            else:
                try:
                    x0.append(_PROP_INIT[prop])
                except KeyError:
                    raise ValueError(f"No initial value available for "
                                     f"property '{prop}'.")

            # Set the bounds (where available).
            try:
                unknown_lb.append(_PROP_LO_BOUND[prop])
            except KeyError:
                unknown_lb.append(-np.inf)

            try:
                unknown_ub.append(_PROP_HI_BOUND[prop])
            except KeyError:
                unknown_ub.append(np.inf)

    # Setup a fixed list of required (f) property names and array
    # of target values. This is done to avoid constantly iterating
    # through dicts.
    f_name = list(target_props.keys())
    f_target = np.array([target_props[name] for name in f_name])

    # Set 'x' and 'f' scale factors using |x0| / |f_target| or lower
    # limit of 1.0.
    x_scale = _scale_factors(x0, low_cutoff=1.0)
    f_scale = _scale_factors(f_target, low_cutoff=1.0)

    # ------------------------------------------------------------------

    def gas_x(x: npt.NDArray) -> Type[T]:
        # Define a function to create gas from unknowns given in 'x',
        # by combining 'unknown' and 'known' __init__ kwargs.

        init_kwargs = {k_: v_ for k_, v_ in zip(unknown_props, x)}
        init_kwargs |= {k_: v_ for k_, v_ in zip(known_props, known_val)}
        return gas_type(**init_kwargs)

    # ------------------------------------------------------------------

    def residuals(x: npt.NDArray) -> npt.NDArray:
        # Residual function computes the error between the target and
        # current properties for a gas using unknowns in 'x'. For 'P'
        # properties and 'N' operating points, this will give a 'PxN'
        # residual array.  Target is broadcast subtracted from each
        # operating point.

        gas = gas_x(x)
        f_x = np.array([getattr(gas, name_) for name_ in f_name])
        f_res_scl = (f_x - f_target) / f_scale
        return f_res_scl.flatten()

    # ------------------------------------------------------------------

    # Solve for the unknowns using a least-squares process.  This covers
    # over-determined cases as well.  At the end we need to check that
    # all the variables are within a close tolerance.
    result = least_squares(
        residuals, x0,
        bounds=(unknown_lb, unknown_ub),
        x_scale=x_scale,
        ftol=tol,  # Note: Scaled.
        xtol=tol,  # Note: Scaled.
        gtol=_ε,
        max_nfev=50 * len(x0)
    )

    if not result.success:
        raise SolverError(f"Failed to initialise gas.", flag=1,
                          details=result.message)

    # Create the resulting gas and reverse-check that the given
    # properties are all reproduced within the tolerance.
    result_gas = gas_x(result.x)
    for prop, target in target_props.items():
        val = getattr(result_gas, prop)

        if np.abs(val - target) > np.maximum(np.abs(target) * tol, tol):
            raise SolverError(
                f"Failed to initialise gas.", flag=2,
                details=f"Property outside tolerance after convergence. "
                        f"Got {prop}={val}, required {prop}={target}."
            )

    return result_gas


# ----------------------------------------------------------------------

def _scale_factors(x: npt.ArrayLike, low_cutoff: float) -> npt.NDArray:
    """Make suitable scale factors for M, P, T, h, s values, etc."""
    return np.clip(np.abs(x), a_min=low_cutoff, a_max=None)
