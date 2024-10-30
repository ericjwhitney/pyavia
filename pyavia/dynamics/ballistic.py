from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp, OdeSolution
# noinspection PyProtectedMember
from scipy.integrate._ivp.ivp import OdeResult  # Shush type checker.

# Written by Eric J. Whitney, October 2023.


# ======================================================================


class BallisticTrajectory:
    """
    Ballistic trajectory calculation results, returned from
    function `ballistic` or `ballistic_variable`.

    Notes
    -----
    * Units correspond to input values used in call to function
      `ballistic`, etc.
    * Objects of this class are not expected to be created by the user.
    """

    def __init__(self, t: NDArray[float], y: NDArray[float],
                 y_sol: OdeSolution, t_apogee: float):
        # t = [t0, t1, ...] until impact.
        # y = [[x0, z0, u0, w0], [x1, z1, u1, w1], ...].
        # t_apogee provided to avoid recalculating it, or zero if there
        # was no separate apogee.

        self._t = t
        self._y = y
        self._y_sol = y_sol
        self._t_apogee = t_apogee

        if self._t.ndim != 1:
            raise ValueError(f"Require t.shape == (npts,), "
                             f"got {self._t.shape}.")

        if self._y.ndim != 2 or self._y.shape[0] != 4:
            raise ValueError(f"Require y.shape == (4, {self._t.size}), "
                             f"got {self._y.shape}")

        if self._t_apogee < self._t[0] or self._t_apogee > self._t[-1]:
            raise ValueError(f"t_apogee = {self._t_apogee} out of range.")

    @property
    def t(self) -> NDArray[float]:
        """Return the discrete / computed `t` values."""
        return self._t

    @property
    def t_apogee(self) -> float:
        """Return the time of apogee."""
        return self._t_apogee

    @property
    def t_impact(self) -> float:
        """Return the time of impact."""
        return float(self._t[-1])

    @property
    def θ(self) -> NDArray[float]:
        """Return the discrete trajectory angles for each `t`."""
        return np.arctan2(self.w, self.u)

    def θ_t(self, t: ArrayLike) -> ArrayLike:
        """Return trajectory / flightpath angle at given time/s `t`."""
        return np.arctan2(self.w_t(t), self.u_t(t))

    @property
    def u(self) -> NDArray[float]:
        """Return the discrete `u` values for each `t`."""
        return self._y[2, :]

    def u_t(self, t: ArrayLike) -> ArrayLike:
        """Return the `u` value at given time/s `t`."""
        return self._y_sol(t)[2]

    @property
    def V(self) -> NDArray[float]:
        """Return discrete total velocity `V` values for each `t`."""
        return np.sqrt(self.u ** 2 + self.w ** 2)

    def V_t(self, t: float) -> float:
        """Return the total velocity `V` at given time/s `t`."""
        return np.sqrt(self.u_t(t) ** 2 + self.w_t(t) ** 2)

    @property
    def w(self) -> NDArray[float]:
        """Return the discrete `w` values for each `t`."""
        return self._y[3, :]

    def w_t(self, t: ArrayLike) -> ArrayLike:
        """Return the `w` value at given time/s `t`."""
        return self._y_sol(t)[3]

    @property
    def x(self) -> NDArray[float]:
        """Return the discrete `x` values for each `t`."""
        return self._y[0, :]

    def x_t(self, t: ArrayLike) -> ArrayLike:
        """Return the discrete `x` value at given time/s `t`."""
        return self._y_sol(t)[0]

    @property
    def z(self) -> NDArray[float]:
        """Return the discrete `z` values for each `t`."""
        return self._y[1, :]

    def z_t(self, t: ArrayLike) -> ArrayLike:
        """Return the `z` value at given time/s `t`."""
        return self._y_sol(t)[1]


# ----------------------------------------------------------------------

def ballistic(V0: float, θ0: float, m: float, g: float, ρ: float = 0.0,
              f: float = 0.0, xz0: (float, float) = (0.0, 0.0), *,
              max_dt: float = None, method: str = 'RK45',
              rtol: float = 1e-4, atol: float = 1e-7) -> BallisticTrajectory:
    """
    Compute the trajectory of a ballistic object.  The object is assumed
    to have a fixed drag area.  Computation is continued until ground
    contact is made (where `z` = 0), and the result is returned in a
    `BallisticTrajectory` object.

    Parameters
    ----------
    V0 : float
        The initial velocity of the object (> 0).
    θ0 : float
        The launch angle of the object with respect to the `x` axis.
        Positive values correspond to an upwards launch direction.
        *Note:* Radians are normally required for consistency.
    m : float
        The mass of the object.
    g : float
        The acceleration due to gravity, e.g. in SI units g = 9.80665
        m/s^2.
    ρ : float, default = 0.0
        The atmospheric air density (constant), e.g. for ISA sea level
        in ISA units ρ = 1.225 kg/m^3.  If zero (the default), then drag
        is ignored.
    f : float, default = 0.0
        The object 'drag area' (constant), which is the product of the
        drag coefficient by the applicable reference area, e.g.
        :math:`f = C_{D}S_{REF}`.  If zero (the default), then drag is
        ignored.
    xz0 : (float, float), default = (0.0, 0.0)
        The object starting position in the x-z plane, where `x` is the
        downrange distance and `z` is the altitude.
    max_dt : float, optional
        Maximum timestep to use in calculation.  If `None` (the default)
        a timestep suitable for normal cases is computed.
    method : str, default = 'RK45'
        Integration method to use - see `scipy.integrate.solve_ivp` for
        more options.
    rtol : float, default = 1e-4
        Same as scipy.integrate.solve_ivp.
    atol : float, default = 1e-7
        Same as scipy.integrate.solve_ivp.

    Returns
    -------
    BallisticTrajectory

    Notes
    -----
    * To calculate a trajectory where the drag is a variable function in
      flight, see :func:`ballistic_variable`.
    * Any units may be used provided they are consistent.
    """

    # noinspection PyUnusedLocal
    def basic_drag(V: float, θ: float, z: float) -> float:
        # Setup drag function using fixed parameters. 'θ' and 'z' are
        # unusued.
        return 0.5 * ρ * V ** 2 * f

    return ballistic_variable(V0, θ0, m, g, basic_drag, xz0,
                              max_dt=max_dt, method=method,
                              rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------


# noinspection PyIncorrectDocstring
def ballistic_variable(V0: float, θ0: float, m: float, g: float,
                       drag_fn: Callable[[float, float, float], float],
                       xz0: (float, float) = (0.0, 0.0), *,
                       max_dt: float = None, method: str = 'RK45',
                       rtol: float = 1e-4, atol: float = 1e-7
                       ) -> BallisticTrajectory:
    """
    Compute the trajectory of a ballistic object with a user-specified
    drag function.  This function has the same parameters as
    `ballistic`, except that atmospheric properties and reference areas
    are replaced by the supplied drag function.

    Parameters
    ----------
    drag_fn : Callable[[array_like, array_like, array_like], array_like]

        Function accepting three parameters (`V`, `θ`, `z`), returning
        the drag force in consistent units.

    method : str, default = 'RK45'
        The default intergration method is the Explicit Runge-Kutta
        method of order 5(4) - see `scipy.integrate.solve_ivp` for more
        options.  If the drag function has sudden variations during
        flight, then an implicit method such as `LSODA` may be more
        appropriate.

    Returns
    -------
    BallisticTrajectory
    """
    if V0 <= 0.0:
        raise ValueError("Requires V0 > 0.")

    if m <= 0.0:
        raise ValueError("Requires m > 0.")

    if not callable(drag_fn):
        raise ValueError("Requires drag_fn to be callable.")

    if np.ndim(xz0) != 1 or len(xz0) != 2:
        raise ValueError("xz0 must be a 1D array like (x0, z0).")

    if max_dt is None:
        # Set to approximately a 5% change in velocity per timestep
        # under gravitational acceleration alone.

        # Note: presently 0.04999 is used to prevent the bug
        # https://github.com/scipy/scipy/issues/19418 which results in
        # failure of scipy.integrate.solve_ivp.  This can happen if the
        # initial conditions are an exact multiple of the timestep.
        max_dt = 0.04999 * V0 / g

    if max_dt <= 0.0:
        raise ValueError("Requires max_dt > 0.")

    # Setup the initial state vector y = [x, z, u, w].
    y0 = np.array([xz0[0], xz0[1], V0 * np.cos(θ0), V0 * np.sin(θ0)])

    # Run the simulation.
    # noinspection PyTypeChecker
    result: OdeResult = solve_ivp(_ballistic_deriv, (0.0, np.inf), y0,
                                  args=(m, g, drag_fn), method=method,
                                  dense_output=True,
                                  events=(_apogee, _impact),
                                  max_step=max_dt,
                                  rtol=rtol, atol=atol)

    if not result.success or result.status != 1:
        # TODO: SolutionError?
        raise RuntimeError(f"Failed to compute trajectory. ODE solver "
                           f"message: {result.message}")

    if len(result.t_events[0]) > 0:
        # Reached an apogee.
        t_apogee = result.t_events[0][0]
    else:
        # No apogee.
        t_apogee = 0.0

    return BallisticTrajectory(result.t, result.y, result.sol, t_apogee)


# ======================================================================

# noinspection PyUnusedLocal
def _apogee(t: float, y: ArrayLike, *unused) -> float:
    # An apogee event is the zero crossing point of vertical velocity
    # (w).
    return y[3]


_apogee.direction = -1  # Downwards crossings only.


# ----------------------------------------------------------------------

# noinspection PyUnusedLocal
def _ballistic_deriv(t: float, y: ArrayLike, m: float, g: float,
                     drag_fn: Callable[[float, float, float], float]
                     ) -> NDArray:
    # Calculates the time derivative of the state vector: dy/dt =
    # f(t, y).
    x, z, u, w = y  # Expand y = [x, z, u, w] for clarity.
    V = np.sqrt(u ** 2 + w ** 2)
    θ = np.arctan2(w, u)  # arctan2 required due to separate drag fn.
    drag = drag_fn(V, θ, z)

    return np.array([u,  # dx/dt
                     w,  # dz/dt
                     -drag * np.cos(θ) / m,  # du/dt
                     -drag * np.sin(θ) / m - g])  # dw/dt


# ----------------------------------------------------------------------


# noinspection PyUnusedLocal
def _impact(t: float, y: ArrayLike, *unused) -> float:
    # An impact event is the zero crossing point of the altitude (z).
    return y[1]


_impact.terminal = True  # Stop on crossing.
_impact.direction = -1  # Downwards crossings only.

# ----------------------------------------------------------------------
