import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import root_scalar

from pyavia.data import PCHIP1D, Function1D, Line1D


# Written by Eric J. Whitney, November 2023.


# ============================================================================


# TODO: Add impact_KE and impact_PE calculated fields, using the values at
#  t_impact.

@dataclass(frozen=True, kw_only=True)
class LandingImpact:
    # noinspection PyUnresolvedReferences
    """
    Results of a landing impact analysis, either transient or non-transient
    (pseudo-steady / averaged).  See `landing_energy` or `landing_dynamics`
    for more details.

    .. note:: If non-transient analysis is used (e.g. `landing_energy`)
       certain time-varying fields (`t`, `δ_s`, `F_s`, etc) will have
       `float` values instead of `ndarray`.  In this case attribute
       `transient == False`.

    Parameters
    ----------
    δ_s : ndarray[float] or float
        Vertical deflection of the overall aircraft rigid body / strut
        (sprung mass items), relative to the axle.

        .. note:: Gearing effects are excluded (e.g. refer `landing_energy`
           for details).

    δ_t : ndarray[float] or float
        Tyre / axle vertical deflection (unsprung mass items).

    E_s : ndarray[float] or float
        Energy stored in the strut.

    E_t : ndarray[float] or float
        Energy stored in the tyre.

    F_s : ndarray[float] or float
        Total vertical compression force produced by the strut acting on the
        aircraft rigid body due to all attachments.

        .. note:: This is the *net* force and does not correspond to any
           single attachment point load.  Gearing effects are excluded (e.g.
           refer `landing_energy` for details).

    F_t : ndarray[float]
        Tyre vertical force due to ground contact.

    g : float
        Acceleration due to gravity.

    L : ndarray[float]
        Lift on the aircraft.

    m_ac : float
        Aircraft *total* mass supported by the tyre/s and strut/s (> 0).
        This includes both the body (sprung) and tyre / axle (unsprung)
        components :math:`m_{ac} = m_s + m_t`.

    m_t : float
        Tyre, axle, brake, etc (unsprung) mass (`>= 0`). If unsprung mass
        effects are ignored this will be zero.

    t : ndarray[float]
        Impact timesteps.

        .. note:: For non-transient analysis a single value `t = 0.0`
           is given.

    t_impact : float
        Time of impact.

    V_sink : float
        Sink / descent velocity at impact.

    Attributes
    ----------
    δ_ac : ndarray[float] or float
        Vertical deflection of the entire aircraft relative to the ground.

    η_s : ndarray[float] or float
        Strut efficiency throughout the impact, defined as
        :math:`η_s = E_s /  (F_s δ_s)`.

    η_t : ndarray[float] or float
        Tyre efficiency throughout the impact, defined as
        :math:`η_t = E_t / (F_t δ_t)`.

    K : ndarray[float]
        Ratio of lift to total weight throughout the landing impact
        :math:`K = W_{ac} / L`.

    m_s : float
        Mass of the aircraft body above the tyre/s (sprung mass).

    N_s : ndarray[float] or float
        Dynamic / reaction factor of the aircraft body / strut forces.  This
        is the ratio of the combined load on the aircraft body to the
        strut static load, defined as :math:`N_s = F_s / W_s`.

        .. note:: This is not the same as the normal load factor experienced
           by the airframe `N_z`.  Also, if a non-zero tyre (unsprung) mass
           is used, then this value will be distinct from `N_t`, as the
           sprung weight does not include the weight of the tyre.

    N_t : ndarray[float] or float
        Dynamic / reaction factor of the tyre force.  This is the ratio of
        the tyre load to its static load, defined as :math:`N_t = F_t / W_ac`.

        .. note:: If a non-zero tyre (unsprung) mass is used, this will be
           different to `N_s`.

    N_z : ndarray[float] or float
        Normal vertical load factor on the aircraft as a whole.  This is
        defined as :math:`N_z = (F_s + L) / W_{ac} = F_b / W_{ac} + K` and
        because of the term :math:`F_b / W_{ac}` the effect of the unsprung
        mass is included (if nonzero).  The aircraft is assumed to be a
        rigid body.

    transient : bool
        `True` if a time-varying fields are present (`t`, `δ_s`, `F_s`,
        etc are `ndarray`) otherwise `False`.

    W_ac : float
        Aircraft *total* mass supported by the tyre/s and strut/s, including
        the sprung and unsprung components :math:`W_{ac} = m_{ac}.g`.  This
        is also equal to the static tyre load.

    W_s : float
        Weight of the body above the tyre/s (sprung) :math:`W_s = m_s.g`.

    W_t : float
        Weight of the tyre, axle, brake, etc (unsprung) :math:`W_t = m_t.g`.
    """
    δ_s: np.ndarray[float] | float  # Transient or pseudo-steady fields.
    δ_t: np.ndarray[float] | float
    E_s: np.ndarray[float] | float
    E_t: np.ndarray[float] | float
    F_s: np.ndarray[float] | float
    F_t: np.ndarray[float] | float
    L: np.ndarray[float] | float
    t: np.ndarray[float] | float

    g: float  # Constant fields.
    m_ac: float
    m_t: float
    t_impact: float
    V_sink: float

    def __post_init__(self):
        if np.ndim(self.t) == 0:
            # Scalar fields.
            reqd_shape = ()

        elif np.ndim(self.t) == 1:
            # List fields.  Check monotonic.
            reqd_shape = (self.t.size,)
            if not np.all(np.diff(self.t) > 0):
                raise ValueError("'t' must be monotonic increasing.")

        else:
            raise ValueError(f"Invalid t.shape = {self.t.shape}. Must be "
                             f"scalar or 1D list.")

        for check in (self.δ_s, self.δ_t, self.E_s, self.E_t,
                      self.F_s, self.F_t, self.L, self.t):
            if np.shape(check) != reqd_shape:
                raise ValueError(f"All inputs must match t.shape = "
                                 f"{reqd_shape}.")

    # -- Public Methods ------------------------------------------------------

    @property
    def δ_ac(self) -> np.ndarray[float] | float:
        return self.δ_s + self.δ_t
    @property
    def η_s(self) -> np.ndarray[float] | float:
        return self.E_s / (self.F_s * self.δ_s)

    @property
    def η_t(self) -> np.ndarray[float] | float:
        return self.E_t / (self.F_t * self.δ_t)


    @property
    def K(self) -> np.ndarray[float] | float:
        return self.L / self.W_ac

    @property
    def m_s(self) -> float:
        return self.m_ac - self.m_t

    @property
    def N_s(self) -> np.ndarray[float] | float:
        return self.F_s / self.W_s

    @property
    def N_t(self) -> np.ndarray[float] | float:
        return self.F_t / self.W_ac

    @property
    def N_z(self) -> np.ndarray[float] | float:
        return (self.F_s + self.L) / self.W_ac
    # TODO ^^^ Check this as the arithmetic might be off - When parked on
    #  the ground this won't equal one?  But it could also be right.

    @property
    def transient(self) -> bool:
        return np.ndim(self.t) > 0

    @property
    def W_ac(self) -> float:
        return self.m_ac * self.g

    @property
    def W_s(self) -> float:
        return self.m_s * self.g

    @property
    def W_t(self) -> float:
        return self.m_t * self.g


# ============================================================================


def _lookup_δ_U(δP_fn: Callable, P: float) -> (float, float):
    # Read deflections from curve and calculate energy by integrating under
    # the P-δ curve.  Note that as we are actually using a δ-P curve, this is
    # integral above the curve so we compute U = P.δ - integral(0 -> P).
    δ = δP_fn(P)
    Pδ = P * δ

    integral, *_ = quad(δP_fn,  # type: ignore
                        0, P,  # x = [0 -> P].
                        epsabs=1e-6 * Pδ,  # As proportion of total energy.
                        epsrel=1e-6)  # Unlikely to be critical.
    E = Pδ - integral
    return δ, E


# ----------------------------------------------------------------------------

def _process_δP_data(δP_data: ArrayLike | float) -> Function1D:
    # Process tabulated data or single constant and return an approximating
    # function of the form: δ = f(P).  Outside the normal domain,
    # this function returns:
    #   - P < 0:     δ = 0
    #   - P > P_max: Linear extrapolate δ (with warning if P_max finite).

    if np.ndim(δP_data) == 0:
        # Stiffness only supplied.  First check that it is positive.
        if δP_data <= 0:
            raise ValueError(f"Require P/δ > 0, got P/δ = {δP_data}.")

        δP_fn = Line1D(slope=1 / δP_data, intercept=0.0,
                       x_domain=(0.0, +np.inf),
                       ext_lo=0.0)

    elif np.ndim(δP_data) == 2:
        # Tabulated data supplied [(δ, P), ...].
        δP_data = np.array(δP_data, dtype=float, copy=True)
        if δP_data.shape[1] != 2:
            raise ValueError(f"(δ, P) data must have shape (N, 2), got "
                             f"{δP_data.shape}.")

        # Sort curve points in increasing order, force and deflection
        # values are positive and strictly monotonic.
        δP_data = δP_data[δP_data[:, 0].argsort()]
        if np.any(δP_data[0] < [0.0, 0.0]):
            raise ValueError("All (δ, P) values must be positive.")

        if np.any(np.diff(δP_data, axis=0) <= 0):
            raise ValueError("(δ, P) values are not monotonic.")

        # Create tyre P-δ function.
        δP_fn = PCHIP1D(δP_data[:, 1], δP_data[:, 0],  # Note order here.
                        x_domain=(0.0, None),  # Extend to zero if required.
                        ext_lo=0.0, ext_hi='linear')
    else:
        raise ValueError(f"Require stiffness of shape () or δ-P table of "
                         f"shape (N, 2), got shape {np.shape(δP_data)}.")
    return δP_fn


# ----------------------------------------------------------------------------

def landing_energy(strut: ArrayLike | float, tyre: ArrayLike | float,
                   g: float, K: float, m_ac: float, m_t: float,
                   V_sink: float) -> LandingImpact:
    r"""
    Compute the deflections and reactions for the tyre and strut during a
    landing impact using conservation of energy.

    Parameters
    ----------
    tyre, strut : array_like of float, shape (N, 2) or float
        Force-deflection characteristics of the tyre/s and strut/s:

        - If an *array* is supplied:  These are curve sample points
          (deflection, force) as rows, i.e. `[(δ_0, P_0), (δ_1, P_1), ...]`.
          These are automatically sorted and are used to generate an
          approximating function curve for the tyre / strut. If the
          unloaded point `(0, 0)` is not included then the approximating
          function is automatically extended to this point.  All deflections
          and forces must be positive (indicating compression) and
          monotonic; see `Notes` for details.

        - If a *single float* is supplied:  This is the linear stiffness
          :math:`K = P / δ` of the tyre / strut in consistent units,
          e.g. `lbf/in` or `N/mm`.

        See `Notes` for the case where there is gearing between the strut
        and the landing gear leg.

    g : float
        Acceleration due to gravity.

    K : float
        Ratio of lift to weight during the landing impact :math:`K =
        L / W_{ac} = L / (m_{ac} g)`.  Typical values are:

        - Light aircraft: `K` = 2/3
        - Transport aircraft: `K` = 1

    m_ac : float
        Aircraft *total* mass (`> 0`).  This includes both the sprung and
        unsprung components.  The sprung component is computed internally as
        :math:`m_s = m_{ac} - m_t`.

    m_t : float
        Tyre, axle, brake, etc (unsprung) mass (`>= 0`). Unsprung mass
        effects can be ignored by setting this to zero.

    V_sink : float
        Sink / descent velocity at impact.

    Returns
    -------
    LandingImpact
        See definition of `LandingImpact` for details.

    Notes
    -----
    - `strut` and `tyre` parameters can represent one or many tyre/s or
      strut/s, provided the supported mass (`m_ac`) and stiffness are chosen
      appropriately.  The `strut` force is defined as the force acting
      directly on the aircraft body.

    - If only part of the aircraft mass is supported during the impact,
      `m_ac` should be adjusted appropriately.

    - If there is gearing in the system (e.g. a levered leg) then this
      effect should be included by factoring the `strut` curve to give the
      resulting force and deflection on the aircraft.

      **Example:** If there was a 2:1 gearing between tthe aircraft and strut
      stroke, then the `strut` curve would be factored to give the aircraft
      force of :math:`F_{ac} = ½ F_s` and deflection of :math:`δ_{ac} = 2 δ_s`.

    - The method used is very similar to commonly used conservation of
      energy methods (e.g. [1]_ Section 3.2), with the main refinement being
      that the wheel / tyre (unsprung) mass can be specified separately.
      Because of this the potential energy of the unsprung at impact is
      computed separately from the total aircraft potential energy.  Because
      of this, a separate dynamic / reaction load factor is given for the
      aircraft / body (sprung) mass and the tyre / axle (unsprung) mass.

      The conservation of energy equation is:

        :math:`PE + KE = E_t + E_s`

      This can be expanded to:

        :math:`[W_s.δ_{ac} + W_t.δ_t - K.W_{ac}.δ_{ac}]_{PE} +
        [½ m_ac V_{sink}^2]_{KE} =
        [\int_0^{δ_t} F_t(δ) dδ]_{E_t} +
        [\int_0^{δ_s} F_s(δ) dδ]_{E_s}`

      Where:
        * :math:`δ_{ac} = δ_s + δ_t` is the whole aircraft stroke.
        * :math:`δ_s` and :math:`δ_t` are the strut and tyre stroke
          respectively.
        * `K` is the ratio of lift to aircraft weight (see `Parameters`).
        * :math:`W_{ac} = W_s + W_t` is the total aircraft weight.
        * :math:`W_s`and :math:`W_t` are the body (spring) and
          tyre (unsprung) weight respectively where :math:`W = mg`.
        * :math:`F_t(δ)` and :math:`F_s(δ)` are the tyre and strut force
          curves.
        * :math:`V_{sink}` is the vertical sink speed at impact.

    - Any units can be used provided they are consistent.

    - If the upper limit of the deflection is exceeded for the tyre or strut
      (array arguments only), a warning is generated.

    - The method used to balance the energy equation is only valid for tyre
      and strut characteristics where the force continuously increases with
      deflection.  This requires all deflections and forces to be
      positive (compression) / monotonic.

    .. [1] Currey, Norman S., "Aircraft Landing Gear Design: Principles and
       Practices", AIAA Education Series 1988.
    """

    # Convert tyre and strut data to functions.
    tyre_fn = _process_δP_data(tyre)
    strut_fn = _process_δP_data(strut)

    # Setup result to hold impact details.  Dummy zeros are used for unknown
    # δ, E, F values at this stage.
    result = LandingImpact(δ_s=0, δ_t=0, E_s=0, E_t=0, F_s=0, F_t=0,
                           g=g, L=K * m_ac * g, m_ac=m_ac, m_t=m_t,
                           t=0, t_impact=0, V_sink=V_sink)

    def ΔE(N_t_: float) -> float:
        # Update the impact object given a trial value for the tyre / ground
        # reaction factor N_t and compute the energy balance error.
        nonlocal result

        # Compute tyre and strut forces using definition of N_t (N_gnd)
        # and equilibrium.
        F_t_ = N_t_ * result.W_ac
        F_s_ = F_t_ - result.W_t

        # Get tyre / strut deflection and absorbed energy and calculate
        # overall stroke.
        δ_t_, E_t_ = _lookup_δ_U(tyre_fn, F_t_)
        δ_s_, E_s_ = _lookup_δ_U(strut_fn, F_s_)
        δ_ac = δ_s_ + δ_t_

        # Update object
        result = replace(result, δ_s=δ_s_, δ_t=δ_t_, E_s=E_s_, E_t=E_t_,
                         F_s=F_s_, F_t=F_t_)

        # Get impact PE, KE and return ΔE = Input - Absorbed.
        # PE = (1 - K) * result.W_ac * (δ_b_ + δ_t_) - result.W_t * δ_b_
        # TODO these can be moved to LandingImpact as members.
        PE = (result.W_s * result.δ_ac + result.W_t * δ_t_
              - K * result.W_ac * result.δ_ac)
        KE = 0.5 * result.m_ac * result.V_sink ** 2

        return (PE + KE) - (E_t_ + E_s_)  # LHS - RHS

    # Compute solution and check convergence.
    sol = root_scalar(ΔE, x0=1.25, x1=1.75,  # Init. guesses for N_t.
                      xtol=1e-6, rtol=1e-7, maxiter=50)

    if not sol.converged or sol.root <= 0:
        raise RuntimeError("Failed to compute ground dynamic load factor.")

    ΔE(sol.root)  # Re-run with converged value to finalise result (JIC).

    # Check that the strut and tyre max loads are not exceeded.
    if result.F_s > strut_fn.x_domain[1]:
        warnings.warn(f"Strut force {result.F_s:.05G} exceeded maximum "
                      f"of {strut_fn.x_domain[1]:.5G}.")

    if result.F_t > tyre_fn.x_domain[1]:
        warnings.warn(f"Tyre force {result.F_t:.05G} exceeded maximum "
                      f"of {tyre_fn.x_domain[1]:.5G}.")

    return result


# WAS 2023-11-14 05:15
# def landing_energy(strut: ArrayLike | float, tyre: ArrayLike | float,
#                    g: float, gearing: float, K: float,
#                    m_ac: float, m_t: float, V_sink: float) -> LandingImpact:
#     r"""
#     Compute the deflections and reactions for the tyre and strut during a
#     landing impact using conservation of energy.
#
#     Parameters
#     ----------
#     tyre, strut : array_like of float, shape (N, 2) or float
#         Force-deflection characteristics of the tyre/s and strut/s:
#
#         - If an *array* is supplied:  These are curve sample points
#           (deflection, force) as rows, i.e. `[(δ_0, P_0), (δ_1, P_1), ...]`.
#           These are automatically sorted and are used to generate an
#           approximating function curve for the tyre / strut. If the
#           unloaded point `(0, 0)` is not included then the approximating
#           function is automatically extended to this point.
#
#           .. note:: The method used to balance the energy equation is only
#              valid for tyre and strut characteristics where the force
#              continuously increases with deflection.  As such all
#              deflections and forces must be positive (indicating
#              compression) and monotonic.
#
#         - If a *single float* is supplied:  This is the linear stiffness
#           :math:`K=P/δ` of the tyre / strut in consistent units,
#           e.g. `lbf/in` or `N/mm`.
#
#         These characteristics can represent one or many tyre/s or strut/s,
#         provided the sprung mass (`m_ac`) is chosen appropriately.
#
#     g : float
#         Acceleration due to gravity.
#
#     gearing : float
#         Gearing (`G`) of the body vertical displacement to the vertical
#         displacement of the strut :math:`G = δ_{body} / δ_s`.  Many aircraft
#         use a vertical oleo strut so that :math:`δ_{body} = δ_s` resulting
#         in :math:`G = 1`.  Some aircraft however use an offset strut (e.g.
#         Lockheed F-104, DHC-6) so that :math:`δ_{body} > δ_s` such that
#         :math:`G > 1`.
#
#     K : float
#         Ratio of lift to weight during the landing impact :math:`K = L/W =
#         L / [(m_{ac} + m_{t}) g]`.  Typical values are:
#             * Light aircraft: `K` = 2/3
#             * Transport aircraft: `K` = 1
#
#     m_ac : float
#         Proportion of aircraft *total* mass supported by the strut/s (`>
#         0`).  This includes both the sprung and unsprung components.  The
#         srpung component is computed internally as :math:`m_s = m_{ac} - m_t`.
#
#     m_t : float
#         Tyre, axle, brake, etc (unsprung) mass (`>= 0`). Unsprung mass
#         effects can be ignored by setting this to zero.
#
#     V_sink : float
#         Sink / descent velocity at impact.
#
#     Returns
#     -------
#     LandingImpact
#         See `LandingImpact` definition for details.
#
#     Notes
#     -----
#
#     - The method used is very similar to commonly used conservation of
#       energy methods (e.g. [1]_ Section 3.2), with the main refinement being
#       that the wheel / tyre (unsprung) mass can be specified separately.
#       Because of this the potential energy of the unsprung at impact is
#       computed separately from the total aircraft potential energy.  Because
#       of this, a separate dynamic / reaction load factor is given for the
#       aircraft / strut (sprung) mass and the tyre (unsprung) mass.
#
#       The conservation of energy equation becomes:
#
#         :math:`E_t + E_s = PE_t + PE_s + KE`
#
#       Where subscript `t` refers to the tyre or unsprung mass and `s` refers
#       to the strut or sprung mass.  This can be expanded to:
#
#         :math:`\int_0^{δ_t} F_t(δ) dδ + \int_0^{δ_s} F_s(δ) dδ =
#         W_t.δ_t + (W_s - L)(δ_s + δ_t) + ½(m_s + m_t) V_{sink}^2`
#
#       Where:
#         * :math:`δ_t` and :math:`δ_s` are the tyre / strut deflection and
#           stroke.
#         * :math:`F_t(δ)` and :math:`F_s(δ)` are the tyre / strut force curves.
#         * :math:`W_t` and :math:`W_s` are the unsprung and sprung weight.
#         * `L` is the ift acting throughout the impact.
#         * :math:`V_{sink}` is the sink speed at impact.
#
#     - Any units can be used provided they are consistent.
#
#     - If the upper limit of the deflection is exceeded for the tyre or strut
#       (array arguments only), a warning is generated.
#
#     .. [1] Currey, Norman S., "Aircraft Landing Gear Design: Principles and
#        Practices", AIAA Education Series 1988.
#     """
#
#     # Convert tyre and strut data to functions.
#     tyre_fn = _process_δP_data(tyre)
#     strut_fn = _process_δP_data(strut)
#
#     # Setup result to hold impact details.  Dummy zeros are used for unknown
#     # δ, E, F values at this stage.
#     result = LandingImpact(δ_s=0.0, δ_t=0.0, E_s=0.0, E_t=0.0,
#                            F_s=0.0, F_t=0.0, g=g, L=K * m_ac * g,
#                            m_ac=m_ac, m_t=m_t, t=0.0, V_sink=V_sink)
#
#     def ΔE(N_t_: float) -> float:
#         # Update the impact object given a trial value for the tyre / ground
#         # reaction factor N_t and compute the energy balance error.
#         nonlocal result
#
#         # Compute tyre and strut forces using definition of N_t (N_gnd)
#         # and equilibrium.
#         F_t_ = N_t_ * result.W_ac
#         F_s_ = F_t_ - result.W_t
#
#         # Get tyre / strut deflection and absorbed energy.
#         δ_t_, E_t_ = _lookup_δ_U(tyre_fn, F_t_)
#         δ_s_, E_s_ = _lookup_δ_U(strut_fn, F_s_)
#
#         # Get impact energy components.
#         PE_t = result.W_t * δ_t_  # PE wheel / tyre (unsprung).
#         PE_s = (result.W_s - result.L) * (δ_s_ + δ_t_)  # PE remain (sprung).
#         KE = 0.5 * result.m_ac * result.V_sink ** 2  # KE (total).
#
#         # Update object and return ΔE = Input - Absorbed
#         result = replace(result, δ_s=δ_s_, δ_t=δ_t_, E_s=E_s_, E_t=E_t_,
#                          F_s=F_s_, F_t=F_t_)
#
#         return (PE_t + PE_s + KE) - (E_t_ + E_s_)
#
#     # Compute solution and check convergence.
#     sol = root_scalar(ΔE, x0=1.25, x1=1.75,  # Init. guesses for N_t.
#                       xtol=1e-6, rtol=1e-7, maxiter=50)
#
#     if not sol.converged or sol.root <= 0:
#         raise RuntimeError("Failed to compute ground dynamic load factor.")
#
#     ΔE(sol.root)  # Re-run with converged value to finalise result (JIC).
#
#     # Check that the strut and tyre max loads are not exceeded.
#     if result.F_s > strut_fn.x_domain[1]:
#         warnings.warn(f"Strut force {result.F_s:.05G} exceeded maximum "
#                       f"of {strut_fn.x_domain[1]:.5G}.")
#
#     if result.F_t > tyre_fn.x_domain[1]:
#         warnings.warn(f"Tyre force {result.F_t:.05G} exceeded maximum "
#                       f"of {tyre_fn.x_domain[1]:.5G}.")
#
#     return result

# ============================================================================
