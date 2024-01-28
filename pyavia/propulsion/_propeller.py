"""
TODO In development.
This module contains functions relating to propeller performance.
"""
# Last updated: 22 January 2023 by Eric J. Whitney.
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

# ---------------------------------------------------------------------------
#
#
# class XROTOR2DAero:
#     """
#     Contains the complete performance of a propeller at a single operating
#     point.
#     """
#
#     def __init__(self, V: float, omega: float, D: float, ρ: float, T: float,
#                  P_shaft: float, r: [DimOpType] = None, cl: [float] = None,
#                  cd: [float] = None,
#                  Vn_prop: [float] = None,dr: [float] = None):
#         """
#         To define propeller performance using other parameters, see class
#         method ``from_...``.
#
#         Parameters
#         ----------
#         V : float
#             Airspeed.
#         omega: float
#             Rotational speed (Ω).
#
#             Note:: Typically rad/s required for consistent units.
#         D: float
#             Diameter.
#         ρ: float
#             Freestream air density.
#         T : float
#             Thrust.
#         P_shaft : float
#             Power.
#         r : [float] (optional)
#             Radial stations for cl, cd data (if provided).
#         cl : [float]
#             cl at radial stations (if provided).
#         cd : [float]
#             cd at radial stations (if provided).
#         Vn_prop : [float]
#             Normal / axial speed at the propeller disc (including induced
#             flow) at radial stations (if provided).
#         dr : [float]
#             Annulus width applicable to each radius (if provided).
#         """
#         self.V, self.omega = V, omega
#         self.D, self.ρ = D, ρ
#         self.T, self.P_shaft = T, P_shaft
#
#         # TODO should probably just used the stations not the element
#         #  values.  We can compute dr internally.  Allows different
#         #  representations.
#
#         self.r, self.cl, self.cd = r, cl, cd
#         self.Vn_prop = Vn_prop
#         self.dr = dr

    # -- Public Methods -----------------------------------------------------

    # @property
    # def C_P(self) -> float | None:
    #     """
    #     Power coefficient C_P = P_shaft / (ρn²D⁵).
    #     """
    #     return self.P_shaft / (self.ρ * self.n ** 3 * self.D ** 5)

    # @property
    # def C_Q(self) -> float:
    #     """
    #     Torque coefficient C_Q = C_P / 2π.
    #     """
    #     return self.C_P / (2 * np.pi)

    # @property
    # def C_T(self) -> float:
    #     """
    #     Thrust coefficient C_T = T / (ρn²D⁴).
    #     """
    #     return self.T / (self.ρ * self.n ** 2 * self.D ** 4)

    # @property
    # def η(self) -> float:
    #     """Efficiency η = J.C_T / C_P."""
    #     return self.J * self.C_T / self.C_P
    #
    # @classmethod
    # def from_TQ(cls, V: float, omega: float, D: float, ρ: float, T: float,
    #             Q: float, *args, **kwargs) -> XROTOR2DAero:
    #     """
    #     Returns a XROTOR2DAero object computed from thrust, torque and other
    #     corresponding values.
    #     """
    #     return XROTOR2DAero(V=V, omega=omega, D=D, ρ=ρ, T=T, P_shaft=omega * Q,
    #                     *args, **kwargs)
    #
    # # TODO: Add more of these from_ as required.
    #
    # @property
    # def Q_shaft(self) -> float:
    #     """Shaft Torque Q = P / Ω."""
    #     return self.P_shaft / self.omega
    #
    # @property
    # def J(self) -> float:
    #     """
    #     Advance ratio J = V / (n.D).
    #     """
    #     return self.V / (self.n * self.D)
    #
    # @property
    # def n(self) -> float:
    #     """
    #     Rotational speed in rev/sec.
    #     """
    #     return self.omega / (2 * np.pi)
    #
    # @property
    # def RPM(self):
    #     """
    #     Rotational speed in rev/min.
    #     """
    #     return self.omega * 60 / (2 * np.pi)
    #
    # @property
    # def Vn_prop_mean(self) -> float | None:
    #     """
    #     Mean normal flow speed (including induced flow) at the propeller
    #     disc.
    #     """
    #     if self.dr is None or self.Vn_prop is None:
    #         return None
    #
    #     # Compute annulus area fractions.
    #     r_i = self.r - 0.5*self.dr
    #     r_o = self.r + 0.5 * self.dr
    #     R_root, R_tip = r_i[0], r_o[-1]
    #     ΔA_A = (r_o ** 2 - r_i ** 2) / (R_tip ** 2 - R_root ** 2)
    #     return float(np.sum(self.Vn_prop * ΔA_A))


# ===========================================================================

class Propeller(ABC):
    """
    Basic definition of characteristics shared by all types of propeller
    model.

    The following subscripts are used to help standardise variable names
    that represent stages of flow through a propeller:
        - 0 - Freestream (or far upstream) properties.
        - 2 - Immediately ahead of the disc.
        - p - At the disc, relative to propeller (moving frame).
        - 3 - Immediately behind the disc.
        - 9 - Distant wake (or far downstream) of the propeller.

    If velocities / forces / etc are broken into components, the following
    suffixes are generally used:
        - `a` - Axial.
        - `t` - Tangential (`θ` direction).
        - `r` - Radial.

    Any units can be used provided they are consistent.  In particular this
    normally means that angles are taken to be radians, and rotational
    velocities `Ω` are radians/s.  No internal corrections for degrees,
    RPM, etc are used.
    """
    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError(f"Got unexpected keyword arguments: " +
                            ", ".join(kwargs.keys()))

    # -- Public Methods -----------------------------------------------------

    @property
    def Ap(self) -> float:
        """Disc area, computed from Ap = πR²."""  # TODO Hub / root?
        return np.pi * self.R ** 2

    @property
    def CP(self) -> float:
        """
        Power coefficient (propeller type).  Defined as C_P = P_shaft /
        (ρ * n ** 3 * D ** 5).
        """
        return self.Ps / (self.ρ0 * self.n ** 3 * self.D ** 5)

    @property
    def CQ(self) -> float:
        """
        Torque coefficient (propeller type).  Defined as C_Q = C_P / (2π).
        """
        return self.CP / (2 * np.pi)

    @property
    def CT(self) -> float:
        """
        Thrust coefficient (propeller type).  Defined as T / (ρ * n ** 2 * D
        ** 4).
        """
        return self.T / (self.ρ0 * self.n ** 2 * self.D ** 4)

    @property
    def D(self) -> float:
        """Tip / overall diameter, computed from D = 2.R."""
        return self.R * 2

    @property
    def η(self) -> float:
        """
        Efficiency η = J.CT / CP.
        """
        return self.J * self.CT / self.CP

    @property
    def FM(self) -> float:
        """
        Figure of Merit (propeller type), defined as: FM = sqrt(2/π) *
        CT**1.5 / CP
        """
        return np.sqrt(2.0 / np.pi) * (self.CT ** 1.5) / self.CP

    @property
    def J(self) -> float:
        """
        Advance ratio, computed from J = V / (n.D).
        """
        return self.V0 / (self.n * self.D)

    @property
    def n(self) -> float:
        """
        Rotational speed in rev/sec, computed from n = Ω / (2π).
        """
        return self.Ω / (2 * np.pi)

    @property
    @abstractmethod
    def Ω(self) -> float:
        """
        Rotational speed `Ω`.  *Note* in consitent units, this is normally
        *rad/s*.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def Ps(self) -> float:
        """
        Shaft power:
            - Positive (+) when the propeller is being driven.
            - Negative (-) in windmill state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def Qs(self) -> float:
        """
        Shaft torque:
            - Positive (+) when the propeller is being driven.
            - Negative (-) in windmill state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def R(self) -> float:
        """Tip / overall radius."""
        raise NotImplementedError

    @property
    @abstractmethod
    def R_root(self) -> float:
        """
        Root (hub, smallest working) radius.  If non-zero this
        represents spinners or sections of blade that have no effect on
        performance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ρ0(self) -> float:
        """Freestream air density."""
        raise NotImplementedError

    @property
    @abstractmethod
    def T(self) -> float:
        """
        Thrust:
            - Positive (+) when the propeller is producing thrust.
            - Negative (-) during propeller braking / windmill.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def V0(self) -> float:
        """
        Freestream (far-field) velocity.
        """
        raise NotImplementedError


# ===========================================================================

class DiscreteProp(Propeller, ABC):
    """
    Extension of `Propeller` adding attributes common to propellers that
    are defined by a series of discrete radial stations.
    """

    # -- Public Methods -----------------------------------------------------

    @property
    def AF(self) -> float:
        """
        Blade Activity Factor (AF) for one blade, computed using:
            AF = (100000 / 16) ∫(c/D).x³ dx

        For most normal propellers AF <= 140, and a practical upper limit
        is AF <= 200 to 230.

        Integration is done over all `x` stations, however it is should be
        noted that a number of earlier references ignore a small section of
        blade root for convenience, only integrating outwards from `x` =
        0.15 or `x` = 0.20. The difference is usually negligible.
        Integration is done via Simpson's rule.

        .. Note:: See the Total Activity Factor property `TAF` for the
                  activity factor of the entire propeller.
        """
        return (100000 / 16) * simpson((self.c() / self.D) * self.x() ** 3,
                                       self.x(), even='last')

    @property
    def B(self) -> int:
        """Number of blades."""
        raise NotImplementedError

    @abstractmethod
    def β(self, *, r: [float] = None, x: [float] = None) -> NDArray:
        """
        Propeller total pitch angle (to chordline) at various radial
        station/s depending on the arguments supplied.

        .. Note:: `β` refers to the pitch angle to the chord line - the angle
                  between the aerofoil chord and the disc.  Many propeller
                  texts use the aerodynamic pitch instead - the angle
                  between the zero lift line and the disc (ZLA), but this
                  is *not* the case here.  If required, adjust `β` values to
                  compensate for the ZLA.

        Returns
        -------
        β : ndarray
            Result depends on keyword arguments:

                - No arguments:  Returns an `ndarray, shape (nr,)` of the
                  total pitch angles `β` at all radial stations.

                  .. Note:: Handling of this argument needs to be
                            implemented by derived classes.

                - `r=[r0, r1, ...]`: Returns an `ndarray` giving the
                  total pitch angles `β` at the given radial stations
                  [β1, β2, ...].

                  .. Note:: Handling of this argument needs to be
                            implemented by derived classes.

                - `x=[x0, x1, ...]: Returns an `ndarray` giving the
                  total pitch angles `β` at the requested radius fractions.
                  Implemented as ``r=np.asarray(x)*self.R)``.
        """
        if x is None:
            if r is None:
                # No arguments.
                raise NotImplementedError("Derived class to implement.")
            else:
                # 'r' only argument.
                raise NotImplementedError("Derived class to implement 'r='")

        else:
            if r is None:
                # 'x' only argument.
                return self.β(r=np.asarray(x) * self.R)
            else:
                # Both arguments.
                raise TypeError("'x=' and 'r=' cannot be used together.")

    def c(self) -> NDArray:
        """
        Returns
        -------
        c : (nr,) ndarray
            Propeller chord at each radial station [c0, c1, ...].
        """
        raise NotImplementedError

    @property
    def TAF(self) -> float:
        """
        Total Activity Factor, defined as: TAF = B.AF.  This is the
        activity factor for the entire propeller.

        See blade activity factor property `AF` for more details.
        """
        return self.B * self.AF

    @property
    def nr(self) -> int:
        """Number of radial stations."""
        raise NotImplementedError

    def p(self) -> NDArray:
        """
        Returns
        -------
        p : (nr,) ndarray
            Pitch (length units) at each radial station.  This is
            computed using `p = 2πr.tan(β)`.

            .. Note:: The `β` value refers to the total pitch angle at each
                      station.
        """
        return 2 * np.pi * self.r() * np.tan(self.β())

    def r(self) -> NDArray:
        """
        Returns
        -------
        x : (nr,) ndarray
           Radial analysis stations [r0, r1, ...].
        """
        raise NotImplementedError

    @property
    def R(self) -> float:
        return self.r()[-1]

    @property
    def R_root(self) -> float:
        return self.r()[0]

    @property
    def σ(self) -> float:
        """
        Propeller solidity, defined as σ = (B/Ap).∫c dr.

        Integration is done via Simpson's Rule which requires an odd number
        of points.  If the propeller has an even number then trapezoidal
        integration is done at the blade root, which allows for more
        accurate at the tip where this is desirable.
        """
        return (self.B / self.Ap) * simpson(self.c(), self.r(), even='last')

    def x(self) -> NDArray:
        """
        Returns
        -------
        x : (nr,) ndarray
            Radial fractions where `x = r/R`.  At the tip `x` = 1.0.
        """
        return self.r() / self.R

# ===========================================================================
