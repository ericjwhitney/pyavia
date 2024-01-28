"""
This module contains functions relating to propellers.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

import pyavia.states as states
from pyavia.states import States


# Written by Eric J. Whitney, January 2023.


# ===========================================================================

# TODO Rename?
class BasePropeller(States, ABC):
    """
    Basic definition of characteristics shared by all types of propeller
    model.

    TODO Propellers are stateful objects

    Notes
    -----

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
    RPM, etc are made.
    """
    def __init__(self, V0: float = None, J: float = None,
                 Ω: float = None, n: float = None, RPM: float = None,
                 ρ0: float = None):
        """
        For the meaning of parameters and which can be used together, see
        `set_states`.

        Notes
        -----
        If any of the optional state variables are omitted (`V0`, `Ω`,
        etc) then these are set to `NaN`.
        """
        # Primary internal states are V0, ρ0, Ω and are set to NaN to
        # prevent accidental errors.
        self._V0, self._Ω, self._ρ0 = np.nan, np.nan, np.nan
        self._set_prop_states(V0=V0, J=J, Ω=Ω, n=n, RPM=RPM, ρ0=ρ0)

    # -- Public Methods -----------------------------------------------------

    @classmethod
    def def_states(cls) -> frozenset[str]:
        return frozenset(['V0', 'Ω', 'ρ0'])

    @classmethod
    def input_states(cls) -> frozenset[str]:
        return frozenset(['J', 'n', 'Ω', 'RPM', 'ρ0'])

    @classmethod
    def output_states(cls) -> frozenset[str]:
        # TODO CHECK
        return frozenset(['CP', 'CQ', 'CT', 'η', 'FM', 'Ps', 'Qs', 'T'])

    def set_states(self, *, V0: float = None, J: float = None,
                   Ω: float = None, n: float = None, RPM: float = None,
                   ρ0: float = None) -> frozenset[str]:
        """
        Set propeller operating state using core or optional states.  Only
        a single value from each state group (defining or optional) may be
        selected as shown below:

        +-----------------------+---------------------+
        |    State Group        |                     |
        +----------+------------+       Affects       |
        | Defining |  Optional  |                     |
        +==========+============+=====================+
        |   `V0`   |    `J`     | Freestream Airspeed |
        +----------+------------+---------------------+
        |   `Ω`    | `n`, `RPM` | Rotational Speed    |
        +----------+------------+---------------------+
        |   `ρ0`   |            | Freetream Density   |
        +----------+------------+---------------------+

        Returns
        -------
        frozenset[str]
            Set of states that were changed.

        Raises
        ------
        ValueError
            If arguments have incorrect values.
        """
        return self._set_prop_states(V0=V0, J=J, Ω=Ω, n=n, RPM=RPM, ρ0=ρ0)

    @property
    @abstractmethod
    def Qs(self) -> float:
        """
        Shaft torque.  The sign convention used gives a positive (+) value
        when the propeller is being driven and negative (-) in windmill state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def Ps(self) -> float:
        """
        Shaft power.  The sign convention used gives a positive (+) value
        when the propeller is being driven and negative (-) in windmill
        state.
        """
        raise NotImplementedError

    @property
    def D(self) -> float:
        """
        Tip / overall diameter.  Computed from D = 2.R."""
        return self.R * 2

    @property
    def FM(self) -> float:
        r"""
        Figure of Merit (propeller type).  Computed from :math:`FM =
        \sqrt{2/π}.C^{3/2}_T / C_P`.
        """
        return np.sqrt(2.0 / np.pi) * (self.CT ** 1.5) / self.CP

    @property
    def Ω(self) -> float:
        """
        Rotational speed `Ω` in *radians/s*.
        """
        return self._Ω

    @property
    def J(self) -> float:
        """
        Advance ratio.  Computed from J = V / (n.D).
        """
        return self.V0 / (self.n * self.D)

    @property
    def CP(self) -> float:
        """
        Power coefficient (propeller type).  Computed from :math:`C_P =
        P_{shaft} / (ρ n^3 D^5)`.
        """
        return self.Ps / (self.ρ0 * self.n ** 3 * self.D ** 5)

    @property
    def RPM(self) -> float:
        """
        Returns
        -------
        RPM : float
            Rotational speed in rev/min.
        """
        return self.Ω * 30 / np.pi

    @property
    def n(self) -> float:
        """
        Returns
        -------
        n : float
            Rotational speed in rev/s, computed from n = Ω / (2π).
        """
        return self.Ω / (2 * np.pi)

    @property
    def CQ(self) -> float:
        """
        Torque coefficient (propeller type).  Computed from :math:`C_Q = C_P
        / (2π)`.
        """
        return self.CP / (2 * np.pi)

    @property
    def ρ0(self) -> float:
        """Freestream air density (:math:`ρ_0`)."""
        return self._ρ0

    @property
    @abstractmethod
    def R_root(self) -> float:
        """
        Returns
        -------
        R_root : float
            Root (hub, smallest working) radius.  If non-zero this
            represents spinners or an inner part of the disc that has no
            effect on performance.
        """
        raise NotImplementedError

    @property
    def Ap(self) -> float:
        """
        Disc area.  Computed from :math:`A_p = π(R - R_{root})^2`.
        """
        return np.pi * (self.R - self.R_root) ** 2

    @property
    def CT(self) -> float:
        """
        Thrust coefficient (propeller type).  Computed from :math:`C_T = T /
        (ρ n^2 D^4)`.
        """
        return self.T / (self.ρ0 * self.n ** 2 * self.D ** 4)

    @property
    @abstractmethod
    def R(self) -> float:
        """
        Tip / overall radius.
        """
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
    def η(self) -> float:
        """
        Propeller efficiency.  Computed from :math:`η = J.C_T / C_P`.
        """
        return self.J * self.CT / self.CP

    @property
    def V0(self) -> float:
        """
        Returns
        -------
        V0 : float
            Freestream (far-field) velocity.
        """
        return self._V0

    # -- Private Methods -----------------------------------------------------

    def _set_prop_states(self, V0: float | None, J: float | None,
                         Ω: float | None, n: float | None, RPM: float | None,
                         ρ0: float | None) -> frozenset[str]:
        """
        Common code for `__init__` and `set_states`.  See `set_states` for
        details.
        """
        changed = frozenset()

        # -- Rotation Speed --------------------------------------------------

        # This is set first (as J requires it later if present).
        Ω_args = [True for arg in (Ω, n, RPM) if arg is not None].count(True)
        if Ω_args > 1:
            raise ValueError("Only one argument from Ω, n, RPM allowed.")

        # Update, converting if given as 'n' or 'RPM'.
        if Ω is not None:
            self._Ω = Ω
            changed |= {'Ω'}

        elif n is not None:
            self._Ω = 2.0 * np.pi * n
            changed |= {'n'}

        elif RPM is not None:
            self._Ω = RPM * np.pi / 30.0
            changed |= {'RPM'}

        # -- Airspeed --------------------------------------------------------

        V0_args = [True for arg in (V0, J) if arg is not None].count(True)
        if V0_args > 1:
            raise ValueError("Only one argument from V0, J allowed.")

        # Update, converting if given as 'J'.
        if V0 is not None:
            self._V0 = V0
            changed |= {'V0'}

        elif J is not None:
            self._V0 = J * self.n * self.D  # 'self.n' avail, Ω set earlier.
            changed |= {'J'}

        # -- Atmosphere ------------------------------------------------------

        if ρ0 is not None:
            self._ρ0 = ρ0
            changed |= {'ρ0'}

        # -- Finalise --------------------------------------------------------

        return changed


# ===========================================================================

class BladedPropeller(BasePropeller, ABC):
    """
    Extension of `BasePropeller` to include properties related to discrete
    blades.
    """

    def __init__(self, *args, B: int, **kwargs):
        """
        Parameters
        ----------
        args :
            See `BasePropeller.__init__`.
        B : int
            Number of blades (> 0).
        kwargs :
            See `BasePropeller.__init__`.
        """
        super().__init__(*args, **kwargs)

        self._B = B
        if self._B < 1:
            raise ValueError(f"Require B >= 1, got {self._B}.")

    # -- Public Methods -----------------------------------------------------

    @property
    @abstractmethod
    def AF(self) -> float:
        """
        Blade Activity Factor (AF) for one blade.  Computed from
        AF = (100000 / 16) ∫(c/D).x³ dx

        Notes
        -----
        - For most normal propellers AF <= 140, and a practical upper limit
          is AF <= 200 to 230.
        - Integration is normally done from `R_root` to `R`, however it
          should be noted that a number of earlier references ignore a small
          section of blade root for convenience, only integrating outwards
          from `x` = 0.15 or `x` = 0.20.
        - See the Total Activity Factor property `TAF` for the activity 
          factor of the entire propeller.
        """
        raise NotImplementedError

    @property
    def B(self) -> int:
        """Number of blades."""
        return self._B

    @abstractmethod
    def β(self, r: ArrayLike) -> ArrayLike:
        """
        Returns the total blade pitch angle (to chordline) at one or more
        radial station/s `r`.

        Notes
        -----
        `β` refers to the pitch angle to the chord line - the angle between
        the aerofoil chord and the disc.  Many propeller texts use the
        aerodynamic pitch instead - the angle between the zero lift line and
        the disc (ZLA), but this is *not* the case here.  If required,
        adjust `β` values to compensate for the ZLA.
        """
        raise NotImplementedError

    @abstractmethod
    def c(self, r: ArrayLike) -> ArrayLike:
        """
        Returns the blade chord at one or more given radial station/s `r`.
        """
        raise NotImplementedError

    def p(self, r: ArrayLike) -> ArrayLike:
        """
        Returns pitch (in length units) at one or more radial station/s `r`.
        This is computed using `p = 2πr.tan(β)`.
        """
        return 2 * np.pi * r * np.tan(self.β(r))

    @property
    def σ(self) -> float:
        """
        Propeller solidity, defined as σ = (B/Ap).∫c dr.

        Notes
        -----
        TODO At the base level this integration is done using  
          scipy.integrate.quad
        Integration is done via Simpson's Rule which requires an odd number
        of points.  If the propeller has an even number then trapezoidal
        integration is done at the blade root, which allows for more
        accurate at the tip where this is desirable.
        """
        # TODO
        raise NotImplementedError

    @property
    def TAF(self) -> float:
        """
        Total Activity Factor, defined as: TAF = B.AF.  This is the activity
        factor for the entire propeller.  See blade activity factor property
        `AF` for more details.
        """
        return self.B * self.AF

    # -- Private Methods -----------------------------------------------------


# ===========================================================================


class VariableMixin2(BladedPropeller, ABC):
    """
    Abstract mixin class that adds properties related to discrete radial
    stations to a `BladedPropeller` object.
    """

    # -- Public Methods -----------------------------------------------------

# ============================================================================


# Future work: Add a plot function for plotting properties along the blade.
