"""
This module contains functions relating to propellers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from operator import is_not
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from pyavia.util import count_op
from pyavia.util.print_styles import rad2str
from pyavia.state import State, InvalidStateError

# Written by Eric J. Whitney, January 2023.

_T = TypeVar('_T')
_SclArray = npt.NDArray[_T] | _T
r2d = np.rad2deg


# ======================================================================

class Propeller(State, ABC):
    """
    Abstract type defining the characteristics shared by all basic types
    of propeller.  `Propeller` objects are *stateful*, meaning that
    the value of any performance properties is dependent on the current
    object state.  See `pyavia.states` for more information.

    Parameters
    ----------
    B : int, ∞ or None
        Number of blades:

            - If the model *includes* discrete blade effects use integer
              `B` >= 1.
            - If the model assumes an infinite number of blades use
              ``B=np.inf``.
            - If the model *ignores* blade effects use ``B=None`` (the
              default).

    β0_range : (float, float), optional
        Range of `β0` values `(min, max)` (in *radians*) for a variable
        pitch propeller or `None` (default) for a fixed pitch propeller.

    V0 : float, optional
        Freestream (far upstream) velocity (see `Notes`).

    J : float, optional
        Advance ratio. :math:`J=V/(nD)`  (see `Notes`).

        .. note::If `Ω`, `n` or `RPM` are zero then `J` will return
           `NaN`.

    Ω : float, optional
        Rotational speed `Ω` in *radians/s* (see `Notes`).

    n : float, optional
        Rotational speed in rev/s :math:`n=Ω/(2π)` (see `Notes`).

    RPM : float, optional
        Rotational speed in rev/min (see `Notes`).

    β0 : float, optional
        Collective incidence of blade (in *radians*, see `Notes`).  `βr`
        is added to this along the blade to give total incidence angle
        :math:`β(r) = β_0(r) + β_r(r)`.  If `β0` is not provided then it
        is set to the following values:
            - If `β0_range` was not provided, `β0` = 0.0.
            - If `β0_range` was provided, `β0` is set to the middle of
              the range i.e. ``β0 = 0.5*(β0_range[0] + β0_range[1])``.

    ρ0 : float
        Freestream air density (:math:`ρ_0 > 0`) (see `Notes`).

    a0 : float, default = ∞
        Freestream speed of sound (:math:`a_0 > 0`)  (see `Notes`).
        The default is ∞ giving incompressible flow.

    μ0 : float, default = 0
        Freestream viscosity (:math:`μ_0 >= 0`)  (see `Notes`).  The
        default is zero giving inviscid flow (Re = ∞).

    Notes
    -----
    -  `V0`, `J`, `Ω`, `n`, `RPM`, `ρ0`, `a0`, `μ0` and `β0` are input
       attributes that define the operating state.  These must be
       supplied in certain combinations to give a valid operating state,
       see `set_states(...)` for details.

    - The following subscripts are used to help standardise variable
      names that represent stages of flow through a propeller:

        - 0 - Freestream (or far upstream) properties.
        - 2 - Immediately ahead of the disc.
        - p - At the disc, relative to propeller (moving frame).
        - 3 - Immediately behind the disc.
        - ∞ - Distant wake (or far downstream) of the propeller.

    - If velocities / forces / etc are broken into components, the
      following suffixes are generally used:

        - `a` - Axial.
        - `t` - Tangential (`θ` direction).
        - `r` - Radial.

    - Any units can be used provided they are consistent.  In
      particular this normally means that angles are taken to be
      radians, and rotational velocities `Ω` are radians/s.  No
      internal corrections for degrees, RPM, etc are presently made.

    Raises
    ------
    InvalidStateError
        If the supplied operating state is invalid.
    """

    def __init__(self, *,
                 # Constants.
                 B: float = np.nan,
                 β0_range: tuple[float, float] = None,
                 # Initial operating state.
                 V0: float = None, J: float = None,
                 Ω: float = None, n: float = None, RPM: float = None,
                 β0: float = None,
                 # Initial freestream state.
                 ρ0: float = None, a0: float = np.inf, μ0: float = 0,
                 **kwargs):
        """
        .. note:: `Propeller.__init__` sets only a subset of internal
           states - only those managed by this class - based on keyword
           arguments. Derived classes are responsible for setting the
           remaining states managed by them.

           After construction, the entire object should either be in a
           valid state or raise an `InvalidStateError` exception.
        """
        super().__init__(**kwargs)

        # Set number of blades.
        if np.isfinite(B):
            if not B.is_integer():
                raise ValueError(f"B must be an integer if finite, got "
                                 f"B={B}.")
            elif B <= 0:
                raise ValueError(f"Require B > 0, got B={B}.")

            self._B = int(B)

        else:
            if B == -np.inf:
                raise ValueError(f"Non-finite B must be NaN or +Inf.")

            self._B = B

        # Set β0 range and β0 (where given).
        if β0_range is not None:
            self._β0_range = (min(β0_range), max(β0_range))
            if β0 is None:
                β0 = 0.5 * (self._β0_range[0] + self._β0_range[1])
        else:
            self._β0_range = None
            if β0 is None:
                β0 = 0.0

        # Initial values are set to np.nan for type checking and to
        # ensure all required states are set.
        self._V0, self._Ω = np.nan, np.nan
        self._ρ0, self._a0, self._μ0 = np.nan, np.nan, np.nan
        self._β0 = np.nan

        # Set and check state inputs at this level.
        self._propeller_set_state(V0=V0, J=J, Ω=Ω, n=n, RPM=RPM,
                                  ρ0=ρ0, a0=a0, μ0=μ0, β0=β0)

    # -- Public Methods ------------------------------------------------

    @property
    @abstractmethod
    def AF(self) -> float:
        """
        Blade Activity Factor (AF) for one blade.  Computed from
        AF = (100000 / 16) ∫(c/D).x³ dx

        Notes
        -----
        - For most normal propellers AF <= 140, and a practical upper
          limit is AF <= 200 to 230.
        - Integration is normally done from `R_root` to `R`, however it
          should be noted that a number of earlier references ignore a
          small section of blade root for convenience, only integrating
          outwards from `x` = 0.15 or `x` = 0.20.
        - See the Total Activity Factor property `TAF` for the activity
          factor of the entire propeller.
        - Returned value depends on `B`:
            * For `B` = 0, this returns 0.
            * For non-finite number of blades i.e. `B` = `NaN` or
              `B` = ∞, this returns `NaN`.
            * For a finite number of blades `B` > 0, the base method
              does not provide an integration method and raises
              `NotImplementedError`. Derived classes should supply the
              return value for this case.
        """
        if not np.isfinite(self._B):
            return np.nan
        elif self._B == 0:
            return 0
        else:
            raise NotImplementedError

    @property
    def Ap(self) -> float:
        """
        Disc area.  Computed from :math:`A_p = π(R - R_{root})^2`.
        """
        return np.pi * (self.R - self.R_root) ** 2

    @property
    def a0(self) -> float:
        return self._a0

    @property
    def B(self) -> float:
        return self._B

    @abstractmethod
    def c(self, r: npt.ArrayLike) -> _SclArray[float]:
        """
        Returns the blade chord at one or more given radial station/s
        `r`.
        """
        raise NotImplementedError

    @property
    def CP(self) -> float:
        """
        Power coefficient (propeller type).  Computed from
        :math:`C_P = P_{shaft} / (ρ n^3 D^5)`.
        """
        return self.Ps / (self._ρ0 * self.n ** 3 * self.D ** 5)

    @property
    def CQ(self) -> float:
        """
        Torque coefficient (propeller type).  Computed from
        :math:`C_Q = C_P / (2π)`.
        """
        return self.CP / (2 * np.pi)

    @property
    def CT(self) -> float:
        """
        Thrust coefficient (propeller type).  Computed from
        :math:`C_T = T / (ρ n^2 D^4)`.
        """
        return self.T / (self._ρ0 * self.n ** 2 * self.D ** 4)

    @property
    def D(self) -> float:
        """Tip / overall diameter.  Computed from D = 2.R."""
        return self.R * 2

    @property
    def FOM(self) -> float:
        r"""
        Figure of Merit (propeller type).  Computed from
        :math:`FM = \sqrt{2/π}.C^{3/2}_T / C_P`.
        """
        return np.sqrt(2.0 / np.pi) * (self.CT ** 1.5) / self.CP

    def get_state(self) -> dict[str, Any]:
        """
        Returns the state of the object as a dict with the containing
        the primary values 'V0', 'Ω', 'β0', 'ρ0', 'a0', and 'μ0' as
        keys.
        """
        return super().get_state() | {
            'V0': self._V0, 'Ω': self._Ω, 'β0': self._β0,  # Oper. state
            'ρ0': self._ρ0, 'a0': self._a0, 'μ0': self._μ0  # Freestream
        }

    @property
    def J(self) -> float:
        """
        Returns the advance ratio :math:`J=V/(nD)`. If `Ω`, `n` or
        `RPM` are zero then `J` will return `NaN`.
        """
        if self.n != 0:
            return self._V0 / (self.n * self.D)
        else:
            return np.nan

    @property
    def n(self) -> float:
        return self._Ω / (2 * np.pi)

    def p(self, r: npt.ArrayLike) -> _SclArray[float]:
        """
        Returns pitch (in length units) at one or more radial station/s
        `r`. This is computed using `p = 2πr.tan(β)`.

        .. note:: If no pitch information is available (i.e. `β` is
           `NaN`) then `p` = `NaN`.
        """
        return 2 * np.pi * r * np.tan(self.β(r))

    @property
    @abstractmethod
    def Ps(self) -> float:
        """
        Shaft power.  The sign convention used gives a positive (+)
        value when the propeller is being driven and negative (-) in
        windmill state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def Qs(self) -> float:
        """
        Shaft torque.  The sign convention used gives a positive (+)
        value when the propeller is being driven and negative (-) in
        windmill state.
        """
        raise NotImplementedError

    @property
    def RPM(self) -> float:
        return self._Ω * 30 / np.pi

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
        represents spinners or an innermost extent of the disc.
        """
        raise NotImplementedError

    def set_state(self, *,
                  # Operating state.
                  V0: float = None, J: float = None,
                  Ω: float = None, n: float = None, RPM: float = None,
                  β0: float = None,
                  # Freestream state.
                  ρ0: float = None, a0: float = None, μ0: float = None,
                  **kwargs) -> frozenset[str]:
        """
        Set propeller operating state using groups of related input
        values.  Only a single value from each state group (primary or
        optional) may be selected as shown below:

        +----------------------+---------------------------+
        |      State Group     |                           |
        +---------+------------+          Affects          |
        | Primary |  Optional  |                           |
        +=========+============+===========================+
        |  `V0`   |    `J`     | Freestream Airspeed       |
        +---------+------------+---------------------------+
        |   `Ω`   | `n`, `RPM` | Rotational Speed          |
        +---------+------------+---------------------------+
        |  `β0`   |            | Collective Incidence      |
        +---------+------------+---------------------------+
        |  `ρ0`   |            | Freetream Density         |
        +---------+------------+---------------------------+
        |  `a0`   |            | Freestream Speed of Sound |
        +---------+------------+---------------------------+
        |  `μ0`   |            | Freestream Viscosity      |
        +---------+------------+---------------------------+

        Returns
        -------
        frozenset[str]
            Set of primary input attribute names that were changed.

        Raises
        ------
        InvalidStateError
            If inputs have incorrect values.

        Notes
        -----
        - If `set_state` is overridden, it should chain up with this
          method or otherwise ensure `_post_set_state` is called.
        """
        changed = super().set_state(**kwargs)
        changed |= self._propeller_set_state(
            V0=V0, J=J, Ω=Ω, n=n, RPM=RPM, β0=β0, ρ0=ρ0, a0=a0, μ0=μ0)
        return changed

    @property
    def stopped(self) -> bool:
        """
        Returns `True` if the propeller is not turning (`Ω` < 0.01
        rad/s (0.6 °/s) ≈ 0). This property can be used to avoid
        running flow solutions in these cases, etc.
        """
        return True if abs(self.Ω) < 0.01 else False

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
    def TAF(self) -> float:
        """
        Total Activity Factor, defined as: TAF = B.AF.  This is the
        activity factor for the entire propeller.  See blade activity
        factor property `AF` for more details.

        .. note:: If number of blades `B` is not finite / undefined,
           this returns `NaN`.
        """
        if not np.isfinite(self._B):
            return np.nan

        return self._B * self.AF

    @property
    def V0(self) -> float:
        return self._V0

    def β(self, r: npt.ArrayLike) -> _SclArray[float]:
        """
        Returns the total blade incidence at one or more radial
        station/s `r` along the blade.  This is the sum of the
        collective incidence and the local variation :math:`β(r) =
        β_0(r) + β_r(r)`.

        Notes
        -----
        - `β` refers to the incidence or pitch angle - the angle
          between the aerofoil chordline and the disc.  Many propeller
          texts use the aerodynamic pitch instead - the angle between
          the zero lift line and the disc (ZLA), but this is *not* the
          case here.  If required, adjust `β` values to compensate for
          the ZLA.

        - Returns `NaN` if no incidence information is available.
        """
        return self._β0 + self.βr(r)

    @property
    def β0(self) -> float:
        return self._β0

    @property
    def β0_range(self) -> tuple[float, float] | None:
        return self._β0_range

    @abstractmethod
    def βr(self, r: npt.ArrayLike) -> _SclArray[float]:
        """
        Returns the component of the pitch angle [radians] that varies
        locally along the blade at one or more radial station/s `r`.
        This is added to the collective incidence `β0` along the blade
        to give the total incidence angle `β0`.
        """
        raise NotImplementedError

    @property
    def η(self) -> float:
        """
        Propeller efficiency.  Computed from :math:`η = J.C_T / C_P`.
        In the case that CP == 0 then the propeller efficiency is given
        as +∞ if `C_T > 0` and -∞ if `C_T < 0`.
        """
        if self.CP != 0:
            return self.J * self.CT / self.CP
        else:
            return np.sign(self.CT) * np.inf

    @property
    def μ0(self) -> float:
        return self._μ0

    @property
    def ρ0(self) -> float:
        return self._ρ0

    @property
    def σ(self) -> float:
        """
        Propeller solidity, defined as σ = (B/Ap).∫c dr.

        Notes
        -----
        - Returned value depends on `B`:
            * For `B` = 0, this returns 0.
            * For non-finite number of blades i.e. `B` = `NaN` or
              `B` = ∞, this returns `NaN`.
            * For a finite number of blades `B` > 0, the base method
              does not provide an integration method and raises
              `NotImplementedError`. Derived classes should supply the
              return value for this case.
        """
        if not np.isfinite(self._B):
            return np.nan
        elif self._B == 0:
            return 0
        else:
            raise NotImplementedError

    @property
    def Ω(self) -> float:
        return self._Ω

    # -- Private Methods -----------------------------------------------

    def _propeller_set_state(
            self, *,  # No defaults checks for missing **kwargs.
            V0: float | None, J: float | None, Ω: float | None,
            n: float | None, RPM: float | None, β0: float | None,
            ρ0: float | None, a0: float | None, μ0: float | None
    ) -> frozenset[str]:
        # Internal implementation of `set_state` that sets, checks and
        # actions all state variables for only this level in the class
        # hiearchy.
        changed = frozenset()

        # Future work: Convert the shamozzle below to a match statement.

        # -- Rotation Speed --------------------------------------------
        # Set first, as J requires it later (if used).

        Ω_args = count_op((Ω, n, RPM), is_not, None)
        if Ω_args > 1:
            raise InvalidStateError("Only one argument from Ω, n, RPM "
                                    "allowed.")
        elif Ω_args == 1:
            if n is not None:
                Ω = 2.0 * np.pi * n  # Convert 'n' to 'Ω'.

            elif RPM is not None:
                Ω = RPM * np.pi / 30.0  # Convert 'RPM' to 'Ω'.

        if (Ω is not None) and (Ω != self._Ω):
            changed |= {'Ω'}
            self._Ω = Ω

        if np.isnan(self._Ω):
            #  'Ω' can be +ve, -ve or zero.
            raise InvalidStateError(f"Invalid Ω = {self._Ω}.")

        # -- Airspeed --------------------------------------------------

        if J is not None:
            if V0 is not None:
                raise InvalidStateError("Only one argument from 'V0' "
                                        "or 'J' allowed.")
            V0 = J * self.n * self.D  # Convert 'J' to 'V0'.

        if (V0 is not None) and (V0 != self._V0):
            changed |= {'V0'}
            self._V0 = V0

        if np.isnan(self._V0):
            # 'V0' can be +ve, -ve or zero.
            raise InvalidStateError(f"Invalid V0 = {self._V0}.")

        # -- Collective Incidence --------------------------------------

        if (β0 is not None) and (β0 != self._β0):
            self._β0 = β0
            changed |= {'β0'}

        # The 'if not x > y' constructions below also check for NaN.
        if not (self._β0_range[0] <= self._β0 <= self._β0_range[1]):
            raise InvalidStateError(
                f"β0 = {rad2str(self._β0)} out of range "
                f"{rad2str(self._β0_range[0])} -> "
                f"{rad2str(self._β0_range[1])}.")

        # -- Density ---------------------------------------------------

        if (ρ0 is not None) and (ρ0 != self._ρ0):
            self._ρ0 = ρ0
            changed |= {'ρ0'}

        if not (self._ρ0 > 0):
            raise InvalidStateError(f"Invalid ρ0 = {self._ρ0} <= 0.")

        # -- Speed of Sound --------------------------------------------

        if (a0 is not None) and (a0 != self._a0):
            self._a0 = a0
            changed |= {'a0'}

        if not (self._a0 > 0):
            raise InvalidStateError(f"Invalid a0 = {self._a0} <= 0.")

        # -- Viscosity -------------------------------------------------

        if (μ0 is not None) and (μ0 != self._μ0):
            self._μ0 = μ0
            changed |= {'μ0'}

        if not (self._μ0 >= 0):
            raise InvalidStateError(f"Invalid μ0 = {self._μ0} < 0.")

        return changed

# ----------------------------------------------------------------------


# Future work: Add a plot function for plotting properties along the
# blade.

# ======================================================================


def const_pitch_β(β_x75: float, x: npt.ArrayLike) -> npt.NDArray[float]:
    """
    Compute total pitch angle `β` along a propeller blade at the given
    radial stations `x` that would give a constant pitch using the `β`
    value given at x=75%.

    Parameters
    ----------
    β_x75 : float
        Pitch angle at x=0.75.

    x : array_like, shape (n,)
        Radial stations along the blade, given as `x = r/R`.

    Returns
    -------
    ndarray, shape (n,)
        Total pitch angle along the blade corresponding to `x`.
    """
    return np.arctan(np.tan(β_x75) * 0.75 / x)

# ======================================================================
