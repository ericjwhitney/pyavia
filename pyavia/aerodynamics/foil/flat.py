
import numpy as np

from pyavia.aerodynamics.foil.base import Foil2DBasic


# Written by Eric J. Whitney, February 2023.

# ======================================================================


class Flat2DAero(Foil2DBasic):  # TODO Consider 'FlatPlateFoil'
    """
    Model of generic flat plate foil characteristics, using the
    following assumptions:

    - Plate is sharp edged and flow is separated at all angles of
      attack. There is no significant linear lift regime.

    - ``Re > 1000`` and Reynolds Number effects are ignored.  This is
      typical for all normal flight vehicle situations.  Above this
      value the effect of changes in Reynolds Number are negligible.

    - There is no boundary layer / skin friction drag.

    - Flow is assumed to be incompressible and Mach Number is ignored.

    Notes
    -----
    :math:`c_l` and :math:`c_d` are derived from the normal force
    coefficient :math:`c_n` given in Chapter XXI, Equation 2 of [1]_
    which aligns very well with test data.  :math:`c_l` is derived; see
    ``cm`` for details.

    .. [1] Hoerner, S., "Fluid Dynamic Lift", Second Ed., 1985.
    """

    def __init__(self, **initial_states):
        super().__init__(**initial_states)
        self._cn, self._cnα = np.nan, np.nan  # init only.
        self._plate_forces()

    # -- Public Methods ------------------------------------------------

    @property
    def α0(self) -> float:
        return 0.0

    @property
    def α_stall_pos(self) -> float:
        """Flat plate :math:`c_{l_{max}}` occurs at `α` = +39.26°."""
        return 0.6852946  # [rad]

    @property
    def α_stall_neg(self) -> float:
        """Flat plate :math:`c_{l_{min}}` occurs at `α` = -39.26°."""
        return -0.6852946  # [rad]

    @property
    def cd(self) -> float:
        return self.cn * np.sin(self.α)

    @property
    def cl(self) -> float:
        return self.cn * np.cos(self.α)

    @property
    def clα(self) -> float:
        # Applying d/dα product rule to cl = cn * cos(α).
        return self.cnα * np.cos(self.α) - self.cn * np.sin(self.α)

    @property
    def cm_qc(self) -> float:
        """
        Quarter-chord pitching moment (:math:`c_{m,c/4}`) is computed
        from `cn` assuming it acts at the midpoint of the plate.

        .. note:: In practice flat plate flows are very unsteady so
           actual pitching moment will fluctuate over a wide range.
        """
        return -self.cn * 0.25

    @property
    def cn(self) -> float:
        """Normal force coefficient (:math:`c_n`)."""
        return self._cn

    @property
    def cnα(self) -> float:
        """
        Normal force coefficient `a`-derivative (:math:`c_{na}` or
        :math:`dc_n/da)`).
        """
        return self._cnα

    # @classmethod
    # def output_states(cls) -> frozenset[str]:
    #     return super().output_states() | {'cn', 'cnα'}

    def set_states(self, **kwargs) -> frozenset[str]:
        """
        See `Foil2DBasic.set_states` for available parameters.

        Notes
        -----
        At the current stage of development, the flow is assumed to be
        above the critical Reynolds number and incompressible, i.e.
        `Re` and `M` are ignored.  They remain included for later
        extensions to this class.
        """
        # Chain up to revise operating state.
        changed = super().set_states(**kwargs)

        # Only need to compute plate forces if 'α' changed.
        if 'α' in changed:
            self._plate_forces()

        return changed

    # @property
    # def valid_state(self) -> bool:
    #     """`Flat2DAero` always returns `True`."""
    #     return True

    # -- Private Methods -----------------------------------------------

    def _plate_forces(self):
        """
        Compute normal force coefficients and gradients.  This requires
        `self.α` to be set.
        """
        # Equation for the normal force coefficient with 0 < α <= π/2 is:
        #   cn = 1.0 / [a + b / sin(α)] with a = 0.222, b = 0.283.
        # Note:
        #   - Function is made symmetrical for high angles π/2 < α < π/2
        #     by reversing back down the curve i.e. α' = π - α.
        #   - Sign is reversed and handled separately for negative angles
        #     α < 0.
        #   - For α near zero, values are computed directly.
        #
        # Derivative of cn is (for 0 < α <= π/2 region):
        #   d(cn)/dα = [b.cos(α)] / [a.sin(α) + b]**2
        # High and negative angles are handled following the same method
        # as above.

        α_abs = np.abs(self.α)
        α_sign = np.sign(self.α)
        if α_abs <= 0.5 * np.pi:
            α_slope = +1
        else:
            α_abs = np.pi - α_abs  # Make equiv. α in first quadrant.
            α_slope = -1

        a, b = 0.222, 0.283
        if α_abs > 1e-9:
            # Normal α.
            self._cn = α_sign * 1.0 / (a + b / np.sin(α_abs))
            self._cnα = α_slope * ((b * np.cos(α_abs)) /
                                   (a * np.sin(α_abs) + b) ** 2)
        else:
            # Small α.
            self._cn = 0.0
            self._cnα = α_slope / b

# ======================================================================
