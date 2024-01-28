"""
Top-level definition for foil performance models and general functions.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from pyavia.states import States, InvalidStateError
from pyavia.solve import step_bracket_min
from pyavia.solve.exception import SolverError


# Written by Eric J. Whitney, January 2023.

# ============================================================================

# TODO likely unnecessary, combine with Basic below.
class Foil2DAero(States, ABC):
    """
    Class defining the basic 2D aerodynamic interface common to all
    foils.

    `Foil2DAero` objects are *stateful*, meaning that the value of any
    performance properties is dependent on the current object state.
    See `pyavia.states` for more information.

    Notes
    -----
    If an individual property can't be computed (for whatever reason)
    under the current flow conditions a value of `NaN` is returned,
    even if the object state is valid i.e. ``Foil2DBasic.valid_state ==
    True``.  TODO Review
    """

    # -- Public Methods ------------------------------------------------------

    @property
    @abstractmethod
    def α(self) -> float:
        """Angle of attack (`α`) in range `[-π, +π]`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def α0(self) -> float:
        """Angle of attack for zero lift (:math:`α_0` or `ZLA`)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def α_stall_neg(self) -> float:
        """
        Returns the negative stalling angle of attack
        (below :math:`α_0`) under the *current* flow conditions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def α_stall_pos(self) -> float:
        """
        Returns the positive stalling angle of attack (below :math:`α_0`)
        under the *current* flow conditions.
        """
        raise NotImplementedError

    @property
    def β(self) -> float:
        r"""
        Prantdl-Meyer compressibility factor :math:`β=\sqrt{1 - M^2}`.
        """
        return np.sqrt(1.0 - self.M ** 2)

    @property
    @abstractmethod
    def cd(self) -> float:
        """Drag coefficient (:math:`c_d`)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cl(self) -> float:
        """Lift coefficient (:math:`c_l`)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def clα(self) -> float:  # TODO consider 'cl_α'
        """
        Lift curve slope (:math:`c_{lα} = dc_l/dα`).

        Notes
        -----
        This applies to the *current* angle of attack (i.e. can be
         angle of attack dependent depending on foil model).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def cl_stall_neg(self) -> float:
        """
        Returns the negative stalling lift coefficient (below
        :math:`α_0`) under the *current* flow conditions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def cl_stall_pos(self) -> float:
        """
        Returns the positive stalling lift coefficient (below
        :math:`α_0`) under the *current* flow conditions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def cm_qc(self) -> float:
        """
        Pitching moment coefficient (1/4 chord :math:`c_{m,c/4}`).
        """
        raise NotImplementedError

    # @classmethod
    # def input_states(cls) -> frozenset[str]:
    #     return frozenset()

    @property
    @abstractmethod
    def M(self) -> float:
        """Mach Number (`M`)."""
        raise NotImplementedError

    # @classmethod
    # def output_states(cls) -> frozenset[str]:
    #     """
    #     Output (readonly) states from base level `Foil2DAero` class.
    #
    #     Notes
    #     -----
    #     Derived classes must remove any output states that become input
    #     states.
    #     """
    #     return frozenset(['α', 'α0', 'α_stall_neg', 'α_stall_pos', 'β', 'cd',
    #                       'cl', 'clα', 'cl_stall_neg', 'cl_stall_pos',
    #                       'cm_qc', 'M', 'Re'])

    @property
    @abstractmethod
    def Re(self) -> float:
        """Reynolds Number (`Re`)."""
        raise NotImplementedError

# ======================================================================


class Foil2DBasic(Foil2DAero, ABC):
    r"""
    Class defining the basic 2D aerodynamic properties typical for most
    foils.  This adds Reynolds number (`Re`), Mach number (`M`) and
    Angle of Attack (`α`) to `Foil2DAero`.  Most require at least these
    values to be able to give unique results for :math:`c_l`,
    :math:`c_d`, etc.

    Parameters
    ----------
    Re : float, default = ∞
        Initial Reynold's number in range `(0, ∞]`.
    M : float, default = 0.0
        Initial Mach number in range `[0, ∞]`.
    α : float, default = 0.0
        Initial angle of attack.  This is normalised to the range of
        :math:`α \in [-π, +π]`

    TODO Properties here?
    """

    def __init__(self, *, Re: float = np.inf, M: float = 0.0,
                 α: float = 0.0):
        super().__init__()
        self._Re, self._M, self._α = np.nan, np.nan, np.nan  # init only
        self._set_ReMα(Re, M, α)

    # -- Public Methods ------------------------------------------------

    @property
    def α(self) -> float:
        return self._α

    @property
    def α_stall_neg(self) -> float:
        """
        Returns the negative stalling angle of attack (below
        :math:`α_0`) under the *current* flow conditions.  For more
        details see property ``α_stall_pos``.
        """
        return self._find_stall(-1)

    @property
    def α_stall_pos(self) -> float:
        """
        Returns
        -------
        float
            Positive stalling angle of attack (above :math:`α_0`).

        Raises
        ------
        SolverError
            If the lift peak could not be bracketed / converged.

        Notes
        -----
        For `Foil2DBasic` this algorithm works by first stepping
        progessively to bracket the *first* lift peak above/below
        :math:`α_0`.  After bracketing, the angle of attack is optimised
        to give the peak value:

        - This is done every time this property is called, so if
          specific implementations use this value repeatedly, it should
          be cached.
        - The algorithm is not guaranteed to always find the first lift
          peak if the curve is particularly noisy or complex.
        """
        return self._find_stall(+1)

    @property
    def cl_stall_neg(self) -> float:
        """
        Returns the negative stalling lift coefficient (below
        :math:`α_0`) under the *current* flow conditions. See
        `cl_stall_pos` for more detail.
        """
        restore = self.get_states()
        self.set_states(α=self.α_stall_neg)
        val = self.cl
        self.set_states(**restore)
        return val

    @property
    def cl_stall_pos(self) -> float:
        """
        Returns the positive stalling lift coefficient (above :math:`α_0`)
        under the *current* flow conditions.

        Notes
        -----
        For `Foil2dBasic` this simply returns the `cl` property after
        setting `α` = `α_stall_pos`.
        """
        restore = self.get_states()
        self.set_states(α=self.α_stall_pos)
        val = self.cl
        self.set_states(**restore)
        return val

    # @classmethod
    # def input_states(cls) -> frozenset[str]:
    #     return frozenset(['Re', 'M', 'α'])

    def get_states(self) -> dict[str, Any]:
        return {'Re': self._Re, 'M': self._M, 'α': self._α}

    @property
    def M(self) -> float:
        return self._M

    # @classmethod
    # def output_states(cls) -> frozenset[str]:
    #     return super().output_states() - {'Re', 'M', 'α'}

    @property
    def Re(self) -> float:
        return self._Re

    def set_states(self, *, Re: float = None, M: float = None,
                   α: float = None) -> frozenset[str]:
        r"""
        Sets the operating state via `Re`, `M` and `α` optional
        arguments. This method is intended to be overridden by specific
        implementations, which should chain-up with this method. This
        (superclass) method checks should be called *prior* to any
        further computations so that it can set and check `Re`, `M`
        and `α` in advance.

        Parameters
        ----------
        Re : float, optional
            Reynold's number in range `(0, ∞]`.
        M : float, optional
            Mach number in range `[0, ∞]`.
        α : float, optional
            Angle of attack.  This is normalised to the range of
            :math:`α \in [-π, +π]`

        Returns
        -------
        frozenset[str]
            Set of state input names that have changed.

        Raises
        ------
        InvalidStateError
            Illegal Reynolds or Mach Numbers.
        """
        return self._set_ReMα(Re, M, α)

    # -- Private Methods -----------------------------------------------------

    def _find_stall(self, side: int) -> float:
        """
        Finds the positive (``side = 1``) or negative (``side = -1``)
        stalling angle; see ``α_stall_pos`` for more details.  Result is
        only returned between an angle of `α0` and +/-90° (depending on
        side), otherwise returns `NaN`..
        """
        # Define function to minimise as min(-cl(α)) for α > 0 or
        # min(cl(α)) for α < 0.
        def cl_fn(α: float) -> float:
            self.set_states(α=α)
            return -side * self.cl

        with self.restore_states():
            # First approximately bracket the peak cl. Setup starting
            # bracket of (α0,  α0 +/- 5°).
            try:
                x1, x2 = step_bracket_min(
                    cl_fn, x1=self.α0, x2=self.α0 + side * 0.08727,
                    x_limit=np.sign(side) * 0.5 * np.pi)

            except SolverError as e:
                # Re-raise with more helpful information.
                raise SolverError(f"Failed to bracket the peak cl - "
                                  f"{e.details}") from e

            # Finally converge to the peak cl.
            res = minimize_scalar(cl_fn, bracket=(x1, x2),
                                  method='brent', tol=1e-3,
                                  options={'maxiter': 50})

            if not res.success:
                raise SolverError("Failed to converge α for peak cl.")

        # Check result sense.
        α_stall = res.x
        if np.sign(α_stall - self.α0) != np.sign(side):
            raise SolverError("Invalid stall angle found.")

        return α_stall

    def _set_ReMα(self, Re: float | None, M: float | None,
                  α: float | None) -> frozenset[str]:
        """Common code for __init__ and set_states."""
        changed = frozenset()

        # Update Reynolds Number.
        if Re is not None and self._Re != Re:
            if Re <= 0.0:
                raise InvalidStateError(f"Invalid Re = {Re} (require > 0.")
            self._Re = Re
            changed |= {'Re'}

        # Update Mach Number.
        if M is not None and self._M != M:
            if M < 0.0:
                raise InvalidStateError(f"Invalid M = {M} (require >= 0).")
            self._M = M
            changed |= {'M'}

        # Update AoA.
        if α is not None:
            α = std_α(α)
            if self._α != α:
                self._α = α
                changed |= {'α'}

        return changed


# ===========================================================================

# Future Improvement: α arguments can also be arrays and we can add Re,
# M trends aligned to these as well.

def plot_foil_aero(foils: list[Foil2DAero] | Foil2DAero,
                   α_start: float, α_end: float, num: int = 50, *,
                   figs: [[str, ...], ...] = (('α', 'cl', 'cm_qc'),
                                              ('cl', 'cd')),
                   title: str = "", block: bool = True,
                   colours: [str] = ('b', 'r', 'g', 'k'),
                   labels: [str] = ('1', '2', '3', '4'),
                   **state_kwargs):
    """
    Plot foil performance.  This function requires ``matplotlib.pyplot``.

    .. note:: Input states set before calling this method are restored on
       return.  Other input states (often `α`) are left 'as is', i.e.
       the last value used by the method.

    Parameters
    ----------
    TODO
    α_start : float
        Start angle of attack.
    α_end : float
        Ending angle of attack.
    num : int, default = 50
        Number of angles of attack to compute.
    figs : [(str, ...), ...], optional
        Sequence of figures to generate.  Each figure is defined by a
        sequence of properties (`x`, `y1`, ...) that are all drawn on
        the same axes. The default is ``plot_props=(('α', 'cl',
        'cm_qc'), ('cl', 'cd')`` which plots :math:`c_l` and
        :math:`c_{m, 1/4/c}` vs. `α` on the first plot and :math:`c_d`
        and  :math:`c_l` on the next plot.

    title : str, optional
        Title to add to all plots.

    clα_vs_α : bool, default = False
        If `True` also plot lift curve slop :math:`c_{l_α}` vs. `α`.
    cd_vs_α : bool, default = False
        In addition to plotting :math:`c_d` vs :math:`c_l`, if `True` also
        plot :math:`c_d` vs `α`.
    block : bool, default = True  TODO
        If `True` block execution once the plot are displayed.
    marker : str, default = ''
        Marker symbol to add to curves (default is no marker).  This
        may be useful when curves are based on data points.  Valid
        `marker` values are the same as in `matplotlib`, e.g. 'o'
        (circle), '^' (triangle), 's' (square), etc.
    state_kwargs :
        Remaining arguments passed as ``self.run(**run_kwargs)``.
    """
    import matplotlib.pyplot as plt

    # Register each figure and all unique properties we need.
    figure_objs = []
    all_props = {'α'}  # α is always required.
    for fig in figs:
        figure_objs.append(None)
        for axis in fig:
            all_props.add(axis)

    if not isinstance(foils, Sequence):
        foils = [foils]

    for foil, colour, label in zip(foils, colours, labels):
        restore = foil.get_states()
        data = defaultdict(list)

        # Sweep over α and get required properties.
        for α_i in np.linspace(α_start, α_end, num=num):
            foil.set_states(α=α_i, **state_kwargs)
            for prop in all_props:
                val = getattr(foil, prop)
                data[prop].append(val)

        data['α'] = np.rad2deg(data['α'])  # -> [°]

        # Define pretty axis labels.
        axis_labels = {'α': "$α$ [°]",
                       'cd': "$c_d$",
                       'cl': "$c_l$",
                       'clα': "$c_{lα}$",
                       'cm_qc': "$c_{m_{1/4c}}$",
                       'xtr_l': "$x_{tr_{lower}}/c$",
                       'xtr_u': "$x_{tr_{upper}}/c$"}

        # TODO labels are incorrect for multiple curves.

        # Make plots.
        for i in range(len(figs)):
            # Set active figure.
            if figure_objs[i] is None:
                figure_objs[i] = plt.figure()
            else:
                plt.figure(figure_objs[i].number)

            x_prop = figs[i][0]
            x_label = axis_labels[x_prop]
            y_labels = []
            for y_prop, ltype in zip(figs[i][1:],
                                     ['-', '--', '-.', '..']):
                plt.plot(data[x_prop], data[y_prop],
                         ltype + colour, label=label)
                y_labels.append(axis_labels[y_prop])

            plt.xlabel(x_label)
            plt.ylabel(', '.join(y_labels))
            plt.title(title)
            plt.legend()
            plt.grid()

        # Reset original state.
        foil.set_states(**restore)

    plt.show(block=block)


# ======================================================================


# Future Improment: Rewrite for array argument if necessary.
def std_α(α: float) -> float:
    r"""
    Normalises angle of attack/s to the range of `α` ∈ [-π, +π].

    Parameters
    ----------
    α : float
        Angle/s of attack to normalise.

    Returns
    -------
    float
        Normalised angle/s of attack.
    """

    # This is slightly more complex than it looks because we want both
    # +ve and -ve sides.
    sgn, mag = np.sign(α), np.abs(α) % (2 * np.pi)  # +/-1 and [0, 2π).
    if mag <= np.pi:
        return sgn * mag  # Return directly α ∈ [-π, +π].
    else:
        return sgn * (mag - 2 * np.pi)  # Flip α ∈ (-2π, -π), (+π, +2π).

# ======================================================================
