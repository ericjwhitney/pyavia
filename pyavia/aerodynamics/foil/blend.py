from copy import deepcopy
from typing import Any

from .base import Foil2DBasic, Foil2DAero


# Written by Eric J. Whitney, January 2023.


# ======================================================================


class Blend2DAero(Foil2DAero):
    """
    `Blend2DAero` allows for linear blending of two identical foil
    objects based on a fixed factor.  For example, this may be used to
    obtain intermediate properties along a geometry dimension.

    Parameters
    ----------
    foil_a, foil_b : Foil2DBasic
        References `Foil2DBasic` foils to be interpolated.

    frac : float
        Fraction (from 0..1) of `foil_a` values forming the combined
        result.  For example:
        - `frac = 1.0` -> Result is 100% `foil_a` and 0% `foil_b`.
        - `frac = 0.5` -> Result is 50% `foil_a` and 50% `foil_b`.
        - `frac = 0.0` -> Result is 0% `foil_a` and 100% `foil_b`.

    deep_copy : bool, default = True
        If `True` deep copies of `foil_a` and `foil_b` are made for
        internal reference (see `Notes`).

    **init_state
        Initial state parameters provided are used to initialise
        reference foils by passing them to
        ``foil_a.set_state(**init_state)`` and
        ``foil_b.set_state(**init_state)``.

    Raises
    ------
    ValueError
        If `frac` is not between 0 and 1.

    Notes
    -----
    - `foil_a` and `foil_b` must both implement any attributes /
      properties required by a calling function.  See
      `__getattr__` for details.

    - If ``deep_copy=False``, references to the original foil objects
      are held internally.  As such, calling ``set_states(...)`` on this
      object will also cause ``foil_a.set_states()`` and
      ``foil_b.set_states(...)`` to be called in turn, and this may lead
      to unexpected behaviour if they are also used elsewhere.

    - Any external calls to ``foil_a.set_states(...)`` or
      ``foil_b.set_states(...)`` may also lead to unexpected results
      from this object. It is recommended to always call `set_states`
      on this object immediately prior to use.
    """

    def __init__(self, foil_a: Foil2DBasic, foil_b: Foil2DBasic,
                 frac: float, *, deep_copy: bool = True, **init_state):
        # Initialise superclass and create references to subfoils.
        super().__init__()

        if deep_copy:
            self._foil_a = deepcopy(foil_a)
            self._foil_b = deepcopy(foil_b)
        else:
            self._foil_a = foil_a
            self._foil_b = foil_b

        # Setup fraction and initialise subfoils.
        self._frac = frac
        if self._frac < 0 or self._frac > 1:
            raise ValueError("Blend fraction must be between 0 and 1.")

        self._foil_a.set_state(**init_state)
        self._foil_b.set_state(**init_state)

    def __getattr__(self, prop: str):
        """
        `__getattr__` is overridden because a contained `foil_a` and
        `foil_b` may have a property that is not part of the basic
        `Foil2DBasic` definition. If we fail to find a base property
        but `foil_a` / `foil_b` includes it, we try to provide a linear
        interpolation.

        Raises
        ------
        AttributeError
            If either of the internal foil objects do not have the given
            property.
        """
        try:
            prop_a = getattr(self._foil_a, prop)
        except AttributeError:
            raise AttributeError(f"'Blend2DAero.foil_a' has no "
                                 f"property '{prop}'.")

        try:
            prop_b = getattr(self._foil_b, prop)
        except AttributeError:
            raise AttributeError(f"'Blend2DAero.foil_b' has no "
                                 f"property '{prop}'.")

        return _linear_interp(self._frac, prop_a, prop_b)

    # -- Public Methods ------------------------------------------------

    @property
    def cd(self) -> float:
        return self._interp_base('cd')

    @property
    def cl(self) -> float:
        return self._interp_base('cl')

    @property
    def clα(self) -> float:
        return self._interp_base('clα')

    @property
    def cl_stall_neg(self) -> float:
        return self._interp_base('cl_stall_neg')

    @property
    def cl_stall_pos(self) -> float:
        return self._interp_base('cl_stall_neg')

    @property
    def cm_qc(self) -> float:
        return self._interp_base('cm_qc')

    @property
    def foil_a(self) -> Foil2DAero:
        """Returns a reference to the stored 'foil_a' object."""
        return self._foil_a

    @property
    def foil_b(self) -> Foil2DAero:
        """Returns a reference to the stored 'foil_b' object."""
        return self._foil_b

    def get_state(self) -> dict[str, Any]:
        """
        Return the combined state of `foil_a` and `foil_b`.  Where both
        foil objects have common attributes, their values are
        interpolated.
        """
        # Get the individual state of each of the contained foils and 
        # duplicate any missing keys in each from the other.
        a_state = self._foil_a.get_state()
        b_state = self._foil_b.get_state()
        
        ab_state = {}
        for k in a_state.keys() | b_state.keys():
            if k not in a_state:
                ab_state[k] = b_state[k]
            elif k not in b_state:
                ab_state[k] = a_state[k]
            else:
                ab_state[k] = _linear_interp(self._frac, a_state[k],
                                             b_state[k])

        return ab_state

    @property
    def M(self) -> float:
        return self._interp_base('M')

    @property
    def Re(self) -> float:
        return self._interp_base('Re')

    def set_state(self, **kwargs) -> frozenset[str]:
        """
        Set the state of each of the contained reference foil objects.

        .. note:: The `Blend2DAero` object has no state variables of its
           own, including for parameters such as `α`, `Re` and `M`.
           Because of this, calling `set_state` on either `foil_a` or
           `foil_b` separately may produce unexpected results.
        """
        changed = self._foil_a.set_state(**kwargs)
        changed |= self._foil_b.set_state(**kwargs)
        return changed

    @property
    def α(self) -> float:
        return self._interp_base('α')

    @property
    def α0(self) -> float:
        return self._interp_base('α0')

    @property
    def α_stall_neg(self) -> float:
        return self._interp_base('α_stall_neg')

    @property
    def α_stall_pos(self) -> float:
        return self._interp_base('α_stall_pos')

    # -- Private Methods -----------------------------------------------

    def _interp_base(self, prop: str) -> float:
        """
        Interpolate a property from 'foil_a' and 'foil_b'. These are
        assumed to exist already in `Foil2DAero` and no checking is
        done to see if the property actually exists.
        """
        return _linear_interp(self._frac,
                              getattr(self._foil_a, prop),
                              getattr(self._foil_b, prop))


# ======================================================================

def _linear_interp(frac: float, a: float, b: float) -> float:
    return frac * a + (1 - frac) * b
