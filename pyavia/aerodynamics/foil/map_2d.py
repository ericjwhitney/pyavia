import warnings
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay, QhullError

from .base import Foil2DAero, Foil2DBasic
from .stall import PostStall2DMixin


# Written by Eric J. Whitney, January 2023.


# ============================================================================

@dataclass
class _ReMFoil:
    """
    Used to combine information about reference foils added to Map2DAero.
    """
    Re: float
    M: float
    foil: Foil2DAero

# TODO Probably remove this now.

# ============================================================================


class Map2DAero(Foil2DBasic):
    """
    Aerodynamic model that interpolates across multiple separate foil
    reference models, with properties depending on the current Reynolds
    number, Mach number and angle of attack.

    Notes
    -----
    - Only the angle of attack (`α`) of the individual foil reference models
      is varied (via `set_states`) when computing interpolated values.  The
      `Map2DAero` object does not adjust other input states (`Re`, `M`,
      etc) separately.
    - Individual foil reference models retain their adjustable input states
      (`Re`, `M`, etc) after insertion. As such if they are adjusted and
      this affects their property values, this could lead to unexpected
      behaviour.  See `Map2DAero.add_foil` for more details.
    - Because a triangulation is used for interpolation, at least three
      (`Re`, `M`) points (foil models) are required (via `add_foil`) before
      any valid aerodynamic property can be computed.
    - Interpolation of Reynolds number are done on a log scale.
    - If the `Re`, `M` values given to `set_states` are outside the mapping
      region, the object will return ``Map2DAero.valid_state == False``  any
      property accessed in this condition will return `NaN`.
    """
    warn_extrapolate = True
    """When `True` any attempts to extrapolate for values outside the `Re`, 
    `M` range will produce a warning. Properties return `NaN` in this 
    situation."""
    warn_no_triangulation = True
    """When `True` any attempt to compute a property when no interpolating 
    triangulation is available will produce a warning. Properties return 
    `NaN` in this situation."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs :
            Passed to `Foil2DBasic.__init__`.
        """
        # Setup base foil object.
        super().__init__(**kwargs)

        # Store (Re, M) and Foil2DAero references together in a list.
        self._ReMFoils: [_ReMFoil] = []
        self._tri_mesh: Delaunay | None = None

        # Define current interpolation triangle in self._tri_current. Values
        # can be:
        #   None: Not computed - find new triangle on next lookup.
        #    < 0: (Re, M) outside interpolation area.
        #   >= 0: Valid interpolation triangle for (Re, M).
        self._tri_current = None

    def __getattr__(self, prop: str):
        """
        `__getattr__` is overridden because a special foil property may be
        requested that is specific to the stored foil objects but that is
        not part of the `Foil2DBasic` definition. If we fail to find a basic
        `Foil2DBasic` property, but the stored foil objects possess it,
        we still attempt interpolation of the stored foils.

        Raises
        ------
        AttributeError
            If none of the internal foil objects have the given property.
        """
        if prop in self.all_states():
            return self._interp_prop(prop)
        else:
            raise AttributeError(f"Map2DAero has no property '{prop}'.")

    # -- Public Methods -----------------------------------------------------

    def add_foil(self, foil: Foil2DAero, Re: float = None, M: float = None):
        """
        Add a new reference foil to the set of interpolating foils.

        Parameters
        ----------
        foil : Foil2DAero
            Reference to a foil object that will be included in the
            interpolating set at the given (`Re`, `M`) point.
        Re : float, optional
            The reference Reynolds number applicable to this foil.  If
            omitted, the current Reynolds number state property of the
            `foil` object itself is used, i.e. ``Re=foil.Re``.
        M : float, optional
            The reference Mach number applicable to this foil.  If
            omitted, the current Mach number state property of `foil` object
            itself is used, i.e. ``M=foil.M``.

        Notes
        -----
        - All foils must be of identical type.
        - Although each individual foil is assigned a specific (`Re`,
          `M`) tuple when added, this does not imply that `Re` or `M` are
          then fixed for these models.  This means that subsequent calls to
          `set_states` may result in unexpected behaviour.  If fixed
          behaviour is desired, individual foil models should have fixed
          `Re`, `M` (e.g. see `Polar2DAero`).
        """
        Re = foil.Re if Re is None else Re
        M = foil.M if M is None else M

        if (Re, M) in self.all_ReM():
            raise ValueError(f"Model for Re = {Re:.03E}, M = {M:.03f} "
                             f"already exists.")

        # Check foil type, synchronise α and add to list.
        ref_foil = self._ref_foil()
        if (ref_foil is not None) and (type(foil) is not type(ref_foil)):
            raise ValueError("All reference foils must be of identical type.")

        foil.set_state(α=self.α)
        self._ReMFoils.append(_ReMFoil(Re=Re, M=M, foil=foil))

        # The current triangulation is now superseded. This generated
        # lazily later on when the next property request is made.
        self._tri_mesh, self._tri_current = None, None

    # Here we override all the basic properties of the Foil2DBasic superclass,
    # sending them for interpolation.  More sophisticated properties that
    # may be used by stored foils are handled by __getattr__.

    @property
    def α0(self) -> float:
        return self._interp_prop('α0')

    @property
    def α_stall_neg(self) -> float:
        return self._interp_prop('α_stall_neg')

    @property
    def α_stall_pos(self) -> float:
        return self._interp_prop('α_stall_pos')

    @property
    def cd(self) -> float:
        return self._interp_prop('cd')

    @property
    def cl(self) -> float:
        return self._interp_prop('cl')

    @property
    def clα(self) -> float:
        return self._interp_prop('clα')

    @property
    def cl_stall_neg(self) -> float:
        return self._interp_prop('cl_stall_neg')

    @property
    def cl_stall_pos(self) -> float:
        return self._interp_prop('cl_stall_pos')

    @property
    def cm_qc(self) -> float:
        return self._interp_prop('cm_qc')

    def input_states(self) -> frozenset[str]:
        """
        Input states are combined from `Foil2DBasic` and the reference foils
        (which all have identical type).
        """
        if (ref_foil := self._ref_foil()) is not None:
            return super().input_states() | ref_foil.input_states()
        else:
            return super().input_states()

    def output_states(self) -> frozenset[str]:
        """
        Returns a set of output states combined from `Foil2DBasic` and the
        reference foils (which all have identical type).  This is computed
        by subtracting the combined input states from all available states.
        """
        Σoutput_states = super().output_states()
        if (ref_foil := self._ref_foil()) is not None:
            Σoutput_states |= ref_foil.output_states()

        return Σoutput_states - self.input_states()

    def all_ReM(self) -> [(float, float)]:
        """
        Returns
        -------
        [(float, float), ...]
            List of (`Re`, `M`) values for the currently stored reference foil
            models.
        """
        return [(ReMFoil.Re, ReMFoil.M) for ReMFoil in self._ReMFoils]

    def set_state(self, *, Re: float = None, M: float = None,
                  α: float = None) -> frozenset[str]:
        """
        Set the state of the `Map2DAero` object itself (`Re`, `M`, `α`).
        For stored reference foils, only `α` is updated.

        Notes
        -----
        - Updating other reference foil states (except for `α`) must be done
          separately.
        """
        changed = super().set_state(Re=Re, M=M, α=α)
        if 'α' in changed:
            for ReMFoil in self._ReMFoils:
                ReMFoil.foil.set_state(α=self.α)

        # If Reynolds or Mach numbers were changed, we need a new
        # interpolating triangle.
        if 'Re' in changed or 'M' in changed:
            self._tri_current = None

        return changed

    @property
    def valid_state(self) -> bool:
        """
        Returns `True` if the current `Re`, `M` values are within the
        interpolation region, otherwise `False`.
        """
        # Generate mesh and find current triangulation (if required).
        self._triangulate()
        if self._tri_current is None or self._tri_current < 0:
            return False
        else:
            return True

    # -- Private Methods ----------------------------------------------------

    def _interp_prop(self, prop: str) -> float:
        """
        Get the property `prop` for the aerofoil at each vertex of the
        current interpolation triangle and do a barycentric interpolation.
        Returns NaN when:
            - No interpolation can be built yet.
            - The current (Re, M) point is outside the available
              interpolation area.
            - If any of the individual vertex models cannot compute the
              value / are missing the data.
        """
        self._triangulate()

        # Return (and warn) if no triangulation is available at all.
        if self._tri_mesh is None:
            if self.warn_no_triangulation:
                warnings.warn(f"No triangulation available for interpolation "
                              f"- {len(self._ReMFoils)} reference models "
                              f"available.")
            return np.nan

        # Return NaN if no triangulation or outside the available
        # interpolation area.
        if self._tri_current is None or self._tri_current < 0:
            return np.nan

        # A valid interpolating triangle containing (log10(Re), M) is
        # available.  Get the required property at each vertex.
        f_v = []
        for v in self._tri_mesh.simplices[self._tri_current]:
            # Get the property from the reference foil assosicated with
            # this vertex.  Note:
            #   - NaN may be produced here which is desired behaviour;
            #     it will propagate thru the barycentric interpolation.
            #   - Some verticies may not produce the property at all.
            #     If so these are also assigned NaN which also propogate.

            foil_v = self._ReMFoils[v].foil
            TEMP_val = getattr(foil_v, prop, np.nan)
            f_v.append(TEMP_val)

        # Barycentric transform for λ -> x: T.λ + r = x
        # Therefore for x -> λ: λ = T^-1.(x - r)
        # Delaunary.transform returns T^-1 with vector r stacked on.
        x = np.array((np.log10(self.Re), self.M))
        T_inv = self._tri_mesh.transform[self._tri_current, :2]
        r = self._tri_mesh.transform[self._tri_current, 2]
        λ_12 = T_inv.dot((x - r).T)  # Find λ for remote nodes.
        λ = np.r_[λ_12, 1.0 - np.sum(λ_12)]  # Add on normalising λ3.

        # Barycentric interpolation: f(x) ≈ λ1.f(x1) + λ2.f(x2) + λ3.f(x3)
        return float(np.sum(λ * f_v))

    def _ref_foil(self) -> Foil2DAero | None:
        """Returns the first reference foil added (if it exists) or `None`."""
        if self._ReMFoils:
            # return self._ReMFoils[0][2]
            return self._ReMFoils[0].foil
        else:
            return None

    def _triangulate(self):
        """
        Does the following steps:
            1. Generate (log10(Re), M) triangulation if it does not exist.
            2. Sets current interpolating triangle, if not set previously.
        """
        # If required, try to generate the 2D (log10(Re), M) triangulation.
        # If we can't, nothing else can be done.
        if self._tri_mesh is None:
            self._tri_current = None  # Reset for new triangulation.

            # A minimum of three points are required.
            if len(self._ReMFoils) < 3:
                return

            try:
                logReMs = np.array([(np.log10(foil_i.Re), foil_i.M)
                                    for foil_i in self._ReMFoils])
                self._tri_mesh = Delaunay(logReMs)

            except QhullError:
                # Couldn't generate triangulation yet.
                return

        # If required, find the current interpolation triangle for our point
        # x = (log10(Re), M).
        if self._tri_current is None:
            x = np.array((np.log10(self.Re), self.M))
            self._tri_current = self._tri_mesh.find_simplex(x)

            # If it can't be found, this generates a warning the first time
            # this (Re, M) pair is used.
            if self._tri_current < 0 and self.warn_extrapolate:
                # Get points around the boundary of the current triangulation.
                boundary_idx = np.unique(self._tri_mesh.convex_hull.ravel())
                all_logReMs = self._tri_mesh.points[boundary_idx]

                # To help eliminate model errors we find nearest available
                # (log10(Re), M) point.
                logReM_dists = np.linalg.norm(all_logReMs - x, axis=1)
                near_idx = np.argmin(logReM_dists)
                near_Re = np.power(10.0, all_logReMs[near_idx][0])
                near_M = all_logReMs[near_idx][1]

                warnings.warn(f"Re = {self.Re:.03E}, M = {self.M:.03f} is "
                              f"outside interpolation area. Nearest datapoint"
                              f" Re = {near_Re:.03E}, M = {near_M:.03f}.")


# ============================================================================

class Map2DAeroPostStall(PostStall2DMixin, Map2DAero):
    """
    'Map2DAeroPostStall' extends `Map2DAero` class by adding the
    `PostStall2DMixin` mixin to give estimated properties at large
    positive and negative (post-stall) angles of attack.

    Default post-stall model behaviour is to replace base model `NaN`
    values only, regardless of lift (unless overridden).
    """
    warn_extrapolate = False

    def __init__(self, **kwargs):
        super().__init__(**({'post_nan': True, 'post_crossover': False} |
                            kwargs))

# ===========================================================================

# Future work: MultiFoilGeo can go here which is used for geometric
# interpolation between aerofoils here.
