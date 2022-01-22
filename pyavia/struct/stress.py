
from typing import Any
import numpy as np

from pyavia.math import kind_atan2
from pyavia.units import dim, DimScalar, Dim, convert


# =============================================================================


def mohr2d(s_xx: DimScalar, s_yy: DimScalar, t_xy: DimScalar,
           rot_ang: DimScalar = None) -> (DimScalar, DimScalar, DimScalar, Dim):
    r"""
    Mohr's circle transformation of stresses in any orientation to principle
    axes or another nominated angle.  If values are ``Dim`` objects,
    the outputs will use the units of `s_xx` and `rot_ang`.

    Parameters
    ----------
    s_xx : Scalar, Dim
        Stress along local x-x axis.
    s_yy : Scalar, Dim
        Stress along local y-y axis.
    t_xy : Scalar, Dim
        Shear stress in local x-y directions.
    rot_ang : None | Scalar, Dim
        Rotation angle (**radians** if no units given, default = None):

        - If None, returns stresses transformed to principle axes,
          i.e. :math:`\sigma_{11}`, :math:`\sigma_{22}` and always
          :math:`\tau_{12} = 0`.
        - If supplied returns stresses transformed by the specified rotation
          (i.e. axis x-x → x'-x').

    Returns
    -------
    s_xx', s_yy', t_xy', rot_ang : Scalar, Dim
        Transformed stresses in original units and rotation angle n radians
        (computed for principle axes if `rot_ang` was ``None``).

    """
    if rot_ang is None:
        # Transform to principle axes.
        θ = 0.5 * kind_atan2(t_xy, 0.5 * (s_xx - s_yy))
        θ_units = ''

    else:
        θ, θ_units = dim(rot_ang)
        if θ_units:
            θ = convert(θ, from_units=θ_units, to_units='rad')

    # Normalise stresses, convert to float.
    σ_xx, σ_units = dim(s_xx)
    σ_yy = dim(s_yy).to_real(σ_units)
    τ_xy = dim(t_xy).to_real(σ_units)

    cos_θ, sin_θ = np.cos(θ), np.sin(θ)
    cos_2θ, sin_2θ = np.cos(2 * θ), np.sin(2 * θ)
    transform = np.array([[cos_θ ** 2, sin_θ ** 2, sin_2θ],
                          [sin_θ ** 2, cos_θ ** 2, -sin_2θ],
                          [-0.5 * sin_2θ, 0.5 * sin_2θ, cos_2θ]])

    σ_rot = transform @ np.array([σ_xx, σ_yy, τ_xy])

    # Restore units if required.
    return (dim(σ_units) * σ_rot[0],  # Mult allows unit dropoff.
            dim(σ_units) * σ_rot[1],
            dim(σ_units) * σ_rot[2],
            dim(θ, 'rad').convert(θ_units) if θ_units else θ)

# -----------------------------------------------------------------------------
