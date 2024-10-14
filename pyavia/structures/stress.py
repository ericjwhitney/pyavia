import numpy as np

# Written by Eric Whitney, September 2020.

# =============================================================================


def mohr2d(σ_xx: float, σ_yy: float, τ_xy: float,
           θ: float = None) -> (float, float, float, float):
    r"""
    Mohr's circle transformation of stresses in any orientation to another
    nominated angle, or principle axes.  For transformation of strains,
    see `Notes`.

    Parameters
    ----------
    σ_xx : float
        Normal stress parallel to local x-x axis.
    σ_yy : float
        Normal stress parallel to local y-y axis.
    τ_xy : float
        Shear stress in local x-y direction.
    θ : float, default = None
        Rotation angle, if provided.  If `None`, returns stresses
        transformed to principal axes (see `Returns`).  **Note:** Radians
        are required.

    Returns
    -------
    σ_xx', σ_yy', σ_xy', θ : float

        - If `θ` was provided: Stresses transformed by the clockwise rotation
         `θ` (which is repeated in the output).

        - If `θ` was `None`: First (larger) and second (smaller) principal
          stresses, an (approximately) zero shear stress and the first
          principal stress angle (:math:`\sigma_{11}`, :math:`\sigma_{22}`,
          :math:`\gamma_{12} \approx 0`, :math:`\theta`).  The angle `θ`
          is defined as the angle required to rotate `σ_xx` local `x-x` axis
          to the axis of the first principal stress `1-1`.

    Notes
    -----
    - Returned value of `θ` is in the range +/- π.
    - This function can be also used to transform strains (:math:`ε_{xx}`,
      :math:`ε_{yy}`, :math:`γ_{xy}`), however in this case the shear
      strain must be halved before transformation.  The actual shear
      strain can then be recovered by multiplying the result
      by two afterwards.  This is due to the classical definition of
      shear strain as twice the tensor shear strain.
    """
    if θ is None:
        # Transform stresses to principle axes.
        θ = 0.5 * np.arctan2(2 * τ_xy, σ_xx - σ_yy)

    # Write 2D stress transform matrix.  This one laid out per:
    # https://web.mit.edu/course/3/3.11/www/modules/trans.pdf
    cosθ, sinθ = np.cos(θ), np.sin(θ)
    cosθ_2, sinθ_2, sincosθ = cosθ ** 2, sinθ ** 2, sinθ * cosθ

    T = np.array([[cosθ_2, sinθ_2, 2 * sincosθ],
                  [sinθ_2, cosθ_2, -2 * sincosθ],
                  [-sincosθ, sincosθ, cosθ_2 - sinθ_2]])

    σ_rot = T @ np.array([σ_xx, σ_yy, τ_xy])

    return σ_rot[0], σ_rot[1], σ_rot[2], θ

# -----------------------------------------------------------------------------
