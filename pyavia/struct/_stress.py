"""
Fundamental functions for working with material / structural stresses.
"""

__all__ = ['mohr2d']

from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
from pyavia import Dim, kind_atan2


def mohr2d(s_xx: Any, s_yy: Any, t_xy: Any, rot_ang: Any = None):
    r"""Mohr's circle transformation of stresses in any orientation to
    principle axes or another nominated angle.

    Parameters
    ----------
    s_xx : scalar
        Stress along local x-x axis.
    s_yy : scalar
        Stress along local y-y axis.
    t_xy : scalar
        Shear stress in local x-y directions.
    rot_ang : None or float
        Rotation angle (**radians** if no units given, default = None):

        - If None, returns stresses transformed to principle axes,
          i.e. :math:`\sigma_{11}`, :math:`\sigma_{22}` and always
          :math:`\tau_{12} = 0`.
        - If supplied returns stresses transformed by the specified rotation
          (i.e. axis x-x → x'-x').

    Returns
    -------
    s_xx', s_yy', t_xy', rot_ang : scalar, float
        Transformed stresses and rotation angle (computed for principle axes if
        rot_ang was None).

    """
    if isinstance(rot_ang, Dim):
        rot_ang = rot_ang.convert('rad').value
    elif rot_ang is None:
        # Transform to principle axes.
        rot_ang = 0.5 * kind_atan2(t_xy, 0.5 * (s_xx - s_yy))

    cos_rot, sin_rot = np.cos(rot_ang), np.sin(rot_ang)
    cos_2rot, sin_2rot = np.cos(2 * rot_ang), np.sin(2 * rot_ang)
    transform = np.array([[cos_rot ** 2, sin_rot ** 2, sin_2rot],
                          [sin_rot ** 2, cos_rot ** 2, -sin_2rot],
                          [-0.5 * sin_2rot, 0.5 * sin_2rot, cos_2rot]])
    s_rot = transform @ np.array([s_xx, s_yy, t_xy])
    return s_rot[0], s_rot[1], s_rot[2], rot_ang
