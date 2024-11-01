"""
Computes three-dimensional stress-concentration factors for
straight-shank or countersunk holes subjected to remote tension, remote
bending, pin loading and wedge loading.

Python version by Eric J. Whitney April 2020 based on FORTRAN code
appearing in Shivakumar, K. N.  and Newman, J. C., "Stress
Concentrations for Straight-Shank and Countersunk Holes in Plates
Subjected  to Tension, Bending, and Pin Loading", NASA Technical Paper
NASA-TP-3192, June 1992.
"""
import numpy as np


# Written by Eric J. Whitney, April 2020.

# ======================================================================


def kt_hole3d(rt: float, bt: float, zt: float, rw: float,
              lcase: str) -> float:
    r"""
    Computes the three-dimensional stress concentration in countersunk
    and straight shank rivet holes.  Original FORTRAN version developed
    by Kunigal N. Shivakumar and J. C. Newman, Jr. April 1991.  Adapted
    to Python by Eric J. Whitney, April 2020.

    .. note:: A countersink angle of :math:`\theta_c` = 100° is assumed
       by the code which is typical for aircraft fasteners.  However as
       indicated in NASA-TP-3192 page 5 the method is accurate to ±3.5%
       for :math:`\theta_c` ± 10°, i.e. :math:`\theta_c` = 90° → 120°.

    Parameters
    ----------
    rt : float
        Hole radius to thickness ratio, :math:`0.25 \leq r/t \leq 2.5`.

    bt : float
        Straight shank length to plate thickness ratio,
        :math:`0 \leq b/t \leq 1`.  If ``bt == 1`` the hole is straight
        shank only.

    zt : float
        Location where stress concentration factor is required as a
        ratio of plate thickness, :math:`-0.5 \leq z/t \leq 0.5`.

        .. note: `z` is measured from the mid-plane of the plate.

    rw : float
        Hole radius to plate half-width, :math:`r/w < 0.25`.

    lcase : str
        Loading case from one of the following options:

        - Straight shank:

          * 'tension': Remote tension.
          * 'bending': Remote bending.
          * 'pin': Pin (or rivet) loading.
          * 'wedge': Wedge loading.

        - Countersunk hole:

          * 'tension': Remote tension.
          * 'bending': Remote bending.

    Returns
    -------
    kt : float
        Three-dimensional stress concentration factor (or :math:`K_t`).
        Depending on load case:

        - Remote tension: :math:`K_t=\sigma_{max} / \sigma_t` where
          :math:`\sigma_t` is the remote applied stress.
        - Remote bending: :math:`K_b=\sigma_{max} / [6M/(t^2)]` where
          `M` = remote applied moment per unit width.
        - Pin (or rivet) load: :math:`K_p=\sigma_{max} / [P / (2rt)]`
          where `P` = pin load.
        - Wedge load: :math:`K_w=\sigma_{max} / [P / (2rt)]` where `P`
          = wedge load

        .. note:: `K` / SCF for simulated rivet loading is obtained
           by adding one-half of the SCF for remote tension S =
           p/(2wt) and one-half of the SCF for wedge loading (`2w`
           is total width of plate).
    """
    if not (0.0 <= bt <= 1.0):
        raise ValueError(f"b/t = {bt:.3f} out of range (0..1).")
    if not (0.25 <= rt <= 2.5):
        raise ValueError(f"r/t = {rt:.3f} out of range (0.25..2.5).")
    if not (-0.5 <= zt <= 0.5):
        raise ValueError(f"z/t = {zt:.3f} out of range (-0.5..+0.5).")
    if rw > 0.25:
        raise ValueError(f"r/w = {rw:.3f} out of range (<= 0.25).")
    if lcase not in ('tension', 'bending', 'pin', 'wedge'):
        raise ValueError(f"Unknown load case: {lcase}.")

    if bt == 1.0:
        if lcase == 'pin':
            scf_tension = _kt_hole_straight(rt, zt, 'tension')
            scf_wedge = _kt_hole_straight(rt, zt, 'wedge')
            return (scf_wedge + rw * scf_tension) * 0.5
        else:
            return _kt_hole_straight(rt, zt, lcase)
    elif bt == 0.0:
        return _kt_hole_csunk(rt, 0, zt, lcase)
    else:
        idx_lwr = int(bt / 0.25)
        z_lwr, z_upr = 0.25 * idx_lwr, 0.25 * (idx_lwr + 1)
        zt1, zt2 = _nzt(bt, zt, z_lwr, z_upr)
        scf1 = _kt_hole_csunk(rt, idx_lwr, zt1, lcase)
        scf2 = _kt_hole_csunk(rt, idx_lwr, zt2, lcase)
        return scf1 + (scf2 - scf1) / 0.25 * (bt - z_lwr)


# ======================================================================
# Straight shank constants and function.

_SS_ALP = np.reshape([
    3.1825, 0.1679, -0.2063, 0.0518,
    0.4096, -1.5125, 1.1650, -0.2539,
    -1.2831, 2.8632, -2.0000, 0.4239,
    2.2778, -6.0148, 4.5357, -0.9983,
    -2.0712, 5.2088, -3.8337, 0.8331,
    1.7130, .1390, -0.1356, 0.0317,
    0.3626, -1.0206, 0.7242, -0.1527,
    -1.5767, 3.0242, -2.0075, 0.4169,
    3.1870, -6.5555, 4.4847, -0.9450,
    -2.3673, 4.6981, -3.1644, 0.6614], (4, 5, 2), order='F')

_SS_ALPB = np.reshape([
    3.1773, -1.7469, 0.9801, -0.1875,
    -0.2924, 0.1503, -0.0395, 0.0040,
    0.8610, -2.1651, 1.5684, -0.3370,
    -1.2427, 2.7202, -1.8804, 0.3957], (4, 4), order='F')


# ----------------------------------------------------------------------

def _kt_hole_straight(rt, zt, lcase):
    """
    Three-dimensional stress concentration equation for straight shank
    rivet hole subjected to:

    - Remote tension.
    - Remote bending.
    - Pin loading in hole (r/w < 0.25).
    - Wedge loading in hole.

    Range of parameters:

    - -0.5 < z/t < 0.5
    - 0.25 < r/t < 2.5
    """
    idxs = {'tension': 0, 'wedge': 1, 'bending': 2}
    l_idx = idxs[lcase]

    scf = 0.0
    z2t = 2 * zt
    if lcase in ('tension', 'wedge'):
        for i in range(4):
            if z2t == 0.0:
                scf += _SS_ALP[i, 0, l_idx] * rt ** i
            else:
                for j in range(5):
                    scf += _SS_ALP[i, j, l_idx] * rt ** i * z2t ** (j * 2)
    else:
        for i in range(4):
            for j in range(4):
                scf += _SS_ALPB[i, j] * rt ** i * z2t ** (2 * j + 1)
    return scf


# ======================================================================
# Countersunk constants and function.

_CS_ALP = np.reshape([
    3.1675, 1.2562, -0.4052, 3.7503, -8.8507, 2.8948, -15.6036,
    23.4071, -7.7898, 22.1981, -30.9691, 10.3670, -11.1465, 15.1933,
    -5.0730, 3.5507, 0.7198, -0.2232, 0.1185, 1.0574, -0.2623,
    -2.2035, 2.0077, -0.4746, -4.2715, 5.0031, -1.4629, -2.9410,
    3.7985, -1.1888, 3.4454, 0.4835, -0.1485, 0.3460, 0.1089,
    0.0844, -2.2150, 1.1287, -0.0843, -6.5876, 7.3731, -2.1234,
    -4.9136, 6.1237, -1.8862, 3.3341, 0.0777, -0.0259, -0.0229,
    -0.5498, 0.3049, -4.7184, 2.8236, -0.5229, -12.1049, 12.3213,
    -3.5036, -8.1604, 9.1806, -2.7318,
    -2.7192, 0.4773, -0.1620, 5.2713, 2.1888, -0.6093, 3.2839,
    -10.8632, 3.2768, -4.5453, 9.9384, -3.0428, 0.7327, -2.2565,
    0.7056, -1.4221, 0.4322, -0.1424, 1.6817, -1.1265, 0.3481,
    1.2863, -2.0711, 0.6784, 2.4568, -3.8178, 1.2723, 1.8492,
    -2.6911, 0.8911, 0.1935, -0.0883, 0.0135, 3.8939, -2.7731,
    0.8887, 3.2128, -6.4904, 2.3056, 5.8885, -10.6559, 3.7525,
    3.9311, -6.2356, 2.1384, 1.7020, -0.7146, 0.2021, 6.4706,
    -4.6850, 1.4482, 8.3737, -12.5101, 4.0720, 14.4058, -19.9993,
    6.4740, 8.3649, -10.8222, 3.4552], (3, 5, 4, 2), order='F')

_CS_BET = np.reshape([
    3.1675, 1.2562, -0.4052, 3.7503, -8.8507, 2.8948, -15.6036,
    23.4071, -7.7898, 22.1981, -30.9691, 10.3670, -11.1465, 15.1933,
    -5.0730, 3.5507, 0.7198, -0.2232, -1.4878, -4.1557, 1.2616,
    0.6958, 8.9708, -2.6866, 2.6002, -13.8774, 4.2240, -3.0363,
    8.2145, -2.5264, 3.4454, 0.4835, -0.1485, -1.1969, -2.6156,
    .7803, 1.0127, 1.8286, -0.5102, 0.3438, -1.8037, 0.5698,
    -1.3109, 1.7708, -0.5768, 3.3341, 0.0777, -0.0259, -0.6655,
    -1.7805, 0.5880, -0.9018, 3.0805, -1.0493, 2.1386, -4.3757,
    1.5303, -1.6774, 2.7382, -0.9445,
    -2.7192, 0.4773, -0.1620, 5.2713, 2.1888, -0.6093, 3.2839,
    -10.8632, 3.2768, -4.5453, 9.9384, -3.0428, 0.7327, -2.2565,
    0.7056, -1.4221, 0.4322, -0.1424, 6.6870, -2.1064, 0.7330,
    -9.2419, 4.3538, -1.5784, 13.6204, -9.2163, 3.1486, -7.6364,
    6.0611, -2.0053, 0.1935, -0.0883, 0.0135, 2.8201, -1.4920,
    0.5510, -0.4453, 1.8097, -0.7420, 0.6186, -3.4144, 1.2552,
    -1.1330, 2.6470, -0.8987, 1.7020, -0.7146, 0.2021, 0.2472,
    -0.4422, 0.2356, 1.8402, 0.0875, -0.2380, -1.9081, -0.4494,
    0.4036, 0.1992, 0.8738, -0.3866], (3, 5, 4, 2), order='F')


# ----------------------------------------------------------------------

def _kt_hole_csunk(rt, k, zt, lcase):
    """
    Three-dimensional stress concentration factor for countersunk rivet
    hole subjected to:

    - Remote tension.
    - Remote bending.

    Solution is for the countersink angle of 100 degrees and a selected
    value of k = 0, 1, 2 or 3 corresponding to b/t = 0, 0.25, 0.50 or
    0.75.  Results for other b/t values are computed by linear
    interpolation between the bracketing b/t values.

    Range of parameters:
        - -0.5 < z/t < 0.5.
        - 0.25 < r/t < 2.5.
    """
    idx_table = {'tension': 0, 'bending': 1}
    l_idx = idx_table[lcase]
    bt = 0.25 * k
    ccor = (1 - 2 * bt) / 2
    scf = 0.0
    if bt != 0.0:
        if -0.5 <= zt <= -ccor:
            t1 = ccor / bt
            z = t1 + zt / bt
            for i in range(3):
                if z == 0.0:
                    scf += _CS_ALP[i, 0, k, l_idx] * rt ** i
                else:
                    for j in range(5):
                        scf += _CS_ALP[i, j, k, l_idx] * rt ** i * z ** j
        else:
            t2 = ccor / (1 - bt)
            z = t2 + zt / (1 - bt)
            for i in range(3):
                if z == 0.0:
                    scf += _CS_BET[i, 0, k, l_idx] * rt ** i
                else:
                    for j in range(5):
                        scf += _CS_BET[i, j, k, l_idx] * rt ** i * z ** j
    else:
        t2 = ccor / (1 - bt)
        z = t2 + zt / (1 - bt)
        for i in range(3):
            if z == 0.0:
                scf += _CS_BET[i, 0, k, l_idx] * rt ** i
            else:
                for j in range(5):
                    scf += _CS_BET[i, j, k, l_idx] * rt ** i * z ** j
    return scf


# ----------------------------------------------------------------------

def _nzt(bt, zt, bt1, bt2):
    """Evaluate appropriate z-location for countersunk hole."""
    if (bt - 0.5) < zt <= 0.5:
        zt1 = bt1 - 0.5 + (zt - bt + 0.5) * (1.0 - bt1) / (1.0 - bt)
        zt2 = bt2 - 0.5 + (zt - bt + 0.5) * (1.0 - bt2) / (1.0 - bt)
    else:
        zt1 = bt1 - 0.5 + (zt - bt + 0.5) * bt1 / bt
        zt2 = bt2 - 0.5 + (zt - bt + 0.5) * bt2 / bt
    return zt1, zt2

# ======================================================================
