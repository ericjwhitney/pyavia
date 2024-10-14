#!/usr/bin/env python3
"""
Example of data-based aerofoil performance models and `Re`, `M` mapping.
"""
# Written by Eric J. Whitney, February 2023.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# noinspection PyUnresolvedReferences
from pyavia.aerodynamics import (Polar2DAero, Polar2DAeroPostStall, Map2DAero,
                                 Map2DAeroPostStall, plot_foil_aero)

d2r, r2d = np.deg2rad, np.rad2deg


# == Example Parameters ======================================================

do_smoothing = True
overlay_points = False  # For visual error checking of triangulation.

datafoil_type = Polar2DAeroPostStall  # This adds a post-stall model ...
multifoil_type = Map2DAero  # ... to each data polar.
sweep_α_deg = [-180, 180]

# datafoil_type = Polar2DAero  # This version adds a post-stall model ...
# multifoil_type = Map2DAeroPostStall  # ... to the combined result.
# sweep_α_deg = [-180, 180]
# Polar2DAero.warn_α_range = False

# datafoil_type = Polar2DAero  # This version has no post-stall model.
# multifoil_type = Map2DAero
# sweep_α_deg = [-15, +25]

# == Aerofoil Raw Data =======================================================

CLARK_Y_RAW = np.array([
    # Raw data is for Re = 1,120,000 and M = 0.
    [-10.00, -0.740, 0.0169, -0.0660],
    [-9.00, -0.600, 0.0140, -0.0710],
    [-7.10, -0.400, 0.0108, -0.0750],
    [-6.00, -0.250, 0.0091, -0.0800],
    [-3.00, 0.100, 0.0075, -0.0900],
    [-1.50, 0.250, 0.0065, -0.0900],
    [0.00, 0.450, 0.0060, -0.0950],
    [+2.40, 0.700, 0.0074, -0.1000],
    [+4.90, 1.000, 0.0102, -0.0980],
    [+8.00, 1.300, 0.0130, -0.0950],
    [+10.40, 1.500, 0.0170, -0.0900],
    [+11.70, 1.600, 0.0241, -0.0640],
    [+13.10, 1.630, 0.0320, -0.0660],
    [+14.00, 1.610, None, -0.0700],  # These points have no drag coeff.
    [+15.00, 1.500, None, -0.0800],
    [+16.00, 1.330, None, -0.1000],
    [+17.00, 1.230, None, -0.1250],
    [+18.50, 1.180, None, -0.1350],
    [+20.00, 1.115, None, -0.1500]])

CLARK_Y_RE, CLARK_Y_M = 1.12e6, 0.0

# == Setup Foil Data Objects =================================================

# Setup multi-Re, M aerofoil and add baseline polar.
clark_y_multidata = multifoil_type()
clark_y_multidata.add_foil(
    datafoil_type(Re=CLARK_Y_RE, M=CLARK_Y_M,
                  α_data=d2r(CLARK_Y_RAW[:, 0].astype('float64')),
                  cl_data=CLARK_Y_RAW[:, 1], cd_data=CLARK_Y_RAW[:, 2],
                  cm_qc_data=CLARK_Y_RAW[:, 3], smooth=do_smoothing))

# Add some different Re, M by adjusting properties.
ADD_RE_M = [(1e5, 0.0), (1e5, 0.6), (1e5, 0.7), (1e5, 0.80),
            (1.12e6, 0.6), (1.12e6, 0.7), (1.12e6, 0.8),  # Baseline +M.
            (5e6, 0.0), (5e6, 0.6), (5e6, 0.7), (5e6, 0.8),
            (5e7, 0.0), (5e7, 0.6), (5e7, 0.7), (5e7, 0.8)]

for Re, M in ADD_RE_M:
    β = np.sqrt(1 - M ** 2)  # Example Prandtl-Galuert M effect.
    f_Re = (Re / CLARK_Y_RE) ** -0.5  # Example Power law Re effect.
    Δcd_Mdd = 0.0 if M < 0.6 else 10 * (M - 0.6) ** 3

    chgd_cl = CLARK_Y_RAW[:, 1] / β
    chgd_cd = np.array(CLARK_Y_RAW[:, 2], dtype=float, copy=True)
    is_nan = np.isnan(chgd_cd)  # ^ dtype converts None to NaN.
    chgd_cd = (chgd_cd * f_Re + Δcd_Mdd).astype(object)
    chgd_cd[is_nan] = None  # Restore None.
    chgd_cm_qc = CLARK_Y_RAW[:, 3] / β

    clark_y_multidata.add_foil(
        datafoil_type(Re=Re, M=M,
                      α_data=d2r(CLARK_Y_RAW[:, 0].astype('float64')),
                      cl_data=chgd_cl, cd_data=chgd_cd,
                      cm_qc_data=chgd_cm_qc, smooth=do_smoothing))

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Data Foil Example:")
    clark_y_multidata.set_state(Re=2e6, M=0.2, α=d2r(5.0))

    # -- Print Example Properties -------------------------------------------

    print(f"Current Aerofoil Properties:")

    # Angles.
    for prop in clark_y_multidata.all_states():
        val = getattr(clark_y_multidata, prop)
        if np.isnan(val):
            # Aerofoil does not cover that property under these conditions.
            continue

        if prop[0] == 'α':
            # Angle format.
            print(f"{prop:>18s} = {r2d(val):+10.03f}°")
        else:
            # General format.
            print(f"{prop:>18s} = {val:10.5G}")

    # -- Plot Example 2D Curves ---------------------------------------------

    if sweep_α_deg is not None:
        plot_foil_aero(clark_y_multidata, α_start=d2r(sweep_α_deg[0]),
                       α_end=d2r(sweep_α_deg[1]),
                       num=int((sweep_α_deg[1] - sweep_α_deg[0]) / 0.25),
                       figs=(('α', 'cl', 'cm_qc'), ('cl', 'cd'),
                             ('α', 'clα')),
                       title="DATAFOIL EXAMPLE", block=False)

    # -- Plot Example Re, M Variation ---------------------------------------

    # Get bounds of log10(Re), M coverage.  Log is used for plotting
    # ootherwise triangulation runs into roundoff errors.
    ReMs = np.array(clark_y_multidata.all_ReM(), ndmin=2)
    lRe_range = (np.log10(np.min(ReMs[:, 0])), np.log10(np.max(ReMs[:, 0])))
    M_range = (np.min(ReMs[:, 1]), np.max(ReMs[:, 1]))

    # Generate points at random within the bounds.
    prop = 'cd'
    lRe_div, M_div = 20, 20
    x, y, z = [], [], []

    # Generate equally spaced points (in the feasible region).
    for lRe in np.linspace(*lRe_range, num=lRe_div):
        for M in np.linspace(*M_range, num=M_div):

            clark_y_multidata.set_state(Re=np.power(10.0, lRe), M=M)
            f = getattr(clark_y_multidata, prop)
            if np.isnan(f):
                continue  # Data not available for that (Re, M) point.

            x.append(lRe)
            y.append(M)
            z.append(f)

    # Contour plot.
    plt.figure()
    ax = plt.axes(projection='3d')
    # noinspection PyUnresolvedReferences
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidths=0.75, edgecolor='0.2',
                    antialiased=True)
    if overlay_points:
        ax.scatter(x, y, z, marker='o', edgecolors='k')

    ax.set_xlabel(r"$\log_{10}(Re)$")
    ax.set_ylabel("$M$")
    ax.set_zlabel(prop)
    plt.grid()
    plt.show()
