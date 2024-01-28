#!/usr/bin/env python3
# Example of XROTOR aerofoil performance model.
# Written by: Eric J. Whitney  Last updated: 8 March 2023.

import numpy as np

# noinspection PyUnresolvedReferences
from pyavia.aerodynamics import (QPROP2DAero, XROTOR2DAero,
                                 XROTOR2DAeroPostStall, plot_foil_aero)

d2r, r2d = np.deg2rad, np.rad2deg

example = 'XROTOR'  # 'XROTOR' or 'QPROP'

# -- Aerofoil Definition -----------------------------------------------

clark_y_xrotor = XROTOR2DAeroPostStall(   # Or plain 'XROTOR2DAero'.
    # Approximate fit of Clark-Y properties for Re = 1.12e6.
    α0=d2r(-3.69),  # [rad]
    clα0=6.460,  # [/rad]
    clα_stall=-4.784,  # [/rad]
    cl_max=1.7454,
    cl_min=-1.000,  # Assumed, not relevant to case.
    cl_inc_stall=0.102,
    cd_min=0.00664,
    cl_cdmin=0.420,
    cd_cl2=0.00988,
    Re_ref=1.12e6,
    Re_exp=-0.158, M_crit=0.67, cm_qc=-0.092)

generic_qprop = QPROP2DAero(
    # These are the 'default' aerofoil properties in QPROP.
    cl0=0.5, clα0=5.8,
    cl_min=-0.4,
    cl_max=1.2,
    cd_min=0.028,
    cd_cl2_pos=0.050,
    cd_cl2_neg=0.050,
    cl_cd_min=0.5,
    Re_ref=70000.0,
    Re_exp=-0.7
)

show_states = ['Re', 'M', 'α', 'cl', 'cm_qc', 'cd', 'clα', 'clα0',
               'α_stall_neg', 'α_stall_pos',
               'cl_stall_neg', 'cl_stall_pos']

# ----------------------------------------------------------------------


if __name__ == '__main__':
    print(f"Rotor/Propeller Foil Example:")

    if example == 'XROTOR':
        # Example of XROTOR performance model.
        foil = clark_y_xrotor
        foil.set_states(Re=1.12e6, M=0.0, α=d2r(5.0))  # Operating conds.
        α_range_deg = [-180, +180]  # Angle of attack range to show [°].

    elif example == 'QPROP':
        # Example of QPROP performance model.
        foil = generic_qprop  # Use QPROP performance model.
        foil.set_states(Re=7.0e5, M=0.0, α=d2r(5.0))
        α_range_deg = [-60, +60]

    else:
        raise ValueError(f"Unknown example '{example}'.")

    # -- Print Example Properties --------------------------------------

    print(f"Current Aerofoil Properties:")

    # Angles.
    for prop in show_states:
        if prop[0] == 'α':
            # Angle format.
            print(f"{prop:>18s} = {r2d(getattr(foil, prop)):+10.03f}°")
        else:
            # General format.
            print(f"{prop:>18s} = {getattr(foil, prop):10.5G}")

    # -- Plot Example Curves -------------------------------------------

    if α_range_deg is not None:
        plot_foil_aero(foil, α_start=d2r(α_range_deg[0]),
                       α_end=d2r(α_range_deg[1]),
                       num=int((α_range_deg[1] - α_range_deg[0]) / 0.25),
                       figs=(('α', 'cl', 'cm_qc'), ('cl', 'cd'),
                             ('α', 'clα'),  # Add lift curve slope.
                             ('α', 'cd')),  # Add drag vs. α.
                       title="ROTOR/PROP FOIL EXAMPLE")
