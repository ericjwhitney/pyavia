#!/usr/bin/env python3
"""
Example of blade element analysis of a propeller using reference
propeller 5868-9 from NACA Report No. 658. Both fixed pitch and constant
speed solutions are shown.
"""
import numpy as np
import matplotlib.pyplot as plt

from examples.aerodynamics.foil_rotor_example import clark_y_xrotor
from pyavia.util.print_styles import ruled_line
from pyavia.propulsion import BEMPropellerCS
import examples.propulsion.NACA658_ref_prop as ref_prop

# TODO TEMP
from pyavia.propulsion._private._bev_propeller import (
    BEVPropeller, BEVPropellerCS)

# The aerofoil performance model is a fit to Clark-Y data for
# Re = 1.12e6, which is slightly low for this application but in the
# ballpark:
#   - For reference at 0.75R where r = 3.75', c = 0.4605' this would
#     correspond to V = 387.23 fps, M = 0.347, Ω = 103.26 rad/s or
#     986.1 RPM.
#   - The tip would be operating at M = 0.463 / Re = 1.49e6.

from pyavia.numeric.solve import SolverError
from pyavia.profiling import profile_snake

# Written by Eric J. Whitney, January 2023.

d2r = np.deg2rad

# == Propeller Definition ==============================================

fixed_type = BEVPropeller
cs_type = BEVPropellerCS

# Build adjustable pitch propeller and set reference conditions.
# Propeller is initialised 'stopped' with RPM / Ω = 0 to avoid running
# solver straight away.

prop_kwargs = {
    'B': ref_prop.B,  # No. of blades.
    'r': ref_prop.r,  # Radii [ft].
    'c': ref_prop.c,  # Chords [ft].
    'β0_range': ref_prop.β0_range,  # Variable pitch range [rad].
    'βr': ref_prop.βr,  # (Fixed) local incidence angles [rad].
    'foils': clark_y_xrotor,  # Single section for all radii.
    'V0': 0.0,
    'RPM': 0.0,  # Initially stopped.
    'ρ0': ref_prop.ρ,  # Atmosphere.
    'a0': ref_prop.a,
    'μ0': ref_prop.μ,
    'display_level': 3  # Verbosity.

    # Other options that may be varied.
    # 'maxits_flow': 10,
    # 'blade_stall': 'du-selig',  # None, 'du-selig'.
    # 'high_inflow': 'wilson',  # None, 'buhl' or 'wilson'
    # 'tip_losses': 'glauert',  # None or 'glauert'.
    # 'tip_losses': 'modified-glauert',
}


# ----------------------------------------------------------------------


# if DO_β0_CONST:

def β0_sweep(profile: bool = False):
    """
    Sweep the propeller through a range of fixed β0 values.
    """
    prop = fixed_type(**prop_kwargs)

    # -- Loop Over β0 Values -------------------------------------------

    for (β0_deg, data_list), colour in zip(ref_prop.DATA_SETS.items(),
                                           ['g', 'b', 'r', 'y']):

        print(ruled_line(
            f"Fixed pitch analysis for β0 @ 0.75R = {β0_deg:.1f}°"))

        # TODO SimpleNamespace for this.
        # Separate out test data.
        data = np.array(data_list)
        J, η_test = data[:, 0], data[:, 1]
        CT_test, CP_test = data[:, 2], data[:, 3]
        RPM, V0 = data[:, 4], data[:, 5]  # [RPM], [fps]

        # Solve propeller at test data points.
        η_calc = np.zeros_like(J)
        CT_calc, CP_calc = np.zeros_like(J), np.zeros_like(J)
        for j in range(data.shape[0]):

            # The basic momentum method has trouble with V = 0, so a small
            # adjustment is made to stationary points to allow solution.
            V0_j = max(float(V0[j]), 1.0)  # Small positive value [fps].

            if profile:
                # Create closure and run profiler.
                def profiling_run():
                    prop.set_state(β0=d2r(β0_deg), V0=V0_j, RPM=RPM[j])

                profile_snake(profiling_run)

            else:
                # Run normally.
                prop.set_state(β0=d2r(β0_deg), V0=V0_j, RPM=RPM[j])

            η_calc[j] = prop.η
            CT_calc[j] = prop.CT
            CP_calc[j] = prop.CP

            # For plotting purposes, ignore extraneous efficiencies.
            if not (0.0 <= η_calc[j] <= 1.0):
                η_calc[j] = np.nan

        # -- Generate Plots --------------------------------------------

        plt.figure(1)
        plt.plot(J, CT_test, colour + 's', label=f"TEST $β_0$={β0_deg:.1f}")
        plt.plot(J, CT_calc, colour + '-', label=f"CALC $β_0$={β0_deg:.1f}")

        plt.figure(2)
        plt.plot(J, CP_test, colour + 's', label=f"TEST $β_0$={β0_deg:.1f}")
        plt.plot(J, CP_calc, colour + '-', label=f"CALC $β_0$={β0_deg:.1f}")

        plt.figure(3)
        plt.plot(J, η_test, colour + 's', label=f"$η$ TEST $β_0$={β0_deg:.1f}")
        plt.plot(J, η_calc, colour + '-', label=f"$η$ CALC $β_0$={β0_deg:.1f}")
        plt.show(block=False)


# ----------------------------------------------------------------------

def V_sweep():
    """
    Sweep the propeller through a range of airspeeds, holding constant
    Ω. This is designed to generate a typical operating curve to
    overplot the β0 sweep results.
    """
    cs_prop = cs_type(β0=np.deg2rad(35), **prop_kwargs)

    # Modify propulsion and set desired power, speed and airspeed range.
    Ps_hp = 350.0  # [hp]
    Ω_RPM = 1000  # [RPM]
    V0_start = 10.0 if isinstance(cs_prop, BEMPropellerCS) else 0.0
    V0_range = np.linspace(V0_start, 400.0, 40)  # [fps]

    # Loop over velocities.
    print(ruled_line(
        f"Constant speed analysis V = {V0_range[0]:.01f} -> "
        f"{V0_range[-1]:.01f} fps, Ω = {Ω_RPM} RPM, Ps = {Ps_hp:.01f} hp"))

    J_calc, C_T_calc, C_P_calc, η_calc = [], [], [], []
    for V in V0_range:
        try:
            cs_prop.set_state(V0=V, RPM=Ω_RPM,
                              Ps=Ps_hp * 550)  # hp -> [ft.lbf/s]

            J_calc.append(cs_prop.J)
            C_T_calc.append(cs_prop.CT)
            C_P_calc.append(cs_prop.CP)
            η_calc.append(cs_prop.η)

            # For plotting purposes, ignore extraneous efficiencies.
            if not (0.0 <= η_calc[-1] <= 1.0):
                η_calc[-1] = 0.0

        except SolverError:
            print(f"-> Failed to converge point V = {V:.01f}.")

    # -- Generate Plots ------------------------------------------------

    plt.figure(1)
    plt.plot(J_calc, C_T_calc, '-k', label="TYP CONST SPEED")
    plt.xlabel("$J$")
    plt.ylabel("$C_T$")
    plt.grid(axis='both')
    plt.legend(loc='lower left')

    plt.figure(2)
    plt.plot(J_calc, C_P_calc, '-k', label="TYP CONST SPEED")
    plt.xlabel("$J$")
    plt.ylabel("$C_P$")
    plt.grid(axis='both')
    plt.legend(loc='upper right')

    plt.figure(3)
    plt.plot(J_calc, η_calc, '-k', label="TYP CONST SPEED")
    plt.xlabel("$J$")
    plt.ylabel(r"η")
    plt.legend(loc='lower right')
    plt.grid(axis='both')
    plt.show(block=False)


# ======================================================================

if __name__ == '__main__':
    β0_sweep(profile=False)  # These can be commented out as required.
    V_sweep()
    plt.show()

