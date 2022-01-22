#!/usr/bin/env python3

# Examples of atmosphere usage.
# Written by: Eric J. Whitney  Last updated: 8 January 2020

from pyavia.units import Dim
from pyavia.aero import Atmosphere

# Set default result units US or SI.  Individual defaults can be set for
# each unit type.
Atmosphere.set_default_style('SI')
Atmosphere.unitdef_press = 'psi'

# Show some ISA SSL values.
atm = Atmosphere(H='SSL')
print(f"ISA S/L -> P = {atm.pressure:.3f}, T = {atm.temperature:.2f}")
# P = 101.325 kPa, T = 288.15 K

# Show density for an ISA standard altitude (note that these are
# formally geopotential altitudes).
atm = Atmosphere(H=Dim(10000, 'ft'))
print(f"H = 10,000' -> ρ = {atm.ρ:.3f}")
# ρ = 0.905 kg/m³

# Show the temperature ratio for a pressure altitude with a temperature offset:
atm = Atmosphere(H_press=Dim(34000, 'ft'), T_offset=Dim(+15, 'Δ°C'))
print(f"Hp = 10,000', T = ISA+15°C -> θ = {atm.theta:.3f}")
# Theta = 0.818

# Show the density ratio for an arbitrary non-standard atmosphere based on
# temperature / pressure.
atm = Atmosphere(P=Dim(90, 'kPa'), T=Dim(-15, '°C'))
print(f"For P = 90 kPa, T = -15°C -> σ = {atm.σ:.3f}")
# σ = 0.991
