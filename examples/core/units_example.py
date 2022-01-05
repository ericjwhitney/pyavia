#!/usr/bin/env python3

# Examples of units.
# Written by: Eric J. Whitney  Last updated: 8 January 2019

from pyavia import dim, similar_to

wing_span = dim(10, 'm')  # 10 m^2
chord = 1 * dim('m')  # 1 m^2
wing_area = wing_span * chord
print(f"Wing area = {wing_area} [{wing_area.convert('ft^2'):.5g}]")
air_density = dim(0.002378, 'slug/ft^3')
print(f"Air density = {air_density:5G} "
      f"[{air_density.convert('kg/m^3'):5G}]")
cl_max = 1.35
takeoff_mass = dim(300, 'kg')
print(f"Takeoff mass = {takeoff_mass:.5G} "
      f"[{takeoff_mass.convert('lbm'):.5G}]")
wing_loading = (takeoff_mass * dim('G') / wing_area).convert('N/m²')
print(f"Wing loading = {wing_loading:.5G} "
      f"[{wing_loading.convert('lbf/ft^2'):.5G}]")

stall_speed = (2 * wing_loading / air_density / cl_max) ** 0.5
print(f"Stall speed = {stall_speed:.5G} "
      f"[{stall_speed.convert('ft/s'):.5G}] "
      f"[{stall_speed.convert('kt'):.5G}]")

g_dim, fpss_dim = dim(1, 'G'), dim(1, 'ft/s²')
print(f"Conversion {g_dim} = {g_dim.convert(fpss_dim.units)}")
print(f"Reverse conversion: {fpss_dim} = "
      f"{fpss_dim.convert(g_dim.units)}")

print(f"Units similar to psi are:", similar_to('psi'))

mileage = dim(40, 'rod/US_hhd')
print(f"Grampa Simpson's car gets {mileage}, and that's the way he likes "
      f"it!")
print(f"(In more conventional units this is equal to"
      f" {mileage.convert('mi/US_gal'):.5f} or "
      f"{(1 / mileage).convert('L/100km'):.1f})")
