from pyavia.motion.landing import landing_energy
from test.motion.test_landing import typ_tyre_ft_lbf

# Written by Eric J. Whale, November 2023.

# Compute landing load factor and deflection using conservation of energy.
# Note the following in the mass values below:
#   - [slug] is used as the consistent mass unit.
#   - Aircraft mass is divided by 2 for per-leg analysis.

result = landing_energy(strut=2158.3,  # Strut spring K [lbf/ft]
                        tyre=typ_tyre_ft_lbf,  # Tabulated (ft, lbf)
                        g=32.174,  # [ft/s^2]
                        K=2 / 3,  # L/W for light aircraft
                        m_ac=1200 * 0.031081 / 2,  # lbm -> [slug]
                        m_t=5.0 * 0.031081,  # lbm -> [slug]
                        V_sink=7.0)  # [fps]

print(f"Total aircraft weight = {result.W_ac:.1f} lbf, "
      f"lift = {result.L:.1f} lbf")
print(f"Landing sink speed = {result.V_sink:.2f} fps")
print(f"Total stroke: Aircraft = {result.δ_ac:.3f} ft, "
      f"Strut = {result.δ_s:.3f} ft, "
      f"Tyre = {result.δ_t:.3f} ft")
print(f"Total energy: Strut = {result.E_s:.1f} ft-lbf, "
      f"Tyre = {result.E_t:.1f} ft-lbf")
print(f"Total force: Strut = {result.F_s:.1f} lbf, "
      f"Tyre = {result.F_t:.1f} lbf")
print(f"Reaction factor: Strut = {result.N_s:.3f}, "
      f"Tyre = {result.N_t:.3f}")
print(f"Aircraft load factor = {result.N_z:.3f}")
print(f"Efficiency: Strut = {result.η_s * 100:.1f}%, "
      f"Tyre = {result.η_t * 100:.1f}%")
