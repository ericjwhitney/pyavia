from fractions import Fraction
import math

from ._base import (add_unit, add_base_unit, set_conversion,
                    block_conversion)
from ._opts import get_unit_options, set_unit_options

# == Base Unit Definitions =============================================

# -- Mass --------------------------------------------------------------

#  Signatures.
add_base_unit(['t', 'kg', 'g', 'mg', 'μg'], 'M')  # Note: t = Metric tonne.
add_base_unit(['slug', 'lbm'], 'M')


# Conversions.
set_conversion('g', 'μg', fwd=1e6)
set_conversion('kg', 'g', fwd=1000)
set_conversion('g', 'mg', fwd=1000)
set_conversion('lbm', 'kg', fwd=0.45359237)  # Defn Intl & US Standard Pound.
set_conversion('slug', 'lbm', fwd=32.17404855643045)  # CALC of G in ft/s^2
set_conversion('t', 'kg', fwd=1000)  # Metric tonne.

# -- Length ------------------------------------------------------------

#  Signatures.
add_base_unit(['km', 'm', 'cm', 'mm'], 'L')
add_base_unit(['NM', 'sm', 'mi', 'rod', 'yd', 'ft_US', 'ft', 'in'], 'L')
# Notes:
#   - 'sm' is the US survey / statute mile which is a tiny bit longer
#     than the international mile 'mi', because it is defined in US survey
#     feet (below).
#   - 'ft_US' is the US survey foot which is a tiny bit larger than the
#     international foot 'ft'.

# Conversions.
set_conversion('NM', 'm', fwd=1852)  # International NM.
# Note the older UK NM was slightly different = 1853 m (or 6,080 ft).

set_conversion('sm', 'ft_US', fwd=5280)
# 'sm' is the US statute mile defined as 5,280 US survey feet.
# Interestingly, although this corresponds to exactly 1,760 yards in the
# same manner as the international mile ('mi', below) there is no formal
# definition of a 'US survey yard'.

set_conversion('mi', 'yd', fwd=1760)
# 'mi' is is the international mile defined relative to the
# international yard and foot.

set_conversion('km', 'm', fwd=1000)
set_conversion('rod', 'yd', fwd=Fraction(11, 2))  # 1 rod = 5-1/2 yd
set_conversion('yd', 'ft', fwd=3)
set_conversion('yd', 'in', fwd=36)  # To shorten conv. path
set_conversion('m', 'ft', fwd=1000 / 25.4 / 12)  # To shorten conv. path
set_conversion('m', 'cm', fwd=100)
set_conversion('m', 'mm', fwd=1000)
set_conversion('ft_US', 'm', fwd=1200 / 3937)
# US Survey Foot per National Bureau of Standards F.R. Doc. 59-5442.

set_conversion('ft', 'in', fwd=12)  # International foot.
set_conversion('in', 'cm', fwd=2.54)  # To shorten conv. path
set_conversion('in', 'mm', fwd=25.4)  # Defn British, US, industry inch

# -- Time --------------------------------------------------------------

#  Signatures.
add_base_unit(['day', 'hr', 'min', 's', 'ms'], 'T')

# Conversions.
set_conversion('day', 'hr', fwd=24)
set_conversion('hr', 'min', fwd=60)
set_conversion('min', 's', fwd=60)
set_conversion('s', 'ms', fwd=1000)

# -- Temperature -------------------------------------------------------

#  Signatures.
add_base_unit(['°C', 'Δ°C', 'K', 'ΔK'], 'θ')
add_base_unit(['°F', 'Δ°F', '°R', 'Δ°R'], 'θ')

block_conversion('°C')  # Automatic conversions for these offset units ...
block_conversion('°F')  # ... is prohibited; they are handled separately.

# Conversions.  Note: Only conversions for temperature changes are given
# here.  Absolute scale temperatures are handled by a special function
# because of offset scales used by °C and °F.
set_conversion('K', 'Δ°C', fwd=1)
set_conversion('K', 'Δ°F', fwd=Fraction(9, 5))
set_conversion('°R', 'Δ°F', fwd=1)

# -- Amount of Substance -----------------------------------------------

# Signatures.
add_base_unit(['Gmol', 'Mmol', 'kmol', 'mol', 'mmol', 'μmol', 'nmol'], 'N')

# Conversions.
set_conversion('Gmol', 'Mmol', fwd=1000)
set_conversion('Mmol', 'kmol', fwd=1000)
set_conversion('kmol', 'mol', fwd=1000)
set_conversion('mol', 'mmol', fwd=1000)
set_conversion('mmol', 'μmol', fwd=1000)
set_conversion('μmol', 'nmol', fwd=1000)
set_conversion('Gmol', 'mol', fwd=1e9)  # To shorten conv. path.
set_conversion('mol', 'nmol', fwd=1e9)  # To shorten conv. path.

# Electric Current -----------------------------------------------------

# Signatures.
add_base_unit(['A', 'mA'], 'I')

# Conversions.
set_conversion('A', 'mA', fwd=1000)

# -- Luminous intensity ------------------------------------------------

# Signatures.
add_base_unit(['cd'], 'J')

# Conversions.

# -- Plane Angle -------------------------------------------------------

# Signatures.
add_base_unit(['deg', 'rad', 'rev', '°'], 'A')
# This is the only use of the ° symbol on its own, but note that ° !=
# deg for comparison purposes.

# Conversions.
set_conversion('rev', 'rad', fwd=2 * math.pi)
set_conversion('rev', 'deg', fwd=360)
set_conversion('rad', 'deg', fwd=180 / math.pi)
set_conversion('deg', '°', fwd=1)

# -- Solid angle -------------------------------------------------------

# Signatures.
add_base_unit(['sp', 'sr'], 'Ω')

# Conversions.
set_conversion('sp', 'sr', fwd=4 * math.pi)  # 1 spat = 4*pr steradians.

# == Derived Unit Definitions ==========================================

# Notes:
# - A a variety of signature types / operators / unicode (*, /, ×, ²,
#   etc) are used in which acts as an automatic check on the parser.
# - Exact standard for G = 9.80665 m/s/s (WGS-84 defn). Full float
#   conversion gives 32.17404855643045 func/s^2.

# Disable caching of constructed units here (if enabled / default) to
# prevent shorthand RHS expressions below from cluttering the cache.
_restore_caching = get_unit_options().cache_made_units
set_unit_options(cache_made_units=False)



# -- Area --------------------------------------------------------------

add_unit('ha', '10000*m^2')

# -- Volume ------------------------------------------------------------

add_unit('cc', 'cm^3')
add_unit('L', '1000×cm^3')
add_unit('US_gal', '231×in^3')  # Fluid gallon
add_unit('Imp_gal', '4.54609*L')  # British defn
add_unit('US_qt', '0.25*US_gal')  # Fluid quart
add_unit('US_fl_bl', '31.5*US_gal')  # Fluid barrel
add_unit('US_hhd', '2×US_fl_bl')  # Hogshead

# -- Speed -------------------------------------------------------------

add_unit('fps', 'ft/s')
add_unit('kt', 'NM/hr')
add_unit('mph', 'sm/hr')  # Normally defined as US statute miles / hour.
add_unit('kph', 'km/hr')

add_unit('RPM', 'rev/min')

# -- Acceleration ------------------------------------------------------

add_unit('G', '9.80665 m/s^2')  # WGS-84 definition

# -- Force -------------------------------------------------------------

add_unit('kgf', 'kg×G')
add_unit('N', 'kg.m.s⁻²')
add_unit('kN', '1000×N')

add_unit('lbf', 'slug.ft/s²')
add_unit('kip', '1000*lbf')

# -- Pressure ----------------------------------------------------------

#  Signatures.
add_unit('MPa', 'N/mm²')
add_unit('Pa', 'N/m²')
add_unit('hPa', '100.Pa')
add_unit('kPa', '1000*Pa')
add_unit('GPa', '1e9*Pa')

add_unit('atm', '101325 Pa')  # ISO 2533-1975
add_unit('bar', '100000*Pa')
add_unit('mmHg', '133.322387415*Pa')
# mmHg conversion from BS 350: Part 1: 1974 – Conversion factors and
# tables

add_unit('Torr', f'{1 / 760}*atm')

add_unit('psf', 'lbf/ft²')
add_unit('inHg', '25.4*mmHg')
add_unit('psi', 'lbf/in²')
add_unit('ksi', '1000*psi')

# -- Energy ------------------------------------------------------------

add_unit('J', 'N.m')
add_unit('kJ', '1000.J')
add_unit('MJ', '1000.kJ')

add_unit('Wh', '3600.J')
add_unit('kWh', '1e3.Wh')
add_unit('MWh', '1e6.Wh')
add_unit('GWh', '1e9.Wh')
add_unit('TWh', '1e12.Wh')
add_unit('mWh', '1e-3.Wh')
add_unit('μWh', '1e-6.Wh')

add_unit('cal', '4.184×J')  # ISO Thermochemical calorie (cal_th).
add_unit('kcal', '1000.cal')
add_unit('Btu', '778.1723212164716×ft.lbf')  # ISO British Thermal Unit.
# The ISO Btu is defined as exactly 1055.06 J. The above value is the full
# float calculated conversion to ft.lbf.

# -- Power -------------------------------------------------------------

add_unit('W', 'J/s')
add_unit('kW', '1000*W')
add_unit('hp', '550×ft.lbf/s')

# -- Luminous ----------------------------------------------------------

add_unit('lm', 'cd.sr')  # Lumen.

# ======================================================================
# Return status of caching for made units after adding derived types.
set_unit_options(cache_made_units=_restore_caching)
