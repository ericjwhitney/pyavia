"""
Atmospheric property calculations including the ISA.

Contains:
    Atmosphere      Class giving fixed atmospheric properties.
    geo_alt_to_pot  Function converting geometric altitude to geopotential.
    pot_alt_to_geo  Function converting geopotential altitude to geometric.

Notes:

- Uses ISO 2533-1975 International Standard Atmsosphere (ISA).  Altitude
from -2 → +80 km

- The ISA is defined in terms of geopotential altitude (H) which is
convenient when computing the pressure distribution through the depth of
the atmosphere.  This assumes a constant value of gravitational
acceleration everywhere.  In reality, gravitational acceleration falls
slightly as geometric (actual) altitude (h) increases.  H and h are
related by an equation; use geo_alt_to_pot() and pot_alt_to_geo() for
conversion.  Unless noted otherwise, all altitudes used in Atmosphere are
geopotential (H).

- Humidity is not presently included.

- Dynamic viscosity is computed via Sutherland's formula which is reasonably
accurate between 180°R - 3400°R (or 100K - 1889K, -173 - 1616°C, ref
NACA TN 1135).

- Uppercase and lowercase are mixed to be consistent with source
documents.  Uppercase H and associated values refer to geopotential
altitudes, lowercase refer to geometric (but these don't appear together
anyway so as to prevent errors).

- Some non-ASCII characters are used.
"""

# Last updated: 5 December 2019 by Eric J. Whitney

from math import exp, log, isclose
from units import Dim, Units, make_total_temp
from util import bracket_list, bisect_root

__all__ = ['Atmosphere', 'geo_alt_to_pot', 'pot_alt_to_geo']


# -----------------------------------------------------------------------------

# noinspection PyPep8Naming
class Atmosphere:
    """
    An Atmosphere class gives a fixed representation of atmospheric
    conditions.  Once constructed, all of the regular assosicated atmospheric
    properties are then made available through class properties.

    Example:
        # Set default result units as US or SI.  Individual defaults can be
        # set for each unit type.
        >>> Atmosphere.set_default_style('SI')
        >>> Atmosphere.unitdef_press = 'psi'

        # Show some ISA SSL values.
        >>> atm = Atmosphere('SSL')
        >>> print(f"P = {atm.pressure:.3f}, T = {atm.temperature:.2f}")
        P = 14.696 psi, T = 288.15 K

        # Show density for an ISA standard altitude (note that these are
        # formally geopotential altitudes).
        >>> atm = Atmosphere(H=Dim(10000, 'ft'))
        >>> print(f"ρ = {atm.ρ:.3f}")
        ρ = 0.905 kg/m³

        # Show the temperature ratio for a pressure altitude with a
        # temperature offset:
        >>> atm = Atmosphere(H_press=(34000,'ft'), T_offset=(+15,'Δ°C'))
        >>> print(f"Theta = {atm.theta:.3f}")
        Theta = 0.818

        # Show the density ratio for an arbitrary non-standard atmosphere
        # based on temperature / pressure.
        >>> atm = Atmosphere(P=(90, 'kPa'), T=(-15, '°C'))
        >>> print(f"σ = {atm.σ:.3f}")
        σ = 0.991
    """

    def __init__(self, ssl_str: str = None, **kwargs):
        """
        To construct an ISA standard sea level atmosphere:
            'SSL' -> ISA standard sea level. Note: This is the only valid
            positional argument.

        To construct all other atmospheres, the following keyword argument
        combinations can be used.  All arguments requiring a dimension must
        be passed as Dim objects, or a 2-tuple (value, 'units') which will
        be converted into a Dim object on the fly:
            H= -> ISA standard atmosphere corresponding to a given
            geopotential altitude.  This is generally what is used in most
            texts (assumes gravity fixed across altitudes).  Note the
            uppercase  H.

            h_geometric= -> ISA standard atmosphere corresponding to a given
            geometric altitude.

            T=, P= -> Arbitrary atmosphere with given temperature and
            pressure.

            H_press=, T= -> Arbitrary atmosphere based on pressure
            altitude (altimeter set to 1013 mb / 29.92 in-Hg) and a given
            temperature.

            H_press=, T_offset= -> Same as H_press=, T_= except temperature
            is given as an offset from the ISA standard value at that
            pressure altitude.
        """
        # Any keywords with 2-element tuples can be swapped to Dim objects
        # taking them as (value, 'units').
        for key, val in kwargs.items():
            if isinstance(val, tuple) and len(val) == 2:
                kwargs[key] = Dim(*val)

        if ssl_str is not None:
            if ssl_str == 'SSL' and not kwargs:
                # Set ISA standard sea level atmosphere.
                tmp = Atmosphere(H=Dim(0, 'm'))
                self._T, self._P = tmp._T, tmp._P
                return
            else:
                raise TypeError(f"Unknown argument: {ssl_str}")

        if len(kwargs) == 1:
            if 'H' in kwargs:
                # Set an ISA atmosphere based on geopotential altitude H.
                # Find the corresponding layer and compute properties
                # from the base.
                H = kwargs['H']
                base_idx, _ = bracket_list(_ISA_HTPrho_b,
                                           [H, None, None, None],
                                           key=lambda x: x[0])
                H_b, T_b, P_b, _ = _ISA_HTPrho_b[base_idx]
                beta = _ISA_beta[base_idx]
                self._T = T_b + beta * (H - H_b)
                self._P = _press_in_layer(H, P_b, H_b, T_b, beta)
                return

            if 'h_geometric' in kwargs:
                # Set an ISA atmosphere based on geometric altitude h.
                tmp = Atmosphere(H=geo_alt_to_pot(kwargs['h_geometric']))
                self._T, self._P = tmp._T, tmp._P
                return

        if len(kwargs) == 2:
            if kwargs.keys() == {'T', 'P'}:
                # Set arbitrary atmosphere via temperature and pressure.
                self._T, self._P = kwargs['T'], kwargs['P']
                self._T = make_total_temp(self._T)  # -> total & check

                if self._P.value <= 0 or self._T.value <= 0:
                    raise ValueError("Invalid T or P.")
                return

            if kwargs.keys() == {'H_press', 'T'}:
                # Set an arbitrary atmosphere based on pressure altitude and
                # given temperature.
                H_press, self._T = kwargs['H_press'], kwargs['T']
                self._T = make_total_temp(self._T)  # -> total & check

                # Find the bracketing layer based on ISA pressure.  Get the
                # standard pressure.
                base_idx, _ = bracket_list(_ISA_HTPrho_b,
                                           [H_press, None, None, None],
                                           key=lambda x: x[0])
                H_b, T_b, P_b, _ = _ISA_HTPrho_b[base_idx]
                beta = _ISA_beta[base_idx]
                self._P = _press_in_layer(H_press, P_b, H_b, T_b, beta)
                return

            if kwargs.keys() == {'H_press', 'T_offset'}:
                # Set an arbitrary atmosphere based on pressure altitude and
                # temperature offset from ISA.
                H_press, T_offset = kwargs['H_press'], kwargs['T_offset']
                if T_offset.units in [Units('°C'), Units('°F')]:
                    raise ValueError(f"ISA offset temperature must be in "
                                     f"total or Δ units.")

                # Find the bracketing layers based on ISA pressure.  Get the
                # standard pressure and temperature.  Add the offset to
                # the standard temperature.
                base_idx, _ = bracket_list(_ISA_HTPrho_b,
                                           [H_press, None, None, None],
                                           key=lambda x: x[0])
                H_b, T_b, P_b, _ = _ISA_HTPrho_b[base_idx]
                beta = _ISA_beta[base_idx]
                self._P = _press_in_layer(H_press, P_b, H_b, T_b, beta)
                self._T = (T_b + beta * (H_press - H_b)) + T_offset
                return

        raise TypeError(f"Incorrect arguments: {', '.join(kwargs.keys())}")

    # Class constants.
    R = Dim(287.05287, 'J/K/kg')  # Gas constant for air.

    # Output defaults.
    unitdef_alt = 'ft'
    unitdef_dens = 'slug/ft^3'
    unitdef_kine = 'ft²/s'
    unitdef_press = 'psf'
    unitdef_spd = 'ft/s'
    unitdef_spheat = 'ft.lbf/slug/°R'
    unitdef_temp = '°R'
    unitdef_visc = 'psf.s'

    @classmethod
    def set_default_style(cls, style: str = 'US'):
        """Setting to give default results in style='US' or 'SI' units."""
        if style == 'US':
            cls.unitdef_alt = 'ft'
            cls.unitdef_dens = 'slug/ft³'
            cls.unitdef_kine = 'ft²/s'
            cls.unitdef_press = 'psf'
            cls.unitdef_spd = 'ft/s'
            cls.unitdef_spheat = 'ft.lbf/slug/°R'
            cls.unitdef_temp = '°R'
            cls.unitdef_visc = 'psf.s'  # Also = slug/ft/s.
        elif style == 'SI':
            cls.unitdef_alt = 'm'
            cls.unitdef_dens = 'kg/m³'
            cls.unitdef_kine = 'm²/s'
            cls.unitdef_press = 'kPa'
            cls.unitdef_spd = 'm/s'
            cls.unitdef_spheat = 'J/kg/K'
            cls.unitdef_temp = 'K'
            cls.unitdef_visc = 'Pa.s'
        else:
            raise ValueError(f"Unknown default style: {style}")

    # Properties.

    # @property
    # def c_p(self) -> Dim:
    #     """Specific heat at constant pressure.  Only corrected above T =
    #     550°R."""
    #     temp_R = self.temperature.convert('°R').value
    #     if temp_R <= 550:
    #         return _C_P_PERF.convert(Atmosphere.unitdef_spheat)
    #
    #     # Correct c_p.
    #     ratio = 5500 / temp_R  # Ratio (Theta) = 5,500°R / T
    #     c_p_corr = _C_P_PERF * (1 + ((_GAMMA_PERF - 1) / _GAMMA_PERF) * (
    #         (ratio ** 2) * exp(ratio) / (exp(ratio) - 1) ** 2))
    #     return c_p_corr.convert(Atmosphere.unitdef_spheat)
    #
    # @property
    # def c_v(self) -> Dim:
    #     """Specific heat at constant volume.  Only corrected above T =
    #     550°R."""
    #     temp_R = self.temperature.convert('°R').value
    #     if temp_R <= 550:
    #         return _C_V_PERF.convert(Atmosphere.unitdef_spheat)
    #
    #     # Correct c_v.
    #     ratio = 5500 / temp_R  # Ratio (Theta) = 5,500°R / T
    #     c_v_corr = _C_V_PERF * (1 + (_GAMMA_PERF - 1) * (
    #             (ratio ** 2) * exp(ratio) / (exp(ratio) - 1) ** 2))
    #     return c_v_corr.convert(Atmosphere.unitdef_spheat)

    @property
    def delta(self) -> float:
        """δ = P / P_SSL."""
        return self._P / Atmosphere('SSL').pressure

    @property
    def density(self) -> Dim:
        return (self._P / Atmosphere.R / self._T).convert(
            Atmosphere.unitdef_dens)

    @property
    def density_altitude(self) -> Dim:
        """After finding the correct altitude range, density altitude is
        computed using a root bisection method, as it is non-linear."""
        dens_reqd = self.density
        l_idx, r_idx = bracket_list(_ISA_HTPrho_b,
                                    [None, None, None, dens_reqd],
                                    key=lambda x: x[3])
        H_lhs, H_rhs = _ISA_HTPrho_b[l_idx][0], _ISA_HTPrho_b[r_idx][0]
        H_units = H_lhs.units
        H_lhs, H_rhs = H_lhs.value, H_rhs.value

        def density_err(H_try: float) -> float:
            return Atmosphere(H=Dim(H_try, H_units)).density.convert(
                dens_reqd.units).value - dens_reqd.value

        maxits = 50
        H_d = bisect_root(density_err, H_lhs, H_rhs, maxits,
                          tol=1e-6)

        return Dim(H_d, H_units).convert(Atmosphere.unitdef_alt)

    @property
    def dynamic_viscosity(self) -> Dim:
        """Dynamic viscosity (μ) also just called 'viscosity'.  It is
        computed using the empirical Sutherland equation shown in ISO
        2533-1975 Eqn 22.  It is invalid for very high or low temperatures and
        conditions at altitudes above 90 km."""

        # Constants s, beta_s from  ISO 2533-1975 Table 1.
        s = Dim(110.4, 'K')
        beta_s = Dim(1.458E-06, 'kg/m/s/K^0.5')
        return (beta_s * self._T ** 1.5 / (self._T + s)).convert(
            Atmosphere.unitdef_visc)

    @property
    def kinematic_viscosity(self) -> Dim:
        """Kinematic viscosity ν = μ / ρ."""
        return (self.dynamic_viscosity / self.density).convert(
            Atmosphere.unitdef_kine)

    @property
    def gamma(self) -> float:
        """γ = c_p / c_v ratio of specific heats.  Only corrected above T =
        550°R."""
        if self.temperature.convert('°R').value <= 550:
            return _GAMMA_PERF
        else:
            return self.c_p / self.c_v

    @property
    def pressure(self) -> Dim:
        return self._P.convert(Atmosphere.unitdef_press)

    @property
    def pressure_altitude(self) -> Dim:
        # Find the ISA level of next highest pressure.
        base_idx, _ = bracket_list(_ISA_HTPrho_b, [None, None, self._P, None],
                                   key=lambda x: x[2])
        H_b, T_b, P_b, _ = _ISA_HTPrho_b[base_idx]
        beta = _ISA_beta[base_idx]
        H_p = _alt_in_layer(self._P, P_b, H_b, T_b, beta)
        return H_p.convert(Atmosphere.unitdef_alt)

    @property
    def sigma(self) -> float:
        """σ = ρ / ρ_SSL."""
        return self.density / Atmosphere('SSL').density

    @property
    def speed_of_sound(self) -> Dim:
        """a = sqrt(γRT)."""
        return ((self.gamma * Atmosphere.R * self._T) ** 0.5).convert(
            Atmosphere.unitdef_spd)

    @property
    def temperature(self) -> Dim:
        return self._T.convert(Atmosphere.unitdef_temp)

    @property
    def theta(self) -> float:
        """θ = T / T_SSL."""
        return self._T / Atmosphere('SSL').T

    # Method aliases.
    P = pressure
    T = temperature
    a = speed_of_sound

    δ = delta
    γ = gamma
    θ = theta
    μ = dynamic_viscosity
    ν = kinematic_viscosity
    ρ = density
    σ = sigma


# -----------------------------------------------------------------------------

# Related constants.

G_N = Dim(9.80665, 'm/s^2')  # Gravitational acceleration
R_EARTH = Dim(6356.766, 'km')  # Earth radius (nom) from ISO 2533-1975 §2.3

# Specific heats are from the American Meteorological Society and are for
# dry air (±2.5 J/kg/K).  Atmosphere class corrects these above T = 550°R.
_C_P_PERF = Dim(1005.7, 'J/kg/K')
_C_V_PERF = Dim(719, 'J/kg/K')
_GAMMA_PERF = 1.4  # γ = Ratio of specific heats c_p/c_v.


# -----------------------------------------------------------------------------

# noinspection PyPep8Naming
def _press_in_layer(H, P_b, H_b, T_b, beta):
    """Returns the pressure computed within layer of constant lapse rate
    given geopotential altitude and base layer values of (_b) of pressure,
    altitude, temperature and lapse rate."""
    if not isclose(beta, 0, abs_tol=1e-6):
        # ISO 2533-1975 Eqn 12.
        return P_b * ((1 + (beta / T_b) * (H - H_b)) ** (
                -G_N / beta / Atmosphere.R))
    else:
        # ISO 2533-1975 Eqn 13.
        return P_b * exp((-G_N / Atmosphere.R / T_b) * (H - H_b))


# noinspection PyPep8Naming
def _alt_in_layer(P, P_b, H_b, T_b, beta):
    """Returns the altitude computed within layer of constant lapse rate
    given pressure and base layer values of (_b) of pressure,
    altitude, temperature and lapse rate."""
    if not isclose(beta, 0, abs_tol=1e-6):
        return H_b + ((P / P_b) ** (-beta * Atmosphere.R / G_N) - 1) * (
                T_b / beta)
    else:
        return H_b - log(P / P_b) * (Atmosphere.R * T_b / G_N)


_ISA_HTPrho_b, _ISA_beta = None, None


# noinspection PyPep8Naming
def _set_ISA_levels():
    global _ISA_HTPrho_b, _ISA_beta

    # ISA altitude levels. Columns are:
    # - H_b: Geopotential altitude from ISO 2533-1975 Table 4.
    # - T_b: Temperature from ISO 2533-1975 Table 4.
    # - P_b: Pressure computed later from each lower base, except H_b = 0
    # (p = 101325 Pa From ISO 2533-1975 Table 1) and H < 0 (computed from
    # above).
    # - rho_b: Density computed from T, P. Only needed for density altitude
    # method.
    #
    # Note: These values are subscript _b meaning they apply at the base of the
    # next range of constant lapse rate.

    _ISA_HTPrho_b = [[Dim(-2.00, 'km'), Dim(301.15, 'K'), None],
                     [Dim(0.00, 'km'), Dim(288.15, 'K'), Dim(101325, 'Pa')],
                     [Dim(11.00, 'km'), Dim(216.65, 'K'), None],
                     [Dim(20.00, 'km'), Dim(216.65, 'K'), None],
                     [Dim(32.00, 'km'), Dim(228.65, 'K'), None],
                     [Dim(47.00, 'km'), Dim(270.65, 'K'), None],
                     [Dim(51.00, 'km'), Dim(270.65, 'K'), None],
                     [Dim(71.00, 'km'), Dim(214.65, 'K'), None],
                     [Dim(80.00, 'km'), Dim(196.65, 'K'), None]]

    # Compute beta (lapse rate) from H_b-T_b levels assuming a constant lapse
    # rate, which is the same as ISO 2533-1975 Table 4.
    _ISA_beta = [(top[1] - bot[1]) / (top[0] - bot[0]) for bot, top in
                 zip(_ISA_HTPrho_b, _ISA_HTPrho_b[1:])]

    # Set level P_b (pressures).  Set H < 0 value from sea level above.
    H_b, H = _ISA_HTPrho_b[1][0], _ISA_HTPrho_b[0][0]
    T_b = _ISA_HTPrho_b[1][1]
    p_b = _ISA_HTPrho_b[1][2]
    beta = _ISA_beta[1]
    _ISA_HTPrho_b[0][2] = _press_in_layer(H, p_b, H_b, T_b, beta)

    # Set remaining pressures up from sea level.
    for idx in range(2, len(_ISA_HTPrho_b)):
        H_b, H = _ISA_HTPrho_b[idx - 1][0], _ISA_HTPrho_b[idx][0]
        T_b = _ISA_HTPrho_b[idx - 1][1]
        p_b = _ISA_HTPrho_b[idx - 1][2]
        beta = _ISA_beta[idx - 1]
        _ISA_HTPrho_b[idx][2] = _press_in_layer(H, p_b, H_b, T_b, beta)

    # Append density values.
    for idx in range(len(_ISA_HTPrho_b)):
        P, T = _ISA_HTPrho_b[idx][2], _ISA_HTPrho_b[idx][1]
        _ISA_HTPrho_b[idx][:] += [P / Atmosphere.R / T]


_set_ISA_levels()


def geo_alt_to_pot(geo_alt):
    """Convert h -> H per ISO 2533-1975 Eqn 8."""
    return R_EARTH * geo_alt / (R_EARTH + geo_alt)


def pot_alt_to_geo(pot_alt):
    """Convert H -> h per ISO 2533-1975 Eqn 9."""
    return R_EARTH * pot_alt / (R_EARTH - pot_alt)
