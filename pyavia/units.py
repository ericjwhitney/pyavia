"""
Units-aware calculations, including commonly used factory function ``dim()``
and other associated functions.

Examples
--------

Creation of dimensioned values is done via ``dim()`` in a natural way,
and gives a ``Dim`` object as a result.  See documentation for ``dim()``
for details on valid formats.

>>> r = dim(10, 'ft')
>>> ω = dim(30, 'rad/s')
>>> r * ω  # Centripetal motion v_t = r.ω
Dim(300, 'fps')

Calling dim() with no arguments results in a dimensionless unity value.

>>> dim()
Dim(1, '')

This gives the expected result of dimensions dropping out in normal
calculations:
>>> 6 * 7 * dim()
42

The ``convert`` function can be used with normal numeric inputs to give
a conversion.  Note that the we try to preserve the type of the input
argument where possible (int to int, float to float, etc):
>>> convert(288, 'in^2', 'ft^2')  # Convert sq. in to sq. ft.
2

Alternatively, ``Dim`` objects can provide a converted value of themselves
directly:
>>> area = dim(144, 'in^2')
>>> area.convert('ft^2')
Dim(1, 'ft^2')

Derived (compound) units will cancel out where possible to provide the
most direct base unit related to the LHS.  For example, because 1 N = 1
kg.m/s², then under 1 G acceleration:

>>> dim(9.80665, 'N') / dim(1, 'G').convert('ft/s^2')
Dim(0.9999999999999999, 'kg')

Units with unusual powers are handled without difficulty:

>>> k_ic_metric = dim(51.3, 'MPa.m⁰ᐧ⁵')  # Metric fracture toughness.
>>> k_ic_metric.convert('ksi.in^0.5')  # Convert to imperial.
Dim(46.68544726800077, 'ksi.in^0.5')

Temperature values are a special case and specific rules exist for handling
them.  This is because different temperatures can have offset scales or
represent different quantities:

    - `Absolute` or `Offset` scales:  `Absolute` temperatures have their zero
      values located at absolute zero whereas `offset` temperatures use
      some other reference point for zero.
    - Represent `Total` values or `Change`/`Δ`:  `Total` temperatures
      represent the actual temperature state of a body, whereas `Δ` values
      represent the change in temperature (or difference between two total
      temperatures).

What the different temperature units can represent can be summarised as
follows:

    +------------------+----------+--------+-------+--------+
    |                  |     Total Scale   | Can Represent  |
    | Temperature Unit +-------------------+----------------+
    |                  | Absolute | Offset | Total | Change |
    +==================+==========+========+=======+========+
    |        K         |    Yes   |   No   |  Yes  |   Yes  |
    +------------------+----------+--------+-------+--------+
    |        °R        |    Yes   |   No   |  Yes  |   Yes  |
    +------------------+----------+--------+-------+--------+
    |        °C        |    No    |   Yes  |  Yes  |   No   |
    +------------------+----------+--------+-------+--------+
    |        °F        |    No    |   Yes  |  Yes  |   No   |
    +------------------+----------+--------+-------+--------+
    |       Δ°C        |    ---   |   ---  |   No  |   Yes  |
    +------------------+----------+--------+-------+--------+
    |       Δ°F        |    ---   |   ---  |   No  |   Yes  |
    +------------------+----------+--------+-------+--------+

Note that although Δ°C and Δ°F are not considered as existing on a total
temperature scale, they can be added or subtracted from total temperatures.
As such these can be used with either absolute or offset total temperature
units, because there is a direct conversion available to an appropriate
type e.g.  Δ°C -> K, Δ°F -> °R.

A simple example converting from an absolute temperature scale to an offset
scale:

>>> dim(373.15, 'K').convert('°F')  # Offset temperature
Dim(211.99999999999994, '°F')

Temperature changes can be added or subtracted from total temperatures,
giving total temperatures:
>>> dim(25, '°C') + dim(5, 'Δ°C')
Dim(30, '°C')
>>> dim(25, '°C') - dim(5, 'Δ°C')
Dim(20, '°C')

Two total temperatures on offset scales can be subtracted, but in this case
they give a temperature change which is considered to be a distinct unit:

>>> dim(32, '°F') - dim(0, '°C')  # Result will be approx. zero.
Dim(5.684341886080802e-14, 'Δ°F')

Total temperatures on absolute scales or temperature changes can be used in
derived / compound units and converted freely:

>>> air_const_metric = dim(287.05287, 'J/kg/K')
>>> air_const_imp_slug = air_const_metric.convert('ft.lbf/slug/°R')

Total temperatures on offset scales are not allowed to be used in derived /
compound units:

>>> dim('km.°F')  # doctest: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
ValueError: Offset temperature units are only permitted to be base units.

``Dim`` objects supports any format statements that can be used directly on
their numeric value:
>>> print(f"Slug value = {air_const_imp_slug:.2f}")
Slug value = 1716.56 ft.lbf/slug/°R
"""

from __future__ import annotations
from collections import namedtuple
from fractions import Fraction
import math
import operator
import re
from numbers import Number
from typing import Callable, Optional, Union
import warnings

import numpy as np

from pyavia.containers import MultiBiDict, WtDirGraph, wdg_edge
from pyavia.math import kind_div
from pyavia.types import coax_type, force_type

__all__ = ['OUTPUT_UCODE_PWR', 'STD_UNIT_SYSTEM', 'CACHE_COMPUTED_CONVS',
           'CONV_PATH_LENGTH_WARNING', 'Dim', 'RealScalar', 'RealArray',
           'DimScalar', 'DimArray', 'add_base_unit', 'add_unit', 'convert',
           'dim', 'is_dimarray', 'set_conversion', 'to_absolute_temp']

# -----------------------------------------------------------------------------


CACHE_COMPUTED_CONVS = True
"""Computed conversions are cached for faster repeat access. To ensure
accurate handling of values only the requested conversion 'direction' is
cached; the reverse conversion is not automatically computed and would
need to be separately cached when encountered."""

# Note: There is presently no size limit on this cache. This is normally not
# a problem as only a few types of conversions occur in any application.

CONV_PATH_LENGTH_WARNING = 4
"""Issue a warning if a unit conversion traces a path longer than 
CONV_PATH_LENGTH_WARNING.  It could indicate a potential buildup of error 
and usually means that a suitable conversion for these units should be 
added. """

OUTPUT_UCODE_PWR = True
"""Generate unicode chars for power values in strings."""

STD_UNIT_SYSTEM = 'kg.m.s.K.mol.A.cd.rad.sr'
"""This variable represents a standard system used as the default value for 
any calls to Dim.to_system().  Some unit types may be omitted with default 
values assumed - refer to the documentation for ``Dim.to_real_sys()`` for 
details. """


# -- Type Definitions ---------------------------------------------------------

class Dim(namedtuple('Dim', ['value', 'units'])):
    """
    ``Dim`` represents a dimensioned quantity, consisting of a value and a
    string giving the associated units.  A ``Dim`` object can be used in
    mathematical expressions and the units conversions are automatically
    done to make the result 'units aware'.

    ``Dim`` objects are implemented as a namedtuple and are thus nominally
    immutable.  This means that they can be treated like scalars,
    and their `value` and `units` fields cannot simply be changed or
    overwritten by other functions once they are created.

    .. note:: ``Dim`` objects are not normally created by the user.
       Refer to factory function ``dim()`` for normal construction methods.
    """

    # -- Unary Operators ------------------------------------------------------

    def __abs__(self):
        return Dim(abs(self.value), self.units)

    def __float__(self):
        """
        Returns float(self.value).

        .. note:: Units are removed and checking ability is lost.
        """
        return float(self.value)

    def __int__(self):
        """
        Returns int(self.value).

        .. note:: Units are removed and checking ability is lost.
        """
        return int(self.value)

    def __neg__(self):
        return Dim(-self.value, self.units)

    def __round__(self, n=None):
        return Dim(round(self.value, n), self.units)

    # -- Binary Operators -----------------------------------------------------

    def __add__(self, rhs: DimArray) -> DimArray:
        r"""
        Add two dimensioned values.  If `rhs` is an ordinary value,
        it is promoted to ``Dim`` before addition.  This allows for the
        case of a dimensionless intermediate product in expressions.


        Addition of isolated temperatures is only permitted in some cases
        and the result depends on the combination of units involved.  Total
        temperatures (e.g. K, °C) and temperature changes (e.g. ΔK, Δ°C) are
        handled differently the result will be as follows:

            +-----------------+---------------+-----------------+
            |                 |             (+) RHS             |
            +       LHS       +---------------+-----------------+
            |                 | Total (K/°C)  | Change (ΔK/Δ°C) |
            +=================+===============+=================+
            |   Total (K/°C)  | Not Permitted |      Total      |
            +-----------------+---------------+-----------------+
            | Change (ΔK/Δ°C) |     Total     |     Change      |
            +-----------------+---------------+-----------------+
        """
        if not isinstance(rhs, Dim):
            # Promote to Dim. Addition to a dimless value would only be
            # valid if we were dimless ourselves anyway.
            rhs = Dim(rhs, '')

        err_str = f"Addition '{self.units}' + '{rhs.units}' is not allowed."

        # Handle special cases of temperature addition.
        if self.is_total_temp():
            if rhs.is_temp_change():
                # Case: Total + Δ -> Total
                # Units are adopted from the LHS.
                if self.is_absolute_temp():
                    rhs = rhs.convert(self.units)
                else:
                    rhs = rhs.convert('Δ' + self.units)

                return Dim(self.value + rhs.value, self.units)

            else:
                raise ValueError(err_str)

        if self.is_temp_change():
            if rhs.is_total_temp():
                # Case: Δ + Total -> Total
                # Units are adopted from the total version of the LHS, which
                # is what the RHS is converted into prior to addition.
                rhs = rhs.convert(self.units[1:])  # By dropping Δ from LHS.
                return Dim(self.value + rhs.value, rhs.units)

            elif rhs.is_temp_change():
                # Case: Δ + Δ -> Δ
                rhs = rhs.convert(self.units)
                return Dim(self.value + rhs.value, self.units)

            else:
                raise ValueError(err_str)

        # Remaining cases can be handled by normal conversion machinery.
        rhs = rhs.convert(self.units)
        res_value = self.value + rhs.value
        res_value = coax_type(res_value, type(self.value + rhs.value),
                              default=res_value)

        res = Dim(res_value, self.units)
        if not res.is_dimless():
            return res
        else:
            return res.value  # Units fell off.

    def __sub__(self, rhs: DimArray) -> DimArray:
        r"""
        Subtract two dimensioned values.  Follows the same rules as
        __add__, with differences noted below.

            +-----------------+---------------+-----------------+
            |                 |           \- RHS                |
            +       LHS       +---------------+-----------------+
            |                 | Total (K/°C)  | Change (K/Δ°C)  |
            +=================+===============+=================+
            |   Total (K/°C)  |  Change (\*)  |     Total       |
            +-----------------+---------------+-----------------+
            |  Change (K/Δ°C) | Not Permitted |     Change      |
            +-----------------+---------------+-----------------+

            (*) For this case both temperatures are required to already be on
            the same scale.  Because absolute temperatures can also represent
            temperature changes, not requiring this results in an ambiguous
            conversion target.  Offset scale temperatures are converted to an
            absolute scale prior to subtraction to give correct results.

        Subtraction of isolated temperatures is only permitted in some cases
        and the result depends on the combination of units involved.  Total
        temperatures (e.g. K, °C) and temperature changes (e.g. ΔK, Δ°C) are
        handled differently the result will be as follows:

        Notice that the result of subtracting temperature type units is
        unsymmetric / not commutative.
        """
        if not isinstance(rhs, Dim):
            # Promote to Dim. Subtracting a dimless value would only be
            # valid if we were dimless ourselves anyway.
            rhs = Dim(rhs, '')

        # Handle special cases of temperature subtraction.
        if self.is_total_temp():
            if rhs.is_total_temp():
                # Case: Total - Total -> Δ
                # This case requires the temperatures to already be on the
                # same scale.
                if self.units != rhs.units:
                    raise ValueError(f"Total temperatures must be on the same "
                                     f"scale for subtraction, got: "
                                     f"'{self.units}' - '{rhs.units}'")

                if self.is_absolute_temp():
                    # Simple case of K, °R.
                    return Dim(self.value - rhs.value, self.units)

                # Offset case requires temperatures to be converted to absolute
                # scales prior to subtraction.
                lhs = to_absolute_temp(self)  # Do subtraction in matching ...
                rhs = rhs.convert(lhs.units)  # ... absolute units.
                res = Dim(lhs.value - rhs.value, lhs.units)

                # Return to the original type of unit on Δ scale.
                return res.convert('Δ' + self.units)

            elif rhs.is_temp_change():
                # Case: Total - Δ -> Total
                # Match scales.
                if self.is_absolute_temp():
                    rhs = rhs.convert(self.units)
                else:
                    rhs = rhs.convert('Δ' + self.units)

                return Dim(self.value - rhs.value, self.units)

            # Other units will fall through to give a conversion error.

        if self.is_temp_change():
            if rhs.is_temp_change():
                # Case: Δ - Δ -> Δ
                rhs = rhs.convert(self.units)  # Match scales.
                return Dim(self.value - rhs.value, self.units)

            # Other units will fall through to give a conversion error.

        # Remaining cases can be handled by normal conversion machinery.
        rhs = rhs.convert(self.units)
        res_value = self.value - rhs.value
        res_value = coax_type(res_value, type(self.value - rhs.value),
                              default=res_value)

        res = Dim(res_value, self.units)
        if not res.is_dimless():
            return res
        else:
            return res.value  # Units fell off.

    def __mul__(self, rhs: DimArray) -> DimArray:
        """
        Multiply two dimensioned values.   Rules are as follows:

            - ``Dim`` * ``Dim``:  Typical case, see general procedure below.
            - ``Dim`` * ``Scalar``:  Returns a ``Dim`` object retaining the
               units string of the LHS argument, with the value simply
               multiplied by the scalar.  This means that *effectively*
               'dimensionless' units (e.g. radians, m/m, etc) can be
               *retained*.  If there are no *actual* units (i.e. units =
               '') then these are dropped.
            - ``Scalar`` * ``Dim``:  ``__rmul__`` case.  The LHS argument is
              promoted to a ``Dim`` object.  This is different to the
              ``__mul__`` case above, because if resulting angular units
              are radians or steradians, these will be *dropped* as they are
              dimensionless and have served their purpose in the
              multiplication of two dimensioned values.

        .. note:: If either argument is an offset temperature base unit
           (°C, °F), this is allowed however it is first converted to an
           absolute scale before multiplying.  Offset temperatures can't be
           part of derived units as there is no way to multiply them out.

        General multiplication procedure:  The general principle is that
        when any units are different between the two arguments, the LHS
        argument is given priority. Multiplication of two ``Dim``
        objects is acheived by multiplying their unit bases (except for
        offset temperature base units - see below).  If either ``self`` or
        ``rhs`` units have a k-factor, this is multiplied out and becomes
        part of `self.value` leaving `k` = 1 for both ``self`` and ``rhs``.
        """
        return _dim_mul_generic(self, rhs, operator.mul)

    def __truediv__(self, rhs: DimArray) -> DimArray:
        """
        Divide two dimensioned values.  The same rules as __mul__ apply.
        """
        if not isinstance(rhs, Dim):
            return Dim(self.value / rhs, self.units)

        return self * (rhs ** -1)

    def __matmul__(self, rhs: DimArray) -> DimArray:
        """
        Matrix multiply operator, handled in the same fashion as __mul__. 
        Note that this applies to amn array wrapped in a ``Dim()`` object, 
        not an array *of* ``Dim()`` objects. 
        """
        return _dim_mul_generic(self, rhs, operator.matmul)

    def __pow__(self, pwr: RealScalar) -> DimArray:
        """
        Raise dimensioned value to a power. Any `k` value in `self.units` is
        multiplied out and becomes part of `result.value`, i.e. the
        resulting units have `k` = 1.
        """

        # TODO Special case -1 for convenience / speed?

        pwr_basis = _Units(self.units) ** pwr
        res_value = pwr_basis.k * (self.value ** pwr)
        res_value = coax_type(res_value, type(self.value), default=res_value)
        res_basis = _Units(1, *pwr_basis[1:])  # k factored out.

        if not res_basis.is_dimless():
            return Dim(res_value, str(res_basis))
        else:
            return res_value  # Units disappeared.

    def __radd__(self, lhs: DimArray) -> DimArray:
        """See ``__add__`` for addition rules."""
        return Dim(lhs, '') + self

    def __rsub__(self, lhs: DimArray) -> DimArray:
        """See ``__sub__`` for subtraction rules."""
        return Dim(lhs, '') - self

    def __rmul__(self, lhs: DimArray) -> DimArray:
        """See ``__mul__`` for multiplication rules."""
        return Dim(lhs, '') * self

    def __rtruediv__(self, lhs: DimArray) -> DimArray:
        """See ``__truediv__`` for division rules."""
        return Dim(lhs, '') / self
    
    # -- Comparison Operators -------------------------------------------------

    def __lt__(self, rhs):
        return _common_cmp(self, rhs, operator.lt)

    def __le__(self, rhs):
        return _common_cmp(self, rhs, operator.le)

    def __eq__(self, rhs):
        return _common_cmp(self, rhs, operator.eq)

    def __ne__(self, rhs):
        return _common_cmp(self, rhs, operator.ne)

    def __ge__(self, rhs):
        return _common_cmp(self, rhs, operator.ge)

    def __gt__(self, rhs):
        return _common_cmp(self, rhs, operator.gt)

    # -- String Magic Methods -----------------------------------------------

    def __format__(self, format_spec: str):
        return format(self.value, format_spec) + f" {self.units}"

    def __repr__(self):
        return f"Dim({self.value}, '{self.units}')"

    def __str__(self):
        return self.__format__('')

    # -- Normal Methods -----------------------------------------------------

    def convert(self, to_units: str) -> DimArray:
        """
        Generate new ``Dim`` object converted to requested units.
        """
        return Dim(convert(self.value, from_units=self.units,
                           to_units=to_units), to_units)

    def is_absolute_temp(self) -> bool:
        """
        Returns ``True`` if this is a basic temperature and is a temperature
        on an absolute scale (i.e. K, °R, not on an offset scale such as °C,
        °F).  Otherwise returns ``False``.
        """
        return _Units(self.units).is_absolute_temp()

    def is_base_temp(self) -> bool:
        """
        Returns ``True`` if this is a 'standalone' temperature, i.e. the only
        field in the units signature is θ¹ (temperature can be of any type) and
        multiplier `k` = 1.  Otherwise returns ``False``.
        """
        return _Units(self.units).is_base_temp()

    def is_dimless(self) -> bool:
        """Return True if the value has no effective dimensions."""
        return _Units(self.units).is_dimless()

    # TODO Remove?
    # def is_equiv(self, rhs) -> bool:
    #     """
    #     Returns ``True`` if ``self`` and ``rhs`` have equivalent unit bases
    #     (e.g. both are pressures, currents, speeds, etc), otherwise
    #     returns ``False``.
    #
    #     - If ``rhs`` is not a ``Dim`` object then we return ``True`` only if
    #       ``self.is_dimless() == True``, otherwise we return ``False``.
    #     - If ``rhs`` has unknown units then we return ``False``.
    #     - The actual compatibility test used is:
    #         ``result = Dim(1, self.units) / Dim(1, rhs.units)``
    #       If the result of this division has no units then we return ``True``.
    #       This allows for cancellation of units (and radians, etc).
    #     """
    #     if not isinstance(rhs, Dim):
    #         if self.is_dimless():
    #             return True
    #         else:
    #             return False
    #
    #     try:
    #         if not isinstance(Dim(1, self.units) / Dim(1, rhs.units), Dim):
    #             return True
    #     except ValueError:
    #         return False

    def is_temp_change(self) -> bool:
        """
        Returns ``True`` if this is a basic temperature and can represent
        a temperature change, e.g. K, °R, Δ°C, Δ°F.  Otherwise returns False.

        .. note:: This is *not* the opposite of `is_total_temp()`.  For
           example ``dim(300, 'K').is_temp_change() == True`` as well as
           ``dim(300, 'K').is_total_temp() == True``.
        """
        return _Units(self.units).is_temp_change()

    def is_total_temp(self) -> bool:
        """
        Returns ``True`` if this is a basic temperature as well as a
        total temperature (on either absolute or offset scales), i.e. K, °R,
        °C, °F.  Otherwise returns ``False``.
        """
        return _Units(self.units).is_total_temp()

    def to_real(self, to_units: str = None) -> RealArray:
        """
        Remove dimensions and return a plain real number type.  The target
        units and number type can be optionally specified.  This is a
        convenience method equivalent to ``self.convert(to_units).value``

        .. note:: Unit information is lost. See operators ``__int__``,
           ``__float__``, etc.

        Parameters
        ----------
        to_units : str (optional)
            Convert to these units prior to returning numeric value
            (default = `self.units`).

        Returns
        -------
        result : Number
            Plain numeric value of type `to_type` using units `to_units`.
        """

        if to_units is not None:
            return self.convert(to_units).value
        else:
            return self.value

    def to_real_sys(self, unit_system: str = None) -> RealArray:
        """
        Similar to ``to_real()`` except that instead of giving target units
        for conversion, a complete target system of units is given instead.
        This is used to put many values on a standard basis for numeric
        algorithms that don't accept ``Dim`` objects.

        Parameters
        ----------
        unit_system : str
            A unit string (as per ``dim()``) representing a consistent set of
            units.  The units must all be of unit power and there must be no
            multiplying factor. If ``unit_system = None`` the module variable
            ``STD_UNIT_SYSTEM`` is used.

            Unit types for mass (M), length (L), time (T) and temperature (θ)
            must always be provided.  Other types may be omitted for brevity
            and will have the following default values assigned:

                - Amount of substance (N): mol
                - Electric current (I): A
                - Luminous intensity (J): cd
                - Plain angle (A): rad
                - Solid angle (Ω): sr

        Returns
        -------
        result :
            A plain number or object of the same type as held in ``Dim.value``.

        """
        unit_system = unit_system or STD_UNIT_SYSTEM
        partial_sys = _Units(unit_system)

        factor = _basis_factor(_Units(self.units), partial_sys)
        return self.value * factor


RealScalar = Union[int, float, complex]
"""A `RealScalar` is a shorthand defined for type checking porpoises as 
``Union[int, float, complex]`` and represents a general numeric scalar.  Such 
a scalar does not have units but could be potentially assigned them. """

RealArray = Union[RealScalar, np.ndarray]
"""A `RealArray` is a shorthand defined for type checking porpoises as 
``Union[RealScalar, np.ndarray]`` and represents either an array or an 
array-like real value that can be coaxed into an array. """

DimScalar = Union[Dim, RealScalar]
"""A `DimScalar` is a shorthand defined for type checking porpoises as 
``Union[Dim, RealScalar]`` and represents a value that is expected to be 
either a general numeric scalar or dimensioned equivalent. """

DimArray = Union[Dim, RealArray]
"""A `DimArray` is a shorthand defined for type checking porpoises as 
``Union[Dim, RealArray]`` and is used where an argument is expected to be 
either an array or an array-like real value that can be coaxed into an array, 
or dimensioned equivalent. """


# -- Public Functions -------------------------------------------------------

def add_base_unit(units: [str], base_type: str):
    """
    Simplified version of ``add_unit()`` for base unit cases where `k` = 1,
    and power = 1 (linear).  Labels are taken in turn and assigned to the
    common base dimension given e.g. ``add_base_unit(['kg', 'g', 'mg'], 'M')``
    creates the mass base units of kg, g, mg.

    Parameters
    ----------
    units : [str]
        New base units to make.
    base_type : str
        Base dimension for all new units, e.g. mass `M` or length `L`.
    """
    for unit in units:
        uargs = {base_type: (unit, 1)}
        _add_unit(unit, _Units(**uargs))


def add_unit(unit: str, basis: str):
    """
    Register a new unit basis associated with the given label. The unit
    basis will be sanity checked as part of this process.

    Parameters
    ----------
    unit : str
        Case-sensitive string to use as a label for the resulting units.
    basis : str
        A string to be parsed for conversion to a unit basis.  See
        ``dim()`` for complete details on valid unit basis strings.

    Raises
    ------
    ValueError
        Incorrect arguments / types.
    """
    _add_unit(unit, _Units(basis))


def convert(value: RealArray, from_units: str, to_units: str) -> RealArray:
    """
    Convert ``value`` currently in ``from_units`` to requested ``to_units``.
    This is used for doing conversions without using ``Dim`` objects.

    Examples
    --------
    >>> length = 1.0  # Inches.
    >>> mm_length = convert(length, from_units='in', to_units='mm')
    >>> print(f"Length = {mm_length} mm.")
    Length = 25.4 mm.

    Parameters
    ----------
    value : scalar or array-like
        Value (not ``Dim`` object) for conversion.
    from_units : str
        Units of ``value``.
    to_units : str
        Target units.

    Returns
    -------
    result : scalar or array-like
        Converted value using new units.
    """
    if to_units == from_units:
        return value  # Shortcut for identical units.

    return _convert(value, _Units(from_units), _Units(to_units))


# noinspection PyIncorrectDocstring
def dim(value: Union[DimArray, str] = 1, units: str = None) -> Dim:
    """
    This factory function is the standard means for constructing a
    dimensioned quantity.  The following argument combinations are possible:

        - dim(value, units): Normal construction.  If units is ``None`` it
          is converted to an empty string.
        - dim(): Assumes value = 1 (integer) and dimensionless.
        - dim(units): If a string is passed as the first argument,
          this is transferred to the ``units`` argument and value = 1 is
          assumed.
        - dim(value): Assumes dimensionless, i.e. this method promotes a
          plain value to Units().
        - dim(Dim): Returns Dim object argument directly (no effect).
        - dim(Dim, units): Returns a Dim object after attempting to convert
          the `value` ``Dim`` object to the given `units`.

    The `units` string has the following format:

        - Base unit strings can include characters from the alphabet,
          ``_ Δ ° μ``.
        - Individual base units can be separated by common operators such as
          ``. * × /``.  The division symbol is not used as it looks too
          similar to addition on screens.
        - A numerical leading constant may be optionally included, along with
          an operator if this is required e.g. `1000m³`, `12.0E-02.ft²`.
        - Unit power can be indicated using `^` or unicode
          superscript characters, e.g. `m³` or `m^3`.
        - Dividing units can be indicated by using the division operator
          `/` or directly showing negative powers, e.g. `kg.m⁻³` or `kg/m^3`.

    Parameters
    ----------
    value : Number
        Non-dimensional value.

    units : str (Optional)
        String representing the combination of base units associated
        with this value.
    """
    # TODO: Performance improvement - Create a dict for 'user created'
    #  non-simple units (similar to _KNOWN_UNITS) where we can store these
    #  for faster lookup the next time the same type is requested.

    if units is None:
        # Called with no arguments or one argument.

        if isinstance(value, Dim):
            # Case dim(Dim): dim() called on Dim.  Return Dim object.
            return value

        elif isinstance(value, str):
            # Case dim(units): Transfer first argument to units, assume
            # value = 1.  If units are not already known, do a sanity check.
            value, units = 1, value
            if units not in _KNOWN_UNITS:
                _check_units(_Units(units))

            return Dim(value, units)

        else:
            # Case dim() or dim(value): No units.
            return Dim(value, '')  # Convert None to empty string.

    else:
        # Called with two arguments.

        if isinstance(value, str):
            # Warn if both value and units were strings.  This is normally
            # unintnetional.
            warnings.warn("Warning: dim() received string where a numeric "
                          "value was expected.")

        if isinstance(value, Dim):
            # Case dim(Dim, units): Attempt to convert Dim to new units
            # and return.
            return value.convert(units)

        else:
            # Case dim(value, units): Typical usage to make Dim object.  If
            # units are not already known, do a sanity check.
            if units not in _KNOWN_UNITS:
                _check_units(_Units(units))
            return Dim(value, units)


def is_dimarray(x: DimArray) -> bool:
    """
    Returns ``True`` if `x` is one of the ``DimArray`` types.
    """
    # if isinstance(x, DimArray):  # TODO Python 3.10 allows this.
    if isinstance(x, (Dim, np.ndarray, int, float, complex)):
        return True
    else:
        return False


def set_conversion(from_label: str, to_label: str, *, fwd: Number,
                   rev: Union[Number, str, None] = 'auto'):
    """
    Set a conversion between units in the directed graph of conversions.

    Parameters
    ----------
    from_label,to_label : str
        Case-sensitive identifier.
    fwd : Number
        Value of the multiplier to convert from -> to.
    rev : Number or str, optional
        Value of reverse conversion to -> from:

        - If rev == None, no reverse conversion is added.
        - If rev == 'auto' then a reverse conversion is deduced from the
          forward conversion as follows:

            - If fwd == Fraction → rev = 1/Fraction.
            - If fwd == int > 1 → rev = Fraction(1, fwd).
            - All others → rev = 1/fwd and coax to int if possible.

    Raises
    ------
    ValueError
        if either unit does not exist or not a base unit.
    """
    from_unit = _KNOWN_UNITS[from_label]
    to_unit = _KNOWN_UNITS[to_label]

    # Record the forward conversion.
    if wdg_edge(from_unit, to_unit) in _CONVERSIONS:
        raise ValueError(f"Conversion '{from_unit}' -> '{to_unit}' already "
                         f"defined.")

    _CONVERSIONS[from_unit:to_unit] = fwd  # Keys are namedtuples.

    # Handle the reverse conversion as requested.
    if rev is None:
        # Case - No reverse conversion: Record nothing.
        pass

    elif rev == 'auto':
        # Case - Automatic reverse conversion:  Deduce a sensible value.
        if isinstance(fwd, Fraction):
            rev = 1 / fwd
            if rev.denominator == 1:  # Check if int.
                rev = rev.numerator
        elif isinstance(fwd, int) and fwd > 1:
            rev = Fraction(1, fwd)
        else:
            # noinspection PyTypeChecker
            rev = 1 / fwd
            rev = coax_type(rev, int, default=rev)

        _CONVERSIONS[to_unit:from_unit] = rev

    else:
        # Case - Explicit conversion given: Record directly.
        _CONVERSIONS[to_label:from_label] = rev


def to_absolute_temp(x: Dim) -> Dim:
    """
    Function to convert an isolated total temperature to an absolute scale.

        - If `x` is on an offset scale the equivalent on an absolute scale is
          used (°C → K or °F → °R).
        - If `x` is already on an absolute scale it is returned directly.

    Parameters
    ----------
    x : Dim
        Value with base temperature dimensions only.

    Returns
    -------
    result : Dim
        `x` converted as required.

    Raises
    ------
    ValueError
        If `x` is not a total temperature, i.e. is mixed units or is a
        temperature change / Δ.
    """
    if x.is_absolute_temp():
        return x

    if not x.is_total_temp():
        raise ValueError(f"'{x.units}' is not a total temperature.")

    # Determine offset -> absolute mapping to use.
    try:
        abs_units = _OFF_ABS_MAP[x.units]
    except KeyError:
        raise ValueError(f"No absolute temperature mapping for '{x.units}'.")

    return x.convert(abs_units)


# == Private Attributes & Functions ===========================================

_BLOCKED_CONVS: set[_Units] = set()  # Will be rejected by _convert().
_COMP_CONV_CACHE: {(_Units, _Units): Number} = {}  # Cached computed convs.
_CONVERSIONS = WtDirGraph()  # Conversions between Units().
_KNOWN_BASE_UNITS: set[_Units] = set()  # Single entry ^1 Units().
_KNOWN_UNITS = MultiBiDict()  # All Units(), base and derived.

# Precompiled parser.

_UCODE_SS_CHARS = ('⁺⁻ᐧ⁰¹²³⁴⁵⁶⁷⁸⁹', '+-.0123456789')
_UC_SGN = _UCODE_SS_CHARS[0][0:2]
_UC_DOT = _UCODE_SS_CHARS[0][2]
_UC_DIG = _UCODE_SS_CHARS[0][3:]
_UC_PWR_PATTERN = fr'''[{_UC_SGN}]?[{_UC_DIG}]+(?:[{_UC_DOT}][{_UC_DIG}]+)?'''

# noinspection RegExpUnnecessaryNonCapturingGroup
_unitparse_rx = re.compile(fr'''
    ([.*×/])?                                               # Operator.
    (?:([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[.*×]?))?    # Mult w/ sep
    ([a-zA-Z_Δ°μ]+)                                         # Unit
    ((?:\^[+-]?[\d]+(?:[.][\d]+)?)|                         # Pwr (ascii), or
    (?:{_UC_PWR_PATTERN}))?                                 # Pwr (ucode)
    |(.+)                                                   # OR mismatch.
''', flags=re.DOTALL | re.VERBOSE)

# The following are setup to hold temperature units so that special rules
# can be applied.

_ABS_SCALE_TEMPS = ('K', '°R')
_OFF_SCALE_TEMPS = ('°C', '°F')
_OFF_ABS_MAP = {'°C': 'K',  # From -> To.
                '°F': '°R'}
_TOTAL_TEMPS = _ABS_SCALE_TEMPS + _OFF_SCALE_TEMPS
_DELTA_TEMPS = ('K', '°R', 'Δ°C', 'Δ°F')
_ALL_TEMPS = _TOTAL_TEMPS + _DELTA_TEMPS


# -----------------------------------------------------------------------------


class _Units(namedtuple('_Units',
                        ['k', 'M', 'L', 'T', 'θ', 'N', 'I', 'J', 'A', 'Ω'],
                        defaults=[1, *(('', 0),) * 9])):
    """
    The ``_Units`` type is a namedtuple typerepresenting a combination of
    individual basis units and a leading factor representing a specific
    base or derived unit.   Each basis unit is a tuple of a string label
    and power. Units are hashable so they can be looked up for quick
    conversions.

    Field names in order:
        - k:  Multiplier.
        - M:  Mass.
        - L:  Length.
        - T:  Time.
        - θ:  Temperature.
        - N:  Amount of substance.
        - I:  Electric current.
        - J:  Luminous intensity.
        - A:  Plane angle (see Units() definition for permanence).
        - Ω:  Solid angle (see Units() definition for permanence).

   .. note:: Direct manipulation of ``_Units`` objects is not normally
      required.

   .. Note::  Plane angle radians and solid angle steradians are effectively
      dimensionless:

        - These 'units' are retained to allow consistency checks on
          mathematics, especially if angular degrees are used, etc.
        - Radians / steradians are retained during creation of new units,
          which allows for correct conversion of things like rotational
          speeds, etc.
        - Radians / steradians disappear after multiplication with a
          different set of units (i.e. a calculation) if they finish with
          power == 1.
   """

    def __new__(cls, *args, **kwargs) -> _Units:
        """
        ``_Units`` objects can either be created by passing a single string
        argument to be parsed, or otherwise by directly initialising as
        a namedtuple.

        After parsing a unit basis string, the units are sanity checked
        prior to return.
        """
        # Case: All except a single string.
        # -> Positional and keyword arguments passed direct to namedtuple.
        # Note: This includes blank i.e. ``_Units()``.
        if not (len(args) == 1 and isinstance(args[0], str)):
            res = super().__new__(cls, *args, **kwargs)
            _check_units(res)
            return res

        # Case:  Single empty string.
        # -> Return blank Units().
        if not args[0]:
            return super().__new__(cls)

        # Case: Single non-empty string.
        # -> Check if already known.  If so, return a copy.
        try:
            return _Units(*_KNOWN_UNITS[args[0]])  # Known units pre-checked.
        except KeyError:
            pass

        # -> Not already known.  Parse string into subunits.
        tokens = re.findall(_unitparse_rx, ''.join(args[0].split()))

        if tokens[0][0]:
            raise ValueError("Invalid leading operator in unit definition.")

        # Build result by multiplying sub-parts.
        res_units = _Units()
        for opstr, multstr, sub_label, pwrstr, mismatch in tokens:
            if mismatch:
                raise ValueError(f"Invalid term '{mismatch}' in unit "
                                 f"definition.")

            # If multiplier is present, try to get an int, otherwise float.
            if multstr:
                mult = coax_type(float(multstr), int, float)
            else:
                mult = 1

            try:
                sub_basis = _KNOWN_UNITS[sub_label]
            except KeyError:
                # If the sub-units doesn't exist, we *really* don't know this
                # unit.
                raise ValueError(f"Unknown component '{sub_label}' in "
                                 f"unit definition.")

            pwrstr = pwrstr.strip('^')  # Reqd for ASCII powers.
            pwrstr = _from_ucode_super(pwrstr)
            if pwrstr:
                pwr = force_type(pwrstr, int, float)
            else:
                pwr = 1
            if opstr == '/':
                mult = kind_div(1, mult)
                pwr = -pwr

            res_units *= sub_basis ** pwr  # Apply power.
            res_units *= _Units(k=mult)  # Apply multiplier.

        _check_units(res_units)
        return res_units

    # -- Binary Operators ---------------------------------------------------

    def __mul__(self, rhs: _Units) -> _Units:
        """
        Multiply two units combining component bases and 'k' value. This is
        the central function used by most other operators / conversions.
        See Dim.__mul__ for expected results.

        .. note:: Angular units of radians or steradians are retained by
           this operator.
        """
        res_k = rhs.k * self.k
        res_bases = []
        for (l_ustr, l_pwr), (r_ustr, r_pwr) in zip(self[1:], rhs[1:]):
            res_pwr = l_pwr + r_pwr
            res_ustr = l_ustr if l_ustr else r_ustr

            # Multiply by a factor if required.
            if l_ustr and r_ustr and l_ustr != r_ustr:
                factor = _conversion_factor(_Units(r_ustr), _Units(l_ustr))
                if factor is None:
                    raise TypeError(f"No conversion available for {r_ustr} ->"
                                    f" {l_ustr}.")
                # noinspection PyUnresolvedReferences
                res_k *= factor ** r_pwr

            # Store or cleanup if cancelled out.
            if res_pwr != 0:
                res_bases += [(res_ustr, res_pwr)]
            else:
                res_bases += [('', 0)]

        res_k = coax_type(res_k, (int, float), default=res_k)
        return _Units(res_k, *res_bases)

    def __pow__(self, pwr: RealScalar) -> _Units:
        """
        Raises unit basis to a given power, including leading factor 'k'.
        """
        # Build resulting units. Most units have integer powers; try to
        # preserve int-ness.
        res_basis = [coax_type(self.k ** pwr, type(self.k),
                               default=self.k ** pwr)]  # Set 'k'.
        for u, p in self[1:]:
            res_basis += [(u, coax_type(p * pwr, int, default=p * pwr))]

        return _Units(*res_basis)

    # -- String Magic Methods -----------------------------------------------

    def __str__(self):
        """
        If a unique label exists return it, otherwise builds a generic
        string.
        """
        try:
            labels = _KNOWN_UNITS.inverse[self]
            if len(labels) == 1:
                return labels[0]
        except KeyError:
            pass

        # Generate generic string.
        base_parts = []
        for base in self[1:]:
            if base[0]:
                u_substr = f'{base[0]}'
                if base[1] != 1:
                    if OUTPUT_UCODE_PWR:
                        u_substr += _to_ucode_super(f'{base[1]}')
                    else:
                        u_substr += f'^{base[1]}'
                base_parts += [u_substr]
        label = '.'.join(base_parts)

        if not label:  # Cover dimless case.
            label = ''

        if self.k != 1:
            label = f'{self.k}*' + label

        return label

    # -- Public Methods -----------------------------------------------------

    def is_absolute_temp(self) -> bool:
        """
        Refer to Dim() for documentation.
        """
        if self.is_base_temp() and self.θ[0] in _ABS_SCALE_TEMPS:
            return True
        else:
            return False

    def is_base_temp(self) -> bool:
        """
        Refer to Dim() for documentation.
        """
        if self.θ[0] in _ALL_TEMPS and self.θ[1] == 1 and self.k == 1:
            # θ¹ present, k == 1.  Check all other powers are zero.
            if all(p == 0 for i, (_, p) in enumerate(self[1:], start=1)
                   if self._fields[i] != 'θ'):
                return True

        return False

    def is_dimless(self) -> bool:
        """
        Returns ``False`` if any indicies in the signature are nonzero or
        k != 1, otherwise returns ``True`` (dimensionless units).
        """
        if any(p != 0 for _, p in self[1:]) or self[0] != 1:
            return False
        else:
            return True

    def is_temp_change(self) -> bool:
        """
        Refer to Dim() for documentation.
        """
        if self.is_base_temp() and self.θ[0] in _DELTA_TEMPS:
            return True
        else:
            return False

    def is_total_temp(self) -> bool:
        """
        Refer to Dim() for documentation.
        """
        if self.is_base_temp() and self.θ[0] in _TOTAL_TEMPS:
            return True
        else:
            return False


# ---------------------------------------------------------------------------


def _add_unit(unit: str, basis: _Units):
    """
    Private function for registering a new ``_Units`` entry.
    """

    # Check string.
    allowed_chars = {'_', '°', 'Δ'}
    if any(not c.isalpha() and c not in allowed_chars for c in unit):
        raise ValueError(f"Invalid unit string: '{unit}'.")

    if unit in _KNOWN_UNITS:
        raise ValueError(f"Unit '{unit}' already defined.")

    # Check unit basis.  If everything is OK, store as a known unit.
    _check_units(basis)
    _KNOWN_UNITS[unit] = basis

    # Is this a derived unit? If so we are already finished.
    base_unit_str = _base_unit(basis)
    if not base_unit_str:
        return

    # Base units are also added to a separate known group.  Labels must match.
    if base_unit_str != unit:
        raise ValueError(f"Basis unit string must match supplied basis, got: "
                         f"'{unit}' != '{base_unit_str}'")

    _KNOWN_BASE_UNITS.add(basis)


def _base_unit(unit: _Units) -> Optional[str]:
    """
    Check if the given basis represents a single base unit.

    Parameters
    ----------
    unit : _Units
        Unit basis to check.

    Returns
    -------
    result : str or None
        If `basis` is a base unit - i.e. `k` = 1 and only one linear base
        unit is used - that string label is returned, otherwise returns None.
    """
    bases_used, last_u, last_p = 0, None, 0
    for u, p in unit[1:]:
        if p != 0:
            bases_used += 1
            last_u, last_p = u, p
    if bases_used != 1 or last_p != 1 or unit.k != 1:
        return None
    else:
        return last_u


def _basis_factor(from_units: _Units, to_basis: _Units) -> RealScalar:
    """
    Similar to _conversion_factor, however this is for changing `from_units`
    to a completely different and consistent basis, so all fields are
    included.  If `to_basis` is only partial, default values are inserted.
    The required multiplier is returned along with a representative string of
    the final applicable units.
    """

    # Check provided system.
    if to_basis[0] != 1:
        raise ValueError(f"Target unit system can't include multiplier, "
                         f"got k = {to_basis[0]}")

    # noinspection PyProtectedMember
    for str_i, pwr_i in to_basis[1:]:
        if str_i and pwr_i != 1:
            raise ValueError(f"Unit system powers must be 1.  Got "
                             f"{str_i}^{pwr_i}.")

    # Assign defaults to any missing units.
    final_sys = []
    # noinspection PyProtectedMember
    for (part_u, _), (def_u, _), field_u in zip(to_basis[1:],
                                                _DEFAULT_UNIT_SYS[1:],
                                                to_basis._fields[1:]):
        if part_u:
            final_sys.append((part_u, 0))
        else:
            # Not provided: Try to use a default.
            if def_u:
                final_sys.append((def_u, 0))
            else:
                raise ValueError(f"Unit system missing type for field "
                                 f"'{field_u}'.")

    final_sys = _Units(1, *final_sys)

    # Final system now has k = 1, entries for all units and all powers = 0.
    # A LHS multiplication gives the required factor to change bases.
    return (final_sys * from_units).k


def _block_conversion(from_label: str):
    """
    Enters ``None`` as a target unit against `from_label`.  This used when
    automatic conversion is to be prohibited (e.g. offset temperatures °C,
    °F, etc).  It is a backup measure to ensure that such a conversion isn't
    accidentally added by a user.
    """
    from_unit = _KNOWN_UNITS[from_label]
    _CONVERSIONS[from_unit:None] = None


def _conversion_factor(from_basis: _Units,
                       to_basis: _Units) -> Optional[Number]:
    """
    Compute the factor for converting between two units.  This is not used
    for total temperatures which are handled separately.
    """
    # Conversion to lhs gives unity.
    if from_basis == to_basis:
        return 1

    # Step 1: See if this conversion has been cached.
    if CACHE_COMPUTED_CONVS:
        try:
            return _COMP_CONV_CACHE[from_basis, to_basis]
        except KeyError:
            pass

    # Step 2: See if a shortcut is available (base units will take
    # this path only).
    if (from_basis in _CONVERSIONS) and (to_basis in _CONVERSIONS):
        path, factor = _CONVERSIONS.trace(from_basis, to_basis, operator.mul)
        if path and len(path) > CONV_PATH_LENGTH_WARNING:
            warnings.warn(
                f"Warning: Converting '{from_basis}' -> '{to_basis}' gives "
                f"overlong path: " + ' -> '.join(str(x) for x in path))

        if factor is not None:
            if CACHE_COMPUTED_CONVS:
                _COMP_CONV_CACHE[from_basis, to_basis] = factor
            return factor

    # Don't compute conversions if we are at base units (otherwise infinite
    # loop).
    if (from_basis in _KNOWN_BASE_UNITS) or (to_basis in _KNOWN_BASE_UNITS):
        return None

    # Step 3: Try to compute the factor from the base units.  This is done
    # by forming new 'power-zero' target basis and multiplying on the LHS,
    # i.e.: (1/k_to)*to_units^0 * original.
    new_basis = []
    for (from_ustr, from_pwr), (to_ustr, to_pwr) in zip(from_basis[1:],
                                                        to_basis[1:]):
        # Check each part must both be either something or nothing.
        if bool(from_ustr) != bool(to_ustr):  # XOR check.
            raise ValueError(f"Inconsistent base units converting "
                             f"'{from_basis}' -> '{to_basis}'")

        # Check each part has the same indicies.
        if from_pwr != to_pwr:
            raise ValueError(f"Inconsistent indices converting '{from_basis}' "
                             f"-> '{to_basis}': '{from_ustr}^{from_pwr}' and "
                             f"'{to_ustr}^{to_pwr}'")
        new_basis += [(to_ustr, 0)]

    new_basis = _Units(1 / to_basis.k, *new_basis)

    # Do multiplication, factor contains the result.
    factor = (new_basis * from_basis).k
    if CACHE_COMPUTED_CONVS:
        _COMP_CONV_CACHE[from_basis, to_basis] = factor

    return factor


def _convert(value: RealArray, from_units: _Units,
             to_units: _Units) -> RealArray:
    """
    Convert ``value`` given ``_Units`` bases.
    """
    if from_units.is_total_temp() and to_units.is_total_temp():
        # Conversion between total temperatures are a special case due to
        # possible offset of scales.
        from_θ, to_θ = from_units.θ[0], to_units.θ[0]
        return _convert_total_temp(value, from_θ, to_θ)

    if from_units in _BLOCKED_CONVS:
        raise ValueError(f"Automatic conversion from unit '{str(from_units)}' "
                         f"is prevented.")
    if to_units in _BLOCKED_CONVS:
        raise ValueError(f"Automatic conversion to unit '{str(to_units)}' is "
                         f"prevented.")

    # All other unit types.
    factor = _conversion_factor(from_units, to_units)
    if factor is None:
        raise ValueError(f"No conversion found: '{from_units}' -> '{to_units}'")

    conv_value = value * factor
    conv_value = coax_type(conv_value, type(value), default=conv_value)
    return conv_value


def _check_units(unit: _Units):
    """
    Do certain checks on validity of unit tuple during construction. Blank
    units cannot have non-zero powers.  Note that the reverse case is
    permitted, and is useful for unit conversion.
    """
    if any((not u and p != 0) for u, p in unit[1:]):
        raise ValueError("Blank units must have no power.")

    if unit.k == 0:
        raise ValueError(f"k must be non-zero.")

    # Offset temperatures must be standalone.  Note that some units (K,°R)
    # can represent absolute temperature as well as changes.
    if unit.θ[0] in _OFF_SCALE_TEMPS and not unit.is_base_temp():
        raise ValueError(f"Offset temperatures are only permitted as "
                         f"base units, got: {str(unit)}")


def _convert_total_temp(x: RealArray, from_unit: str,
                        to_unit: str) -> RealArray:
    """
    Conversions of total temperatures are a special case due to the offset of
    the °C and °F base units scales.

    .. note: Arithmetic is simplified by converting `x` to Kelvin then
       converting to final units.

    Parameters
    ----------
    x : scalar or array-like
        Temperature to be converted.
    from_unit, to_unit : str
        String matching °C, °F, K, °R.

    Returns
    -------
    result : scalar or array-like
        Converted temperature value.
    """
    # Convert 'from' -> Kelvin.
    if from_unit == '°C':
        x = x + 273.15  # This style (instead of +=) allows for NumPy ufunc.
    elif from_unit == 'K':
        pass
    elif from_unit == '°F':
        x = (x + 459.67) * 5 / 9
    elif from_unit == '°R':
        x = x * 5 / 9
    else:
        raise ValueError(f"Cannot convert total temperature: {from_unit} -> "
                         f"{to_unit}")

    # Convert Kelvin -> 'to'.
    if to_unit == '°C':
        x = x - 273.15
    elif to_unit == 'K':
        pass
    elif to_unit == '°F':
        x = (x * 9 / 5) - 459.67
    elif to_unit == '°R':
        x = x * 9 / 5
    else:
        raise ValueError(f"Cannot convert total temperature: {from_unit} -> "
                         f"{to_unit}")

    return x


def _common_cmp(lhs: Dim, rhs, op: Callable[..., bool]):
    """
    Common method used for comparison of a Dim object and another object
    called by all magic methods.  If ``rhs`` is not a ``Dim`` object is is
    promoted before comparison.
    """
    if not isinstance(rhs, Dim):
        rhs = Dim(rhs, '')

    if lhs.units == rhs.units:
        return op(lhs.value, rhs.value)
    else:
        return op(lhs.value, rhs.convert(lhs.units).value)


# Default entries when doing conversion between consistent bases.  Types M, L,
# T, θ are not given here, they must always be provided.
_DEFAULT_UNIT_SYS = _Units(N=('mol', 0), I=('A', 0), J=('cd', 0), A=('rad', 0),
                           Ω=('sr', 0))


def _dim_mul_generic(lhs: Dim, rhs: DimArray, mult_op: Callable) -> DimArray:
    """
    Multiply two dimensioned values using multuiplication operator ``op``.
    This is the generic version used by Dim.__mul__, Dim.__matmul__,
    etc allowing for scalar or matrix multiplication.  See Dim.__mul__ for
    rules.
    """
    if not isinstance(rhs, Dim):
        if lhs.units:
            # Return using existing units.
            return Dim(mult_op(lhs.value, rhs), lhs.units)
        else:
            # Drop dimensions.
            return mult_op(lhs.value, rhs)

    # Total temperatures are allowed as multiplication arguments however
    # they are converted to absolute scales before used.
    if lhs.is_total_temp():
        lhs = to_absolute_temp(lhs)
    if rhs.is_total_temp():
        rhs = to_absolute_temp(rhs)

    # Build final basis and factor out 'k' into result.
    res_basis = _Units(lhs.units) * _Units(rhs.units)
    res_value = mult_op(lhs.value, rhs.value)
    res_value *= res_basis.k  # Scalar multiply for factor.

    # noinspection PyProtectedMember
    res_basis = res_basis._replace(k=1)

    # Check if multiplication resulted in radians, which can be of any
    # power (as they are dimensionless).
    if res_basis.A[0] == 'rad':
        # noinspection PyProtectedMember
        res_basis = res_basis._replace(A=('', 0))

    if res_basis.Ω[0] == 'sr':
        # noinspection PyProtectedMember
        res_basis = res_basis._replace(Ω=('', 0))

    if not res_basis.is_dimless():
        return Dim(res_value, str(res_basis))
    else:
        return res_value  # Units fell off.


# -- Unicode Functions --------------------------------------------------------

def _from_ucode_super(ss: str) -> str:
    """
    Convert any unicode numeric superscipt characters in the string ``ss``
    to normal ascii text.
    """
    result = ''
    for c in ss:
        idx = _UCODE_SS_CHARS[0].find(c)
        if idx >= 0:
            result += _UCODE_SS_CHARS[1][idx]
        else:
            result += c
    return result


def _to_ucode_super(ss: str) -> str:
    """
    Convert numeric characters in the string ``ss`` to unicode superscript.
    """
    result = ''
    for c in ss:
        idx = _UCODE_SS_CHARS[1].find(c)
        if idx >= 0:
            result += _UCODE_SS_CHARS[0][idx]
        else:
            result += c
    return result


# == Base Unit Definitions ==================================================

# -- Mass -------------------------------------------------------------------

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

# -- Length -----------------------------------------------------------------

#  Signatures.
add_base_unit(['km', 'm', 'cm', 'mm'], 'L')
add_base_unit(['NM', 'sm', 'mi', 'rod', 'yd', 'ft_US', 'ft', 'in'], 'L')
# Notes:
# -'sm' is the US survey / statute mile which is a tiny bit longer than the
# international mile 'mi', because it is defined in US survey feet (below).
# - 'ft_US' is the US survey foot which is a tiny bit larger than the
# international foot 'ft'.

# Conversions.
set_conversion('NM', 'm', fwd=1852)  # International NM.
# Note the older UK NM was slightly different = 1853 m (or 6,080 ft).

set_conversion('sm', 'ft_US', fwd=5280)
# 'sm' is the US statute mile defined as 5,280 US survey feet.
# Interestingly, although this corresponds to exactly 1,760 yards in the
# same manner as the international mile ('mi', below) there is no formal
# definition of a 'US survey yard'.

set_conversion('mi', 'yd', fwd=1760)
# 'mi' is is the international mile defined relative to the international
# yard and foot.

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
set_conversion('in', 'mm', fwd=25.4)  # Defn British / US / Industrial Inch

# -- Time -------------------------------------------------------------------

#  Signatures.
add_base_unit(['day', 'hr', 'min', 's', 'ms'], 'T')

# Conversions.
set_conversion('day', 'hr', fwd=24)
set_conversion('hr', 'min', fwd=60)
set_conversion('min', 's', fwd=60)
set_conversion('s', 'ms', fwd=1000)

# -- Temperature ------------------------------------------------------------

#  Signatures.
add_base_unit(['°C', 'Δ°C', 'K', 'ΔK'], 'θ')
add_base_unit(['°F', 'Δ°F', '°R', 'Δ°R'], 'θ')

_BLOCKED_CONVS.update([_Units('°C'), _Units('°F')])

# Conversions.  Note: Only conversions for temperature changes are given
# here.  Absolute scale temperatures are handled by a special function
# because of offset scales used by °C and °F.
set_conversion('K', 'Δ°C', fwd=1)
set_conversion('K', 'Δ°F', fwd=Fraction(9, 5))
set_conversion('°R', 'Δ°F', fwd=1)

# -- Amount of Substance ----------------------------------------------------

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

# Electric Current ----------------------------------------------------------

# Signatures.
add_base_unit(['A', 'mA'], 'I')

# Conversions.
set_conversion('A', 'mA', fwd=1000)

# -- Luminous intensity -----------------------------------------------------

# Signatures.
add_base_unit(['cd'], 'J')

# Conversions.

# -- Plane Angle ------------------------------------------------------------

# Signatures.
add_base_unit(['deg', 'rad', 'rev', '°'], 'A')
# This is the only use of the ° symbol on its own, but note that ° != deg
# for comparison purposes.

# Conversions.
set_conversion('rev', 'rad', fwd=2 * math.pi)
set_conversion('rev', 'deg', fwd=360)
set_conversion('rad', 'deg', fwd=180 / math.pi)
set_conversion('deg', '°', fwd=1)

# -- Solid angle ------------------------------------------------------------

# Signatures.
add_base_unit(['sp', 'sr'], 'Ω')

# Conversions.
set_conversion('sp', 'sr', fwd=4 * math.pi)  # 1 spat = 4*pr steradians.

# == Derived Unit Definitions ===============================================

# Notes:
# - A a variety of signature types / operators / unicode (*, /, ×, ²,
#   etc) are used in which acts as an automatic check on the parser.
# - Exact standard for G = 9.80665 m/s/s (WGS-84 defn). Full float
#   conversion gives 32.17404855643045 func/s^2.

# -- Area -------------------------------------------------------------------

add_unit('ha', '10000*m^2')

# -- Volume -----------------------------------------------------------------

add_unit('cc', 'cm^3')
add_unit('L', '1000×cm^3')
add_unit('US_gal', '231×in^3')  # Fluid gallon
add_unit('Imp_gal', '4.54609*L')  # British defn
add_unit('US_qt', '0.25*US_gal')  # Fluid quart
add_unit('US_fl_bl', '31.5*US_gal')  # Fluid barrel
add_unit('US_hhd', '2×US_fl_bl')  # Hogshead

# -- Speed ------------------------------------------------------------------

add_unit('fps', 'ft/s')
add_unit('kt', 'NM/hr')
add_unit('mph', 'sm/hr')  # Normally defined as US statute miles / hour.
add_unit('kph', 'km/hr')

add_unit('RPM', 'rev/min')

# -- Acceleration -----------------------------------------------------------

add_unit('G', '9.80665 m/s^2')  # WGS-84 definition

# -- Force ------------------------------------------------------------------

add_unit('kgf', 'kg×G')
add_unit('N', 'kg.m.s⁻²')
add_unit('kN', '1000×N')

add_unit('lbf', 'slug.ft/s²')
add_unit('kip', '1000*lbf')

# -- Pressure ---------------------------------------------------------------

#  Signatures.
add_unit('MPa', 'N/mm²')
add_unit('Pa', 'N/m²')
add_unit('hPa', '100.Pa')
add_unit('kPa', '1000*Pa')
add_unit('GPa', '1e9*Pa')

add_unit('atm', '101325 Pa')  # ISO 2533-1975
add_unit('bar', '100000*Pa')
add_unit('mmHg', '133.322387415*Pa')
# mmHg conversion from BS 350: Part 1: 1974 – Conversion factors and tables

add_unit('Torr', f'{1 / 760}*atm')

add_unit('psf', 'lbf/ft²')
add_unit('inHg', '25.4*mmHg')
add_unit('psi', 'lbf/in²')
add_unit('ksi', '1000*psi')

# -- Energy -----------------------------------------------------------------

add_unit('J', 'N.m')
add_unit('kJ', '1000.J')
add_unit('MJ', '1000.kJ')
add_unit('cal', '4.184×J')  # ISO Thermochemical calorie (cal_th).
add_unit('kcal', '1000.cal')
add_unit('Btu', '778.1723212164716×ft.lbf')  # ISO British Thermal Unit.
# The ISO Btu is defined as exactly 1055.06 J. The above value is the full
# float calculated conversion to ft.lbf.

# -- Power ------------------------------------------------------------------

add_unit('W', 'J/s')
add_unit('kW', '1000*W')
add_unit('hp', '550×ft.lbf/s')

# -- Luminous ---------------------------------------------------------------

add_unit('lm', 'cd.sr')  # Lumen.
