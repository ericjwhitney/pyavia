from __future__ import annotations
import operator
import warnings
from typing import NamedTuple, Callable, TypeVar, Generic

from ._base import (to_absolute_temp,
                    _Units, convert, STD_UNIT_SYSTEM,
                    _basis_factor, _KNOWN_UNITS, _check_units,
                    _MADE_UNITS)

from pyavia.type_ext import coax_type

T = TypeVar('T')


# ===========================================================================

# TODO change to hasable dataclass
class Dim(NamedTuple, Generic[T]):
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

    # value: DimValueType TODO DELETE
    value: T
    units: str

    # TODO needs constructor to check unit string if directly constructed,
    #  and convert to standard unit string.

    # -- Unary Operators ----------------------------------------------------

    def __abs__(self) -> Dim[T]:
        return Dim(abs(self.value), self.units)

    def __complex__(self) -> complex:
        """
        Returns complex(self.value).

        .. note:: Units are removed and checking ability is lost.
        """
        return complex(self.value)

    def __float__(self) -> float:
        """
        Returns float(self.value).

        .. note:: Units are removed and checking ability is lost.
        """
        return float(self.value)

    def __int__(self) -> int:
        """
        Returns int(self.value).

        .. note:: Units are removed and checking ability is lost.
        """
        return int(self.value)

    def __neg__(self) -> Dim[T]:
        return Dim(-self.value, self.units)

    def __round__(self, n: int = None) -> Dim[T]:
        return Dim(round(self.value, n), self.units)

    # -- Binary Operators ---------------------------------------------------

    def __add__(self, rhs: Dim[T]) -> Dim[T]:
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
        # TODO rethink promotion rules, eliminate this?
        #  NOTE - No dims i.e. '' is allowed for Dim objects so might need
        #  to keep?
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

        # TODO Removed - Addition / subtraction should not cancel units.
        # res = Dim(res_value, self.units)
        # if not res.is_dimless():
        #     return res
        # else:
        #     return res.value  # Units fell off.

        return Dim(res_value, self.units)

    def __sub__(self, rhs: Dim[T]) -> Dim[T]:
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
        # TODO rethink promotion rules, eliminate this?
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
                    raise ValueError(
                        f"Total temperatures must be on the same scale for "
                        f"subtraction, got: '{self.units}' - '{rhs.units}'")

                if self.is_absolute_temp():
                    # Simple case of K, °R.
                    return Dim(self.value - rhs.value, self.units)

                # Offset case requires temperatures to be converted to
                # absolute scales prior to subtraction.
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

        # TODO Removed - Addition / subtraction should not cancel units.
        # res = Dim(res_value, self.units)
        # if not res.is_dimless():
        #     return res
        # else:
        #     return res.value  # Units fell off.

        return Dim(res_value, self.units)

    def __mul__(self, rhs: Dim[T] | T) -> Dim[T] | T:
        """
        Multiply two dimensioned values.   Rules are as follows:

            - ``Dim`` * ``Dim``:  Typical case, see general procedure below.
            - ``Dim`` * ``Scalar``:  Returns a ``Dim`` object retaining the
               units string of the LHS argument, with the value simply
               multiplied by the scalar.  This means that *effectively*
               'dimensionless' units (e.g. radians, m/m, etc) can be
               *retained*.  If there are no *actual* units (i.e. units =
               '') then these are dropped. TODO review re: radians
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

    def __truediv__(self, rhs: Dim[T] | T) -> Dim[T] | T:
        """
        Divide a dimensioned value.  The same rules as __mul__ apply.
        """
        if not isinstance(rhs, Dim):
            return Dim(self.value / rhs, self.units)

        return self * (rhs ** -1)

    def __matmul__(self, rhs: Dim[T] | T) -> Dim[T] | T:
        """
        Matrix multiply operator, handled in the same fashion as __mul__.
        Note that this applies to any array wrapped in a `Dim` object,
        not an array *of* `Dim` objects.
        """
        return _dim_mul_generic(self, rhs, operator.matmul)

    def __pow__(self, pwr: int | float) -> Dim[T] | T:
        """
        Raise dimensioned value to a power. Any `k` multiplier value in
        `self.units` is multiplied out and becomes part of `result.value`,
        i.e. the resulting units have `k` = 1.
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

    # TODO reconsider if required?  Can also raise NotImplemented if we
    #  expressly don't allow something.
    # TODO should only work if dimless?
    def __radd__(self, lhs: Dim[T]) -> Dim[T]:
        """See ``__add__`` for addition rules."""
        return Dim(lhs, '') + self

    # TODO Reconsider if required?
    # TODO should only work if dimless?
    def __rsub__(self, lhs: Dim[T]) -> Dim[T]:
        """See ``__sub__`` for subtraction rules."""
        return Dim(lhs, '') - self

    def __rmul__(self, lhs: T) -> Dim[T] | T:
        """See ``__mul__`` for multiplication rules."""
        return Dim(lhs, '') * self

    def __rtruediv__(self, lhs: T) -> Dim[T] | T:
        """See ``__truediv__`` for division rules."""
        return Dim(lhs, '') / self

    # -- Comparison Operators -----------------------------------------------

    # TODO add RHS type hints Dim[T] | T
    def __lt__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.lt)

    def __le__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.le)

    def __eq__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.eq)

    def __ne__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.ne)

    def __ge__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.ge)

    def __gt__(self, rhs) -> bool:
        return _common_cmp(self, rhs, operator.gt)

    # -- String Magic Methods -----------------------------------------------

    # TODO str can optionally generate unicode powers.
    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec) + f" {self.units}"

    # TODO make repr always use non-unicode powers.
    def __repr__(self) -> str:
        # Lowercase dim() used so that __repr__ builds use factory function.
        return f"dim({self.value}, '{self.units}')"

    # TODO str can optionally generate unicode powers.
    def __str__(self) -> str:
        return self.__format__('')

    # -- Normal Methods -----------------------------------------------------

    def convert(self, to_units: str) -> Dim:
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

    def is_base_temp(self) -> bool:  # TODO consider 'is_pure_temp'
        """
        Returns ``True`` if this is a 'standalone' temperature, i.e. the
        only field in the units signature is θ¹ (temperature can be of any
        type) and multiplier `k` = 1.  Otherwise returns ``False``.
        """
        return _Units(self.units).is_base_temp()

    def is_dimless(self) -> bool:
        """
        Return True if the value has no effective dimensions.
        """
        return _Units(self.units).is_dimless()

    def is_similar(self, rhs: Dim[T] | T) -> bool:
        """
        Returns ``True`` if ``self`` and ``rhs`` have equivalent unit bases
        (e.g. both are pressures, currents, speeds, etc), otherwise
        returns ``False``.

        - If ``rhs`` is not a ``Dim`` object then we return ``True`` only if
          ``self.is_dimless() == True``, otherwise we return ``False``.
        - If ``rhs`` has unknown units then we return ``False``.
        - The actual compatibility test used is:
            ``result = Dim(1, self.units) / Dim(1, rhs.units)``
          If the result of this division has no units then we return ``True``.
          This allows for cancellation of units (and radians, etc).
        """
        if not isinstance(rhs, Dim):
            if self.is_dimless():
                return True
            else:
                return False

        try:
            if not isinstance(Dim(1, self.units) / Dim(1, rhs.units), Dim):
                return True
        except ValueError:
            return False

    def is_temp_change(self) -> bool:  # TODO consider is_delta_temp()
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

    def to_value(self, to_units: str = None) -> T:
        """
        Remove dimensions and return a plain value type.  The target units
        can be optionally specified.  This is a convenience method equivalent
        to ``self.convert(to_units).value``

        .. note:: Unit information is lost. See operators ``__int__``,
           ``__float__``, etc.

        Parameters
        ----------
        to_units : str (optional)
            Convert to these units prior to returning numeric value
            (default = `self.units`).

        Returns
        -------
        result :
            Plain numeric value, with units `to_units`.
        """
        if to_units is not None:
            return self.convert(to_units).value
        else:
            return self.value

    # TODO consider to_system instead.
    def to_value_sys(self, unit_system: str = None) -> T:
        """
        Similar to ``to_value()`` except that instead of giving target units
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
            A plain numeric value with units conforming to ``unit_system``.
        """
        unit_system = unit_system or STD_UNIT_SYSTEM
        partial_sys = _Units(unit_system)

        factor = _basis_factor(_Units(self.units), partial_sys)
        return self.value * factor


# ----------------------------------------------------------------------------

def dim(value: Dim[T] | T | str = 1, units: str = None) -> Dim[T] | T:
    # TODO add copy option for dim(Dim(...))

    """
    This factory function is the preferred way to construct a
    dimensioned quantity (`Dim` object).  The following argument combinations
    are possible:

        - ``dim(value, units)``: Normal construction.  If `units` is ``None``
          it is converted to an empty string.
        - ``dim()``: Assumes value = 1 (integer) and dimensionless.
        - ``dim(units)``: If a string is passed as the first argument,
          this is transferred to the ``units`` argument and value = 1 is
          assumed.
        - ``dim(value)``: Assumes dimensionless, i.e. this method promotes a
          plain value to `Dim`. TODO change this - should leaves as value
        - ``dim(Dim)``: Returns `Dim` object argument directly (no effect).
        # TODO add copy option
        - ``dim(Dim, units)``: Returns a `Dim` object after attempting to
          convert the `value` of the  `Dim` argument to the given `units`.

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
    value :
        Non-dimensional value.

    units : str, optional
        String representing the combination of base units associated
        with this value.

    Returns
    -------
    Dim or numeric
        xxxx
    """
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
            if units not in _KNOWN_UNITS or units not in _MADE_UNITS:
                _check_units(_Units(units))
            return Dim(value, units)


# ---------------------------------------------------------------------------

# TODO where is this used? Is it still needed?
# def is_dimoptype(x: DimOpType) -> bool:
#     """
#     Returns ``True`` if `x` is one of the `DimOpType` types.
#
#     Note that Python 3.10 allows this directly, e.g.
#     ``if isinstance(x, DimOpType):``
#     """
#     if isinstance(x, (Dim, DimValueType)):
#         return True
#     else:
#         return False


# ---------------------------------------------------------------------------

# TODO where is this used? Is it still needed?
def str2dim(s: str) -> Dim:
    """
    Convert an input string `s` consisting of a number followed by (optional)
    units to a `Dim` object with numeric value e.g. ``5.63 kg`` or ``10.0 +
    5.0j A``. The number and units must contain no spaces and be separated by
    at least one space. Conversion of the number part is attempted using
    ``int``, ``float`` and ``complex`` in that order.

    Parameters
    ----------
    s : str
        Input string to convert to `Dim`.

    Returns
    -------
    result : `Dim`
        Converted value with dimensions (note that this may itself by
        dimensionless).

    Raises
    ------
    ValueError :
        If splitting was impossible or conversion of the numeric part was
        impossible.
    """
    s_parts = s.split(maxsplit=1)
    if len(s_parts) == 1:
        num_str, unit_str = s, None
    elif len(s_parts) == 2:
        num_str, unit_str = s_parts
    else:
        raise ValueError(f"Couldn't split '{s}' into (value, units) parts.")

    # Convert to numeric value.
    for conv_op in (int, float, complex):
        try:
            num_val = conv_op(num_str)
        except ValueError:
            continue

        # Successful conversion.
        return dim(num_val, unit_str)

    # Failed to convert numeric part.
    raise ValueError(f"Couldn't convert '{num_str}' to a numeric value.")


# ===========================================================================


def _common_cmp(lhs: Dim[T], rhs, op: Callable[[T, T], bool]) -> bool:
    # TODO add rhs type hint ^^^
    """
    Common method used for comparison of a Dim object and another object
    called by all magic methods.  If ``rhs`` is not a ``Dim`` object is is
    promoted before comparison.
    """
    # TODO Rethink if promotion is still valid?
    if not isinstance(rhs, Dim):
        rhs = Dim(rhs, '')

    if lhs.units == rhs.units:
        # TODO might be too vague just using strings
        # here for comparison
        return op(lhs.value, rhs.value)
    else:
        return op(lhs.value, rhs.convert(lhs.units).value)


# ---------------------------------------------------------------------------

def _dim_mul_generic(lhs: Dim[T], rhs: Dim[T] | T,
                     mult_op: Callable[[T, T], T]) -> Dim[T] | T:
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
            # Drop dimensions.  TODO still reqd?
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

# ----------------------------------------------------------------------------
