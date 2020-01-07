"""
Practical unit conversions that don't use any underlying absolute or fixed
values.

Contains:
    output_ucode_pwr    Convert numerical power strings to unicode.
    UnitsError          Exception class for units-specific errors.
    Units               Class representing constructions of base units.
    Dim                 Class representing a dimensioned value.
    add_base_unit       Function to add units to the library of base units.
    add_unit            Function to add a unit to the library.
    base_unit           Function to determine if a tuple is a base unit.
    get_conversion      Returns the conversion factor between two units.
    set_conversion      Sets the conversion factor between units (directional).
    similar_to          Function givinga set of known units similar to the
                        supplied unit.
    total_temperature_convert  Temperature conversion function.
    make_total_temp     Convert a temperature into its total equivalent.
"""

# Last updated: 28 December 2019 by Eric J. Whitney

from __future__ import annotations
import warnings
from collections import namedtuple
import operator
import re
from fractions import Fraction
import math
from typing import Optional, Any, Union, Iterable, Callable

from containers import WeightedDirGraph, MultiBiDict
from util import (from_ucode_super, UCODE_SS_CHARS, to_ucode_super,
                  coax_type, force_type, kind_div)

__all__ = ['output_ucode_pwr', 'UnitsError', 'Units', 'Dim',
           'add_base_unit', 'add_unit', 'base_unit', 'get_conversion',
           'set_conversion', 'similar_to', 'total_temperature_convert',
           'make_total_temp']

# Generate unicode chars for power values in strings.
output_ucode_pwr = True

# Computed conversions are cached for faster repeat access. To ensure
# accurate handling of values only the requested conversion 'direction' is
# cached; the reverse conversion is not automatically computed and would
# need to be separately cached when encountered.
cache_computed_convs = True

# Note: There is presently no size limit on this cache. This is normally not
# a problem as only a few types of conversions occur in any application.

# Issue a warning if a unit conversion traces a path longer than
# _CONV_PATH_LENGTH_WARNING.  It could indicate a potential buildup of error
# and usually means that a suitable conversion for these units should be
# added.
_CONV_PATH_LENGTH_WARNING = 4

# -----------------------------------------------------------------------------

# Module level containers.

_known_units = MultiBiDict()  # All Units(), base and derived.
_known_base_units = set()  # Single entry ^1 Units().
_conversions = WeightedDirGraph()  # Conversions between Units().
_comp_conv_cache = {}  # {(Units, Units): Factor, ...}


class UnitsError(Exception):
    """Exception class signifying errors specifically related to units /
    conversions / etc. Other types of error are signalled normally."""
    pass


# Precompiled parser.

_UC_SGN = UCODE_SS_CHARS[0][0:2]
_UC_DOT = UCODE_SS_CHARS[0][2]
_UC_DIG = UCODE_SS_CHARS[0][3:]
_uc_pwr_pattern = fr'''[{_UC_SGN}]?[{_UC_DIG}]+(?:[{_UC_DOT}][{_UC_DIG}]+)?'''

_unitparse_rx = re.compile(fr'''
    ([.*×/])?                                               # Operator.
    (?:([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[.*×]?))?    # Mult w/ sep
    ([a-zA-Z_Δ°μ]+)                                         # Unit
    ((?:\^[+-]?[\d]+(?:[.][\d]+)?)|                         # Pwr (ascii), or
    (?:{_uc_pwr_pattern}))?                                 # Pwr (ucode)
    |(.+)                                                   # OR mismatch.
''', flags=re.DOTALL | re.VERBOSE)

# EJW: ((?:\^[+-]?[\d]+(?:[\.][\d]+)?)|  # Removed \ from this line before .

# Special case units.

_OFFSET_TEMPS = {'°C', '°F'}


# -----------------------------------------------------------------------------
# Field names:
#   k:  Multiplier (always first).
#   M:  Mass.
#   L:  Length.
#   T:  Time.
#   θ:  Temperature.
#   N:  Amount of substance.
#   I:  Electric current.
#   J:  Luminous intensity.
#   A:  Plane angle (1).
#   Ω:  Solid angle (1).

# Notes:
#   (1) Plane angle radians and solid angle steradians are dimensionless and
#   eliminated by multiplication and division.  These 'units' are retained
#   to allow consistency checks on mathematics, especially if degrees are
#   used, etc.


class Units(namedtuple('_Units', ['k', 'M', 'L', 'T', 'θ', 'N', 'I', 'J',
                                  'A', 'Ω'],
                       defaults=[1, *(('', 0),) * 9])):
    """
    The Units object expresses any unit (base or derived) as a tuple of an
    overall multiplier k and its basis units.  Each basis unit is a
    tuple of a string label and power.  Units are hashable so they can be
    looked up for quick conversions.
    """

    # noinspection PyTypeChecker
    def __new__(cls, *args, **kwargs):
        """
        Units objects can be created in three ways:
            - As conventional namedtuple where the first item is the factor
            and remaining items are each basis unit as a tuple of an
            alpha-only label and power. Unused fields default to
            dimensionless.  This example is equivalent to a metric tonne:
            >>> print(Units(1000, ('kg', 1)))
            1000*kg

            Alternatively the named fields can be used:
            >>> print(Units(k=1000, M=('kg', 1)))
            1000*kg

            No arguments or a single None results in a dimensionless unit
            object:
            >>> print(Units())  # k = 1, all bases are ('', 0)
            (No Units)
            >>> print(Units(None))  # Same as Units().
            (No Units)

            - As a string expression to be parsed.  This will multiply out
            sub-units L -> R giving the resulting unit.  Whitespace is
            ignored. Unicode characters for powers and separators can be
            used. Note: the divide symbol is not supported due it looking
            very similar to the 'plus' on-screen.
            >>> print(Units('1000 × kg'))
            1000*kg
            >>> print(Units('N.m^-2'))
            Pa
            >>> print(Units('slug.ft/s²'))
            lbf
            >>> print(Units('kg*m*s⁻²'))
            N
        """
        if len(args) == 1 and args[0] is None:
            # Intentionally blank case: Units(None).
            res = super().__new__(cls)
            _check_units(res)
            return res

        if not (len(args) == 1 and isinstance(args[0], str)):
            # All other normal cases, including Units().  Handle positional
            # and keyword arguments as a tuple.
            res = super().__new__(cls, *args, **kwargs)
            _check_units(res)
            return res

        # Remaining case is a single string argument is for parsing.  If
        # already known, return a copy.
        if args[0] in _known_units:
            res = super().__new__(cls, *_known_units[args[0]])
            _check_units(res)
            return res

        # Otherwise, parse string into subunits.
        tokens = re.findall(_unitparse_rx, ''.join(args[0].split()))
        if tokens[0][0]:
            raise ValueError("Invalid leading operator in unit "
                             "definition.")

        combined_units = Dim(1, None)
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
                sub_sig = Dim(1, _known_units[sub_label])
            except KeyError:
                # If the sub-units doesn't exist, we *really* don't
                # know this unit.
                raise ValueError(f"Unknown sub-unit '{sub_label}' in "
                                 f"unit definition.")

            pwrstr = pwrstr.strip('^')  # Reqd for ASCII powers.
            pwrstr = from_ucode_super(pwrstr)
            if pwrstr:
                pwr = force_type(pwrstr, int, float)
            else:
                pwr = 1
            if opstr == '/':
                mult = kind_div(1, mult)
                pwr = -pwr

            combined_units *= mult * sub_sig ** pwr

        # Return with k factored back in.
        res = Units(combined_units.value, *combined_units.units[1:])
        _check_units(res)
        return res

    def __mul__(self, rhs):
        """For all multiplication / division (and right side versions) Units()
        (and the LHS if required) are promoted to Dim() which then does the
        multiplication."""
        return Dim(1, self) * rhs

    def __truediv__(self, rhs):
        """Same rules as __mul__."""
        return Dim(1, self) / rhs

    def __rmul__(self, lhs):
        """Same rules as __mul__."""
        return Dim(lhs) * Dim(1, self)

    def __rtruediv__(self, lhs):
        """Same rules as __mul__."""
        return Dim(lhs) / Dim(1, self)

    # Misc operators.

    def __str__(self):
        """If a unique label exists return it, otherwise make a generic
        string. """
        try:
            labels = _known_units.inverse[self]
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
                    if output_ucode_pwr:
                        u_substr += to_ucode_super(f'{base[1]}')
                    else:
                        u_substr += f'^{base[1]}'
                base_parts += [u_substr]
        label = '.'.join(base_parts)
        if not label:  # Cover dimless case.
            label = '(No Units)'
        if self.k != 1:
            label = f'{self.k}*' + label
        return label

    # noinspection PyUnresolvedReferences
    def __repr__(self):
        unique_label = _unique_unit_label(self)
        if unique_label:
            return f"Units('{unique_label}')"

        arglist = []
        for f in self._fields:
            thisarg = getattr(self, f)
            if thisarg != self._fields_defaults[f]:
                arglist += [f + '=' + repr(thisarg)]
        return 'Units(' + ', '.join(arglist) + ')'

    def dimless(self) -> bool:
        """Returns false if any indicies in the signature are nonzero,
        otherwise true."""
        if any(p != 0 for _, p in self[1:]):
            return False
        else:
            return True


# -----------------------------------------------------------------------------


class Dim:
    def __init__(self, *args):
        """
        Constructs a dimensioned quantity, consisting of a value, units and
        optional label.  Calling methods:
        - Dim(): Assumes value = 1 and dimensionless.
        - Dim(Units): Assumes value = 1.  Promotes Units() to Dim().
        - Dim(value): Assumes dimensionless.  Promotes value to Dim().
        - Dim(value, units): Normal construction.
            - If units is Units() object: Directly assigned, no label given.
            - If units is string:  Parsed into Units() and used as label.
            - If units is None: Dimensionless (equivalent to Units())
        """
        self.label = None

        if len(args) == 0:
            self.value, self.units = 1, Units()

        elif len(args) == 1:

            if isinstance(args[0], Units):
                self.value, self.units = 1, args[0]
            else:
                self.value, self.units = args[0], Units()
                if isinstance(args[0], str):
                    warnings.warn("Warning: Assigned string to Dim() value: " +
                                  args[0])

        elif len(args) == 2:
            self.value = args[0]
            if isinstance(args[1], Units):
                self.units = args[1]
            elif isinstance(args[1], str):
                self.units = Units(args[1])
                self.label = args[1]
            elif args[1] is None:
                self.units = Units()
            else:
                raise TypeError(f"Units must be string, Units() or None.")

        else:
            raise TypeError(f"Incorrect number of arguments.")

    #
    # Unary operators.
    #

    def __abs__(self):
        ret_val = Dim(abs(self.value), self.units)
        ret_val.label = self.label
        return ret_val

    def __float__(self):
        """Note that units are removed."""
        return float(self.value)

    def __int__(self):
        """Note that units are removed."""
        return int(self.value)

    def __neg__(self):
        ret_val = Dim(-self.value, self.units)
        ret_val.label = self.label
        return ret_val

    def __round__(self, n=None):
        ret_val = Dim(round(self.value, n), self.units)
        ret_val.label = self.label
        return ret_val

    #
    # Binary operators.
    #

    def __add__(self, rhs) -> Dim:
        """If RHS is Units() or an ordinary value, it is promoted before
        addition (allows the case of dimensionless intermediate product).
        Addition to offset temperatures is limited to Δ values only."""
        if not isinstance(rhs, Dim):
            rhs = Dim(rhs)

        new_rhs_units = self.units

        # Offset temperatures are a special case.
        if self.units.θ[0] in _OFFSET_TEMPS:
            assert base_unit(self.units)

            # Disallowed:  LHS (°C, °F) + RHS (°C, °F)
            if rhs.units.θ[0] in _OFFSET_TEMPS:
                raise UnitsError(f"Cannot add offset temperatures: "
                                 f"{self.units.θ[0]} + {rhs.units.θ[0]}")

            # For LHS (°C, °F) change conversion target to (Δ°C, Δ°F)
            # noinspection PyProtectedMember
            new_rhs_units = new_rhs_units._replace(**{'θ': ('Δ' +
                                                            self.units.θ[0],
                                                            1)})

        # Normal addition.
        res_value = self.value + rhs.convert(new_rhs_units).value
        res_value = coax_type(res_value, type(self.value + rhs.value),
                              default=res_value)

        if not self.units.dimless():
            res = Dim(res_value, self.units)
            res.label = self.label
            return res
        else:
            return res_value

    def __sub__(self, rhs) -> Dim:
        """Similar rules to __add__.  Subtraction of offset temperatures is
        permitted and results in Δ value."""
        if not isinstance(rhs, Dim):
            rhs = Dim(rhs)

        # LHS offset temperatures are a special case.
        res_units = self.units
        new_rhs_units = self.units
        if self.units.θ[0] in _OFFSET_TEMPS:
            assert base_unit(self.units)

            # Possible cases are:
            # Case 1: LHS (°C, °F) - RHS (°C, °F) -> Result (Δ°C, Δ°F):
            #   - Direct conversion of RHS -> LHS and subtract.  Return Δ
            #   result.
            # Case 2: LHS (°C, °F) - RHS (All others Δ) -> Result (°C, °F)
            #   - Convert to corresponding (Δ°C, Δ°F) then subtract.  Return
            #   offset result.

            if rhs.units.θ[0] in _OFFSET_TEMPS:
                res_units = Units('Δ' + self.units.θ[0])
            else:
                new_rhs_units = Units('Δ' + self.units.θ[0])
        else:
            # Isolated RHS offset temperatures are not allowed.
            if rhs.units.θ[0] in _OFFSET_TEMPS:
                raise UnitsError(f"Cannot subtract offset from total "
                                 f"temperatures: {self.units} - {rhs.units}")

        # Normal subtraction.
        res_value = self.value - rhs.convert(new_rhs_units).value
        res_value = coax_type(res_value, type(self.value + rhs.value),
                              default=res_value)

        if not res_units.dimless():
            res = Dim(res_value, res_units)
            res.label = self.label
            return res
        else:
            return res_value

    def __mul__(self, rhs):
        """Multiplication of dimensioned values.  This is the central
        function used by most other operators / conversions.

        If either LHS or RHS units have a k-factor, this is multiplied out
        into the value leaving k = 1 for both LHS and RHS. If RHS is Units()
        or an ordinary value, it is promoted before multiplication.

        If the resulting units include radians^1 or steradians^1,
        these disappear as they are dimensionless and have served their
        purpose.
        """
        # Shortcut scalar multiplication. Check for dimless case.
        if not isinstance(rhs, (Dim, Units)):
            if not self.units.dimless():
                res = Dim(self.value * rhs, self.units)
                res.label = self.label
                return res
            else:
                return self.value * rhs

        if not isinstance(rhs, Dim):
            rhs = Dim(rhs)  # Promote.

        res_k = rhs.units.k * self.units.k
        res_basis = []
        for (l_u, l_p), (r_u, r_p) in zip(self.units[1:], rhs.units[1:]):
            res_p = l_p + r_p
            res_u = l_u if l_u else r_u

            # Multiply by a factor if required.
            if l_u and r_u and l_u != r_u:
                factor = get_conversion(r_u, l_u)
                if factor is None:
                    raise UnitsError(f"No conversion available for "
                                     f"{r_u} -> {l_u}.")
                res_k *= factor ** r_p

            # Store or cleanup if cancelled out.
            if res_p != 0:
                res_basis += [(res_u, res_p)]
            else:
                res_basis += [('', 0)]

        # Build units and check if multiplication has cancelled radians.
        res_units = Units(1, *res_basis)
        if res_units.A == ('rad', 1):
            # noinspection PyProtectedMember
            res_units = res_units._replace(A=('', 0))

        # Uncertain at this stage that steradians cancel out. XX TODO
        # if res_units.Ω == ('sr', 1):
        #     res_units = res_units._replace(Ω=('', 0))

        res_value = self.value * rhs.value * coax_type(res_k, (int, float),
                                                       default=res_k)

        if not res_units.dimless():
            return Dim(res_value, res_units)
        else:
            return res_value

    def __truediv__(self, rhs):
        """Same rules as __mul__."""
        if not isinstance(rhs, Dim):
            rhs = Dim(rhs)
        return self * (rhs ** -1)

    def __pow__(self, pwr):
        """Raise to pwr.  Units k is multiplied out into value."""

        # Build resulting units. Most units have integer powers; try to
        # preserve int-ness.
        res_basis = []
        for u, p in self.units[1:]:
            res_basis += [(u, coax_type(p * pwr, int, default=p * pwr))]
        res_units = Units(1, *res_basis)

        # Try to retain value type in result.
        res_value = (self.units.k * self.value) ** pwr
        res_value = coax_type(res_value, type(self.value), default=res_value)

        if not res_units.dimless():
            return Dim(res_value, res_units)
        else:
            return res_value

    def __radd__(self, lhs):
        """LHS is promoted to Dim()."""
        return Dim(lhs) + self

    def __rsub__(self, lhs):
        """LHS is promoted to Dim()."""
        return Dim(lhs) - self

    def __rmul__(self, lhs):
        """Commutaive with __mul__."""
        return self * lhs

    def __rtruediv__(self, lhs):
        """LHS is promoted to Dim()."""
        return Dim(lhs) / self

    #
    # Comparison operators.
    #

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

    # String magic methods.

    def __format__(self, format_spec: str):
        if self.label is not None:
            ustr = self.label
        else:
            ustr = str(self.units)
        return format(self.value, format_spec) + f" {ustr}"

    def __repr__(self):
        if self.label is not None:
            ustr = repr(self.label)
        else:
            ustr = _unique_unit_label(self.units)
            if ustr:
                ustr = repr(ustr)
            else:
                ustr = repr(self.units)
        return f"Dim({self.value}, {ustr})"

    def __str__(self):
        return self.__format__('')

    # Methods.

    def convert(self, to_units: Union[Units, str]) -> Dim:
        """
        Generate new object converted to compatible units.
        Args:
            to_units: Units object or label string.

        Returns:
            Dim.
        """
        use_label = None
        if isinstance(to_units, str):
            use_label = to_units
            to_units = Units(to_units)

        # Shortcut conversion to units already in use.
        if to_units == self.units:
            res = Dim(self.value, self.units)
            res.label = use_label or self.label
            return res

        # Conversion to °C and °F base units are special cased due to
        # offset of scales.
        from_temp, to_temp = self.units.θ[0], to_units.θ[0]
        if from_temp in _OFFSET_TEMPS or to_temp in _OFFSET_TEMPS:
            if not base_unit(self.units) or not base_unit(to_units):
                raise UnitsError(f"Cannot convert offset temperature "
                                 f"between derived units: {self.units} -> "
                                 f"{to_units}")
            res_value = total_temperature_convert(self.value, from_temp,
                                                  to_temp)

        else:
            # All rhs unit types.
            factor = get_conversion(self.units, to_units)
            if factor is None:
                raise UnitsError(f"No conversion found: {self.units} -> "
                                 f"{to_units}")
            res_value = self.value * factor
            res_value = coax_type(res_value, type(self.value),
                                  default=res_value)

        res = Dim(res_value, to_units)
        res.label = use_label
        return res


# -----------------------------------------------------------------------------

def _check_units(u_tuple):
    """Do certain checks on validity of unit tuple during construction."""
    # Blank units cannot have powers.  Note: Reverse is permitted, and is
    # useful for unit conversion.
    if any((not u and p != 0) for u, p in u_tuple[1:]):
        raise UnitsError(f"Blank units must have no power.")

    # Offset temperatures must be standalone.
    if u_tuple.θ[0] in _OFFSET_TEMPS and not base_unit(u_tuple):
        raise UnitsError(f"Offset temperature units are only permitted to "
                         f"be base units.")


def _common_cmp(lhs: Dim, rhs: Dim, op: Callable[..., bool]):
    """Common comparison method between two Dim objects called by all
    magic methods."""
    if lhs.units == rhs.units:
        return op(lhs.value, rhs.value)
    else:
        return op(lhs.value, rhs.convert(lhs.units).value)


def _unique_unit_label(u: Units) -> str:
    """If 'u' is unique in _known_units return it's label.  Otherwise return
    an empty string."""
    matches = _known_units.inverse.get(u, [])
    if len(matches) == 1:
        return matches[0]
    else:
        return ''


def add_base_unit(labels: Iterable[str], base: str) -> None:
    """
    Simplified version of add_unit(...) for base unit cases where k = 1,
    power = 1.  Labels are taken in turn and assigned to the common base
    dimension given. E.G. add_base_unit(['kg', 'g', 'mg'], 'M') creates mass
    base units of kg, g, mg.

    Args:
        labels: Iterable of strings.
        base: String.

    Returns:
        None.
    """
    for label in labels:
        uargs = {base: (label, 1)}
        add_unit(label, Units(**uargs))


def add_unit(label: str, new_unit: Union[Units, str]) -> None:
    """
    Register a new standard Units() object with given label.

    Args:
        label: Case-sensitive string to assign as a label for the resulting
        Units object.
        new_unit: Units() object or string to be parsed for conversion
        to Units().  String can include alphabetic characters, ° or Δ.

    Returns:
        None.

    Raises:
        ValueError for incorrect arguments.
        TypeError for incorrect argument types.
        UnitsError for mismatched labels.
    """
    # Check label.
    allowed_chars = {'_', '°', 'Δ'}
    if any(not c.isalpha() and c not in allowed_chars for c in label):
        raise ValueError(f"Invalid unit label '{label}'.")

    if label in _known_units:
        raise ValueError(f"Unit '{label}' already defined.")

    if isinstance(new_unit, str):
        new_unit = Units(new_unit)

    if not isinstance(new_unit, Units):
        raise TypeError(f"Expected Units or str argument, got "
                        f"'{type(new_unit)}'.")

    if new_unit.k == 0:
        raise UnitsError(f"k must be non-zero.")

    _known_units[label] = new_unit

    # Is this a derived unit? If so we are finished.
    base_u = base_unit(new_unit)
    if not base_u:
        return

    # Base units are added to known list.  Labels must match.
    if base_u != label:
        raise UnitsError(f"Basis unit label must match Units object, got: "
                         f"'{label}' != '{base_u}'")
    _known_base_units.add(new_unit)


def base_unit(u_tuple):
    """If u_tuple is a base unit, i.e. k = 1 and only one linear base unit
    is used, the string label is returned, otherwise returns None."""
    bases_used, last_u, last_p = 0, None, 0
    for u, p in u_tuple[1:]:
        if p != 0:
            bases_used += 1
            last_u, last_p = u, p
    if bases_used != 1 or last_p != 1 or u_tuple.k != 1:
        return None
    else:
        return last_u


def get_conversion(from_unit: [Units, str], to_unit: [Units, str]):
    """
    Determine the conversion factor between from_unit and to_unit.  When
    multiplying a from_label quantity this factor would result in the
    correct to_units quantity.   A three step process is used:
    1. If caching is active, see if this conversion has been done previously.
    2. Look for a conversion by tracing between (combining) Units()
    conversions already known.
    3. Compute the factor by breaking down Units() objects corresponding to
    from_units and to_units.  Note: This step is skipped when converting
    base units because in that case it cannot be further broken down by
    computation (and trying to do so would result in an infinite loop).

    Note: If caching is active, this can occur at Step 2 or 3.

    Args:
        from_unit: Units object or string label.
        to_unit: Units object or string label.

    Returns:
        Conversion factor coaxed to int or float

    Raises:
        UnitsError for inconsistent units or indicies.
    """
    if isinstance(from_unit, str):
        from_unit = Units(from_unit)
    if isinstance(to_unit, str):
        to_unit = Units(to_unit)

    # Conversion to lhs gives unity.
    if from_unit == to_unit:
        return 1

    # Step 1: See if this conversion has been cached.
    if cache_computed_convs:
        try:
            return _comp_conv_cache[from_unit, to_unit]
        except KeyError:
            pass

    # Step 2: See if a shortcut is available (base units will only take
    # this path).
    if (from_unit in _conversions) and (to_unit in _conversions):
        path, factor = _conversions.trace(from_unit, to_unit, operator.mul)
        if path and len(path) > _CONV_PATH_LENGTH_WARNING:
            warnings.warn(f"Warning: Converting '{from_unit}' -> '{to_unit}'"
                          f" gives overlong path: " + ' -> '.join(str(x) for
                                                                  x in path))
        if factor is not None:
            if cache_computed_convs:
                _comp_conv_cache[from_unit, to_unit] = factor
            return factor

    # Don't compute conversions if we are at base units (otherwise infinite
    # loop).
    if (from_unit in _known_base_units) or (to_unit in _known_base_units):
        return None

    # Step 3: Try to compute the factor from the base units.  Conversion is
    # equivalent to multiplying by (1/k)*to_units^0 on LHS.  Also check
    # consistency.
    new_basis = []
    for (from_u, from_p), (to_u, to_p) in zip(from_unit[1:], to_unit[1:]):
        # Dim must both be either something or nothing.
        if bool(from_u) != bool(to_u):  # XOR check.
            raise UnitsError(f"Inconsistent base units converting "
                             f"'{from_unit}' -> '{to_unit}': "
                             f"'{from_u}' and '{to_u}'")

        # Must have same indicies.
        if from_p != to_p:
            raise UnitsError(f"Inconsistent indices converting "
                             f"'{from_unit}' -> '{to_unit}': "
                             f"'{from_u}^{from_p}' and '{to_u}^{to_p}'")
        new_basis += [(to_u, 0)]

    factor = (Dim(1 / to_unit.k, Units(1, *new_basis)) * from_unit).value
    if cache_computed_convs:
        _comp_conv_cache[from_unit, to_unit] = factor
    return factor


def set_conversion(from_label: str, to_label: str, *, fwd: Any,
                   rev: Optional[Any] = 'auto') -> None:
    """
    Set a conversion between units in the directed graph of conversions.

    Args:
        from_label: String label for known unit.
        to_label: String label for known unit.
        fwd: Value of the multiplier to convert from -> to.
        rev: (Optional) Value of reverse conversion to -> from.  If rev ==
        None, no reverse conversion is added.  If rev == 'auto' then a
        reverse conversion is deduced from the forward conversion as follows:
            - If fwd == Fraction -> rev = 1/Fraction.
            - If fwd == int > 1 -> rev = Fraction(1, fwd).
            - All others:  -> rev = 1/fwd and coax to int if possible.

    Raises:
        ValueError if either unit does not exist or not a base unit.
    """
    from_unit = _known_units[from_label]
    to_unit = _known_units[to_label]

    _conversions[from_unit:to_unit] = fwd  # Keys are namedtuples.

    if rev is None:
        return
    if rev != 'auto':
        _conversions[to_label:from_label] = rev
        return

    # Deduce a sensible reverse conversion.
    if isinstance(fwd, Fraction):
        rev = 1 / fwd
        if rev.denominator == 1:  # Check if int.
            rev = rev.numerator
    elif isinstance(fwd, int) and fwd > 1:
        rev = Fraction(1, fwd)
    else:
        rev = 1 / fwd
        rev = coax_type(rev, int, default=rev)

    _conversions[to_unit:from_unit] = rev


def similar_to(example: Union[Units, Dim, str]) -> set:
    """
    Returns a set of labels of all known units with the same dimensions as the
    example given.
    Args:
        example: Units or Dim object.

    Returns:
        Set of label strings.
    """
    if isinstance(example, Dim):
        example = example.units
    if isinstance(example, str):
        example = Units(example)

    matching = set()
    for label, unit in _known_units.items():
        if all(True if kn_p == ex_p else False
               for (_, kn_p), (_, ex_p) in zip(unit[1:], example[1:])):
            matching.add(label)
    return matching


def total_temperature_convert(x, from_u: str, to_u: str):
    """
    Conversions of total temperatures are a special case due to the offset of
    the °C and °F base units scales.  Conversion is simplified by converting x
    to Kelvin then converting to final units.

    Args:
        x:  Value to be converted
        from_u: String: °C, °F, K, °R
        to_u: String as above.

    Returns:
        x converted to new units.
    """
    # Convert 'from' -> Kelvin.
    if from_u == '°C':
        x += 273.15
    elif from_u == 'K':
        pass
    elif from_u == '°F':
        x = (x + 459.67) * 5 / 9
    elif from_u == '°R':
        x *= 5 / 9
    else:
        raise UnitsError(f"Cannot convert total temperature: {from_u} -> "
                         f"{to_u}")

    # Convert Kelvin -> 'to'.
    if to_u == '°C':
        x -= 273.15
    elif to_u == 'K':
        pass
    elif to_u == '°F':
        x = (x * 9 / 5) - 459.67
    elif to_u == '°R':
        x *= 9 / 5
    else:
        raise UnitsError(f"Cannot convert total temperature: {from_u} -> "
                         f"{to_u}")

    return x


def make_total_temp(x: Dim) -> Dim:
    """
    Function to convert an offset to a total temperature if required
    (°C -> K or °F -> °R).  Raises ValueError for Δ°C or Δ°F.  If neither,
    it returns the original object.
    Args:
        x: Dim object.

    Returns:
        Dim object converted if required.

    Raises:
        ValueError if x = delta temperature.
    """
    if x.units == Units('°C'):
        return x.convert('K')
    elif x.units == Units('°F'):
        return x.convert('°R')
    elif x.units.θ[0] == 'Δ':
        raise ValueError(f"Delta temperature not allowed.")
    else:
        return x


# -----------------------------------------------------------------------------

#
# Base units.
#

#
# Mass.
#

#  Signatures.
add_base_unit(['t', 'kg', 'g', 'mg', 'μg'], 'M')
# Note: t = Metric tonne.
add_base_unit(['slug', 'lbm'], 'M')

# Conversions.
set_conversion('g', 'μg', fwd=1e6)
set_conversion('kg', 'g', fwd=1000)
set_conversion('g', 'mg', fwd=1000)
set_conversion('lbm', 'kg', fwd=0.45359237)  # Defn Intl & US Standard Pound.
set_conversion('slug', 'lbm', fwd=32.17404855643045)  # CALC of G in ft/s^2
set_conversion('t', 'kg', fwd=1000)  # Metric tonne.

#
# Length.
#

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
# 'mi' is is the international mile defined relative to the international yard
# and foot.
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

#
# Time.
#

#  Signatures.
add_base_unit(['day', 'hr', 'min', 's', 'ms'], 'T')

# Conversions.
set_conversion('day', 'hr', fwd=24)
set_conversion('hr', 'min', fwd=60)
set_conversion('min', 's', fwd=60)
set_conversion('s', 'ms', fwd=1000)

#
# Temperature.
#

#  Signatures.
add_base_unit(['°C', 'Δ°C', 'K'], 'θ')
add_base_unit(['°F', 'Δ°F', '°R'], 'θ')

# Conversions.  Note: Only temperature differences are given here.  Absolute
# temperatures are handled by a special function because the °C and °F
# scales are offset.
set_conversion('K', 'Δ°C', fwd=1)
set_conversion('K', 'Δ°F', fwd=Fraction(9, 5))  # Changed from float 9/5.
set_conversion('°R', 'Δ°F', fwd=1)

#
# Amount of substance.
#

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

#
# Electric current.
#

# Signatures.
add_base_unit(['A', 'mA'], 'I')

# Conversions.
set_conversion('A', 'mA', fwd=1000)

#
#   Luminous intensity.
#

# Signatures.
add_base_unit(['cd'], 'J')

# Conversions.

#
# Plane angle.
#

# Signatures.
add_base_unit(['rev', 'rad', 'deg', '°'], 'A')
# This is the only use of the ° symbol on its own, but note that ° != deg
# for comparison purposes.

# Conversions.
set_conversion('rev', 'rad', fwd=2 * math.pi)
set_conversion('rev', 'deg', fwd=360)
set_conversion('rad', 'deg', fwd=180 / math.pi)
set_conversion('deg', '°', fwd=1)

#
# Solid angle.
#

# Signatures.
add_base_unit(['sp', 'sr'], 'Ω')

# Conversions.
set_conversion('sp', 'sr', fwd=4 * math.pi)  # 1 spat = 4*pr steradians.

# -----------------------------------------------------------------------------

#
# Derived units.
#

# Notes:
# -  A a variety of signature types / operators / unicode (*, /, ×, ²,
# etc) are used in which acts as an automatic check on the parser.
# - Exact standard for G = 9.80665 m/s/s (WGS-84 defn). Full float
# conversion gives 32.17404855643045 func/s^2.

#
# Area
#

add_unit('ha', '10000*m^2')

#
# Volume
#

add_unit('cc', 'cm^3')
add_unit('L', '1000×cm^3')
add_unit('US_gal', '231×in^3')  # Fluid gallon
add_unit('Imp_gal', '4.54609*L')  # British defn
add_unit('US_qt', '0.25*US_gal')  # Fluid quart
add_unit('US_fl_bl', '31.5*US_gal')  # Fluid barrel
add_unit('US_hhd', '2×US_fl_bl')  # Hogshead

#
# Speed.
#

add_unit('kt', 'NM/hr')
add_unit('mph', 'sm/hr')  # Normally defined as US statute miles / hour.
add_unit('kph', 'km/hr')

#
# Acceleration.
#

add_unit('G', '9.80665 m/s^2')  # WGS-84 definition

#
# Force.
#

add_unit('kgf', 'kg×G')
add_unit('N', 'kg.m.s⁻²')
add_unit('kN', '1000×N')

add_unit('lbf', 'slug.ft/s²')
add_unit('kip', '1000*lbf')

#
# Pressure
#

#  Signatures.
add_unit('MPa', 'N/mm²')
add_unit('Pa', 'N/m²')
add_unit('hPa', '100.Pa')
add_unit('kPa', '1000*Pa')

add_unit('atm', '101325 Pa')  # ISO 2533-1975
add_unit('bar', '100000*Pa')
add_unit('mmHg', '133.322387415*Pa')
# mmHg conversion from BS 350: Part 1: 1974 – Conversion factors and tables
add_unit('Torr', f'{1 / 760}*atm')

add_unit('psf', 'lbf/ft^2')
add_unit('inHg', '25.4*mmHg')
add_unit('psi', 'lbf/in^2')
add_unit('ksi', '1000*psi')

#
# Energy.
#

add_unit('J', 'N.m')
add_unit('kJ', '1000.J')
add_unit('MJ', '1000.kJ')
add_unit('cal', '4.184×J')  # ISO Thermochemical calorie (cal_th).
add_unit('kcal', '1000.cal')
add_unit('Btu', '778.1723212164716×ft.lbf')  # ISO British Thermal Unit.
# The ISO Btu is defined as exactly 1055.06 J. The above value is the
# calculated conversion to ft.lbf.

#
# Power.
#

add_unit('W', 'J/s')
add_unit('kW', '1000*W')
add_unit('hp', '550×ft.lbf/s')

#
# Luminous.
#

add_unit('lm', 'cd.sr')  # Lumen.
