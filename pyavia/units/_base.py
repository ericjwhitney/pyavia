from __future__ import annotations
from collections import namedtuple
from fractions import Fraction
import operator
import re
from typing import Optional, Union, Any
import warnings

from pyavia.containers import MultiBiDict, WtDirGraph, wdg_edge
from pyavia.numeric.math_ext import kind_div
from pyavia.util.type_ops import coax_type, force_type
from ._opts import _unit_options

# TODO if this is the only place coax_type / force_type is used, move here?

# Written by Eric J. Whitney, January 2020.


# ======================================================================

# TODO change these into 'accessed' parameters like how NumPy does
#  printoptions.

# CACHE_COMPUTED_CONVS = True
# """<<MOVED>>"""

CACHE_MADE_UNITS = True
"""<<MOVED>>."""

# CONV_LENGTH_WARNING = 4  # TODO RENAME conversion_path_warning
# """<<MOVED>> """

# OUTPUT_UCODE_PWR = True
# """<<MOVED>>"""

# TODO reevaluate - Or just add default parts like mol, A, etc.
STD_UNIT_SYSTEM = 'kg.m.s.K.mol.A.cd.rad.sr'
"""This variable represents a standard system used as the default value for 
any calls to Dim.to_value_sys().  Some unit types may be omitted with default 
values assumed - refer to the documentation for ``Dim.to_value_sys()`` for 
details. """



# ----------------------------------------------------------------------

# TODO reevaluate
# TODO Old definitions
# DimScalarType = Union[int, float, complex]
# """`DimScalarType` is a shorthand defined for type checking porpoises as
# ``Union[int, float, complex]``.  These represent the restricted set of
# scalar types that the ``.value`` field of a `Dim` object may take.  For the
# complete set, refer to `DimValueType`"""
#
# DimValueType = Union[DimScalarType, npt.NDArray]
# """`DimValueType` is a shorthand defined for type checking porpoises as
# ``Union[DimScalarType, npt.NDArray]``.  These represent the possible
# types that the ``.value`` field of a `Dim` object may take."""
#
# DimOpType = Union['Dim', DimValueType]
# """`DimOpType` is a shorthand defined for type checking porpoises as
# ``Union[Dim, DimValueType]. This represents types that could appear either
# alongside of - or as the result of - a mathematical operation involving a
# `Dim` object."""
#

ConvFactorType = int | float | Fraction
"""`ConvFactorType` represents possible types that conversion factors may 
take (excluding temperatures which are a special case)."""


# ===========================================================================


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


# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------


def block_conversion(label: str):
    """
    This is used when automatic conversion is to be expressly prohibited
    (e.g. offset temperatures °C, °F, etc).  It is a backup measure to
    ensure that such a conversion isn't accidentally added by a user.
    """
    _BLOCKED_CONVS.add(_Units(label))


# ---------------------------------------------------------------------------

def convert(value,  # TODO removed old type hint, add new.
            from_units: str, to_units: str):  # TODO add hint
    """
    Convert `value` currently in `from_units` to requested `to_units`.
    This is used for doing conversions without using `Dim` objects.

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


# ---------------------------------------------------------------------------

def set_conversion(from_label: str, to_label: str, *, fwd: ConvFactorType,
                   rev: Union[ConvFactorType, str, None] = 'auto'):
    """
    Set a conversion between units in the directed graph of conversions.

    Parameters
    ----------
    from_label, to_label : str
        Case-sensitive identifier.
    fwd : TODO
        Value of the multiplier to convert from -> to.
    rev : TODO or str (optional)
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


# ---------------------------------------------------------------------------

# TODO Is this in the right spot?
def to_absolute_temp(x: 'Dim') -> 'Dim':
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


# ---------------------------------------------------------------------------


# TODO is this still used?  Should it be in _Dim?
def to_zero(x) -> Any:
    """
    Shorthand function for ``x * 0`` that generates a zeroed version of the
    parameter `x` retaining any units.  If `x` is a plain number type,
    the zero version of that number type is returned.
    """
    return x * 0


# ---------------------------------------------------------------------------

# TODO Presently suspended, new unit system approach.
# # == Units-Aware Function Decorator
# =========================================
#
# def units_aware(input_units: {str, Union[str, None]}=None,
#                 output_units: [Union[str, None]] = None):
#     """
#     Decorator to make a function "units aware".  Input arguments are
#     converted to realvalues with the units given in `input_units` which are
#     then passed to the decorated function.  In this way the function does not
#     need to be specifically written to accomodate units.
#
#     .. Note:: The undecorated original function can still be accessed using
#        the ``__wrapped__`` property, e.g. ``func_nodims = func.__wrapped__``.
#
#     Parameters
#     ----------
#     input_units: {str: Union[str, None], ...}
#         A dict of input argument names with associated strings representing
#         valid units for that input argument.  The units must follow the
#         conventions of ``dim(...)``.  ``None`` can be used if a
#         specific argument has no units.  As a safety policy, all arguments
#         (including default arguments) must have units defined (or ``None``)
#         otherwise an exception is raised.
#
#     output_units: str or [Union[str, None], ...]
#         A unit string or sequence of strings to use when converting the
#         function output to `dim()` values.  If ``None`` then the output is
#         returned directly (no conversion via ``dim()``).  If only a single
#         unit string is given then the output is treated as a single value
#         and a tuple is not created.
#
#     Returns
#     -------
#     result : Any
#         The output as a `dim()` object where required (see `output_units`
#         above).
#
#     Raises
#     ------
#     ValueError :
#         If some arguments have inconsistent or missing units whether others
#         have been provided (or required in the case of ``units_reqd = True``.
#     """
#     if input_units is None:
#         input_units = {}
#
#     def decorator(func):
#         func_sig = signature(func)
#
#         @wraps(func)
#         def wrapped_func(*args, **kwargs):
#             # Get all input arguments bound to their names.
#             bound_args = func_sig.bind(*args, **kwargs)
#             bound_args.apply_defaults()
#             if bound_args.arguments.keys() != input_units.keys():
#                 raise ValueError(
#                     f"@units_aware: Input arguments "
#                     f"[{', '.join(input_units.keys())}] do not match wrapped "
#                     f"function [{', '.join(bound_args.arguments.keys())}].")
#
#             # Convert arguments where required.
#             for k, v in bound_args.arguments.items():
#                 to_units = input_units[k]
#
#                 if not isinstance(v, Dim):
#                     # Case: Plain argument supplied.  No conversion required
#                     # provided target units were 'None'.
#
#                     if to_units is not None:
#                         raise ValueError(f"@units_aware: Required Dim value "
#                                          f"not supplied for argument '{k}'.")
#
#                 else:
#                     # Case: Dim() supplied.
#                     if to_units is None:
#                         raise ValueError(f"@units_aware: Dim value supplied "
#                                          f"for argument '{k}' but no "
#                                          f"conversion given.")
#
#                     # Do conversion.
#                     bound_args.arguments[k] = v.to_value(to_units)
#
#             # Run function by passing in plain arguments.
#             res = func(*bound_args.args, **bound_args.kwargs)
#             if output_units is None:
#                 # Case: No dimensioned result required.  Return directly.
#                 return res
#
#             elif isinstance(output_units, str):
#                 # Case: Single conversion applied to result.
#                 return dim(res, output_units)
#
#             else:
#                 # Case: Convert each element of result.
#                 if not isinstance(res, Sequence):
#                     raise ValueError(
#                         f"@units_aware: Expected multiple return values "
#                         f"from function for conversion.")
#
#                 if len(res) != len(output_units):
#                     raise ValueError(f"@units_aware: Number of returned "
#                                      f"values doesn't match number of "
#                                      f"conversions requested.")
#
#                 # Give outputs dimensions in place.
#                 for i in range(len(res)):
#                     if output_units[i] is not None:
#                         # noinspection PyUnresolvedReferences
#                         res[i] = dim(res[i], output_units[i])
#
#                 return tuple(res) if len(res) > 1 else res[0]
#
#         return wrapped_func
#
#     return decorator


# ===========================================================================

_BLOCKED_CONVS: set[_Units] = set()  # Will be rejected by _convert().
_COMP_CONV_CACHE: {(_Units, _Units): ConvFactorType} = {}  # Cache computed.
_CONVERSIONS = WtDirGraph()  # Defined conversions between Units().
_KNOWN_BASE_UNITS: set[_Units] = set()  # Single entry ^1 Units().
_KNOWN_UNITS = MultiBiDict()  # All defined Units(), base and derived.
_MADE_UNITS = MultiBiDict()  # New units parsed at runtime.

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

        TODO Update
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
        # -> Check if already among known unit strings that have been
        # pre-checked.  If so, return a copy.
        try:
            return _Units(*_KNOWN_UNITS[args[0]])
        except KeyError:
            pass

        # -> Check for units already parsed during this session.
        if _unit_options.cache_made_units:
            try:
                return _Units(*_MADE_UNITS[args[0]])
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

        # -> Cache units parsed during this session.
        if _unit_options.cache_made_units:
            _MADE_UNITS[args[0]] = res_units

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

    def __pow__(self, pwr: Union[int, float]) -> _Units:
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
        If a unique label already exists return it, otherwise builds a generic
        string.
        """
        try:
            labels = _KNOWN_UNITS.inverse[self]
            if len(labels) == 1:
                return labels[0]
        except KeyError:
            pass

        if _unit_options.cache_made_units:
            try:
                labels = _MADE_UNITS.inverse[self]
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
                    if _unit_options.unicode_str:
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
    # TODO rename 'unit' to 'symbol'.  Rename 'basis' to something more
    #  useful?
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


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def _basis_factor(from_units: _Units, to_basis: _Units) -> ConvFactorType:
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


# -----------------------------------------------------------------------------

def _conversion_factor(from_basis: _Units,
                       to_basis: _Units) -> Optional[ConvFactorType]:
    """
    Compute the factor for converting between two units.  This is not used
    for total temperatures which are handled separately.
    """
    # Conversion to lhs gives unity.
    if from_basis == to_basis:
        return 1

    # Step 1: See if this conversion has been cached.
    # if CACHE_COMPUTED_CONVS:  TODO WAS
    if _unit_options.cache_conversions:
        try:
            return _COMP_CONV_CACHE[from_basis, to_basis]
        except KeyError:
            pass

    # Step 2: See if a shortcut is available (base units will take
    # this path only).
    if (from_basis in _CONVERSIONS) and (to_basis in _CONVERSIONS):
        path, factor = _CONVERSIONS.trace(from_basis, to_basis, operator.mul)
        if path and len(path) > _unit_options.conversion_length_warning:
            warnings.warn(
                f"Warning: Converting '{from_basis}' -> '{to_basis}' gives "
                f"overlong path: " + ' -> '.join(str(x) for x in path))

        if factor is not None:
            # if CACHE_COMPUTED_CONVS:  TODO WAS
            if _unit_options.cache_conversions:
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
            raise ValueError(
                f"Inconsistent indices converting '{from_basis}' -> "
                f"'{to_basis}': '{from_ustr}^{from_pwr}' and "
                f"'{to_ustr}^{to_pwr}'")
        new_basis += [(to_ustr, 0)]

    new_basis = _Units(1 / to_basis.k, *new_basis)

    # Do multiplication, factor contains the result.
    factor = (new_basis * from_basis).k
    if _unit_options.cache_conversions:
        _COMP_CONV_CACHE[from_basis, to_basis] = factor

    return factor


# -----------------------------------------------------------------------------

# TODO Is this the best place for this now?
def _convert(value,  # TODO removed old type hint, add new.
             from_units: _Units, to_units: _Units) -> ConvFactorType:
    """
    Convert ``value`` given ``_Units`` bases.
    """
    if from_units.is_total_temp() and to_units.is_total_temp():
        # Conversion between total temperatures are a special case due to
        # possible offset of scales.
        from_θ, to_θ = from_units.θ[0], to_units.θ[0]
        return _convert_total_temp(value, from_θ, to_θ)

    if from_units in _BLOCKED_CONVS:
        raise ValueError(f"Automatic conversion from unit "
                         f"'{str(from_units)}' is prevented.")
    if to_units in _BLOCKED_CONVS:
        raise ValueError(f"Automatic conversion to unit '{str(to_units)}' "
                         f"is prevented.")

    # All other unit types.
    factor = _conversion_factor(from_units, to_units)
    if factor is None:
        raise ValueError(f"No conversion found: '{from_units}' -> "
                         f"'{to_units}'")

    conv_value = value * factor
    conv_value = coax_type(conv_value, type(value), default=conv_value)
    return conv_value


# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------

def _convert_total_temp(x,  # TODO removed old type hint, add new.
                        from_unit: str, to_unit: str):  # TODO add hint
    """
    Conversions of total temperatures are a special case due to the offset of
    the °C and °F base units scales.

    .. note: Arithmetic is simplified by converting `x` to Kelvin then
       converting to final units.

    Parameters
    ----------
    x : TODO
        Temperature to be converted.
    from_unit, to_unit : str
        String matching °C, °F, K, °R.

    Returns
    -------
    result : TODO
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


# TODO Re-evaluate this.
# Default entries when doing conversion between consistent bases.  Types M,
# L, T, θ are not given here, they must always be provided.
_DEFAULT_UNIT_SYS = _Units(N=('mol', 0), I=('A', 0), J=('cd', 0),
                           A=('rad', 0), Ω=('sr', 0))


# ---------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------


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
