from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# Written by Eric J. Whitney, January 2020.

# ======================================================================


@dataclass(frozen=True, kw_only=True)
class UnitOptions:
    """
    Dataclass that holds option flags for handling units.  See
    'get_unit_options' and  'set_unit_options' for full details.
    """
    cache_conversions: bool
    cache_made_units: bool
    conversion_length_warning: int
    unicode_str: bool

    def __post_init__(self):
        """Check certain values"""
        if self.conversion_length_warning <= 1:
            raise ValueError("Require 'conversion_length_warning' > 1.")


# Create single instance and set defaults.
_unit_options = UnitOptions(
    cache_conversions=True,
    cache_made_units=True,
    conversion_length_warning=4,
    unicode_str=True
)


# ----------------------------------------------------------------------

def get_unit_options() -> {str: Any}:
    """
    Returns
    -------
    unit_options : UnitOptions
        Returns a UnitOptions object containing the options.  For a
        full description of each option, see `set_unit_options`.
    """
    return replace(_unit_options)


# noinspection PyIncorrectDocstring
def set_unit_options(**kwargs):
    """
    Set the current unit options.

    Parameters
    ----------
    cache_conversions : bool, default = True
        If `True`, computed conversions are cached for faster repeat
        access. To ensure accurate handling of values only the requested
        conversion 'direction' is cached; the reverse conversion is not
        automatically computed and would need to be separately cached
        when encountered.

        .. note:: There is presently no size limit on this cache. This
           is normally not a problem as only a few types of conversions
           occur in any application.

    cache_made_units : bool, default = True
        If `True`, cache the internal representations of units generated
        by parsing non-standard unit strings at runtime.  This allows
        for faster repeat access and provides a large performance
        improvement.

        .. note:: There is presently no size limit on this cache. This
           is normally not a problem as only a few types of conversions
           occur in any application.

    conversion_length_warning : int, default = 4
        Issue a warning if a unit conversion traces a path longer than
        this value to determine the final conversion factor. It could
        indicate a potential buildup of error and usually means that a
        suitable conversion for these units should be added.

    unicode_str : bool, default = True
        Generate unicode characters for superscripts when the `__str__`
        method is called on dimensioned values.

    See Also
    --------
    get_unit_options, unit_options

    Examples
    --------
    By default the square value is represented with a unicode character:
    >>> from pyavia.units import dim, set_unit_options
    >>> area = dim(4.0, 'ft')**2
    >>> print(area)
    16.0 ft²

    TODO ^^^ THIS SEEMS TO BE HIT OR MISS?

    If this is disabled, the caret '^' is used instead:
    >>> set_unit_options(unicode_str=False)
    >>> second_area = dim(4.0, 'ft')**2
    >>> print(second_area)
    16.0 ft^2
    """
    global _unit_options
    _unit_options = replace(_unit_options, **kwargs)

    # TODO WAS
    # # Get non-None arguments and merge with current options.
    # opt_args = {k: v for k, v in locals().items() if v is not None}
    # _unit_options.update(opt_args)
    #
    # # Check options are legal.
    # if _unit_options['conversion_path_warning'] <= 1:
    #     raise ValueError("Require 'conversion_length_warning' > 1.")

# ----------------------------------------------------------------------

# TODO Context handler unit_options
