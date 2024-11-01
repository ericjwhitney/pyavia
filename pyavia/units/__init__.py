"""
Units (:mod:`pyavia.units`)
===========================

.. currentmodule:: pyavia.units

Units-aware calculations, objects and associated functions.

Examples
--------

Creation of dimensioned values is done using the factory function
``dim()`` in a natural way, and gives a ``Dim`` object as a result.  See
documentation for ``dim()`` for details on valid formats for the
dimension string.

>>> r = dim(10, 'ft')
>>> ω = dim(30, 'rad/s')
>>> r * ω  # Centripetal motion v_t = r.ω
dim(300, 'fps')

Calling dim() with no arguments results in a dimensionless unity value.

>>> dim()
dim(1, '')

This gives the expected result of dimensions dropping out in normal
calculations:
>>> 6 * 7 * dim()
42

The ``convert`` function can be used with normal numeric inputs to give
a conversion.  Note that the we try to preserve the type of the input
argument where possible (int to int, float to float, etc):
>>> convert(288, 'in^2', 'ft^2')  # Convert sq. in to sq. ft.
2

Alternatively, ``Dim`` objects can provide a converted value of
themselves directly:
>>> area = dim(144, 'in^2')
>>> area.convert('ft^2')
dim(1, 'ft^2')

Derived (compound) units will cancel out where possible to provide the
most direct base unit related to the LHS.  For example, because 1 N = 1
kg.m/s², then under 1 G acceleration:

>>> dim(9.80665, 'N') / dim(1, 'G').convert('ft/s^2')
dim(0.9999999999999999, 'kg')

Units with unusual powers are handled without difficulty:

>>> k_ic_metric = dim(51.3, 'MPa.m⁰ᐧ⁵')  # Metric fracture toughness.
>>> k_ic_metric.convert('ksi.in^0.5')  # Convert to imperial.
dim(46.68544726800077, 'ksi.in^0.5')

Temperature values are a special case and specific rules exist for
handling them.  This is because different temperatures can have offset
scales or represent different quantities:

    - `Absolute` or `Offset` scales:  `Absolute` temperatures have their
      zero values located at absolute zero whereas `offset` temperatures
      use some other reference point for zero.
    - Represent `Total` values or `Change`/`Δ`:  `Total` temperatures
      represent the actual temperature state of a body, whereas `Δ`
      values represent the change in temperature (or difference between
      two total temperatures).

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
temperature scale, they can be added or subtracted from total
temperatures.  As such these can be used with either absolute or offset
total temperature units, because there is a direct conversion available
to an appropriate type e.g.  Δ°C -> K, Δ°F -> °R.

A simple example converting from an absolute temperature scale to an
offset scale:

>>> dim(373.15, 'K').convert('°F')  # Offset temperature
dim(211.99999999999994, '°F')

Temperature changes can be added or subtracted from total temperatures,
giving total temperatures:
>>> dim(25, '°C') + dim(5, 'Δ°C')
dim(30, '°C')
>>> dim(25, '°C') - dim(5, 'Δ°C')
dim(20, '°C')

Two total temperatures on offset scales can be subtracted, but to make
sure everyone is clear on what is being done, a rule is enforced that
they must be on the same scale:

>>> dim(32, '°F') - dim(0, '°C')
... # doctest: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
ValueError: Total temperatures must be on the same scale for subtraction,
got: '°F' - '°C'

Total temperatures on absolute scales or temperature changes can be used
in derived / compound units and converted freely:

>>> air_const_metric = dim(287.05287, 'J/kg/K')
>>> air_const_imp_slug = air_const_metric.convert('ft.lbf/slug/°R')

Total temperatures on offset scales are not allowed to be used in
derived / compound units:

>>> dim('km.°F')  # doctest: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
ValueError: Offset temperature units are only permitted to be base units.

``Dim`` objects supports any format statements that can be used directly
on their numeric value:

>>> print(f"Slug value = {air_const_imp_slug:.2f}")
Slug value = 1716.56 ft.lbf/slug/°R

"""

from ._base import (block_conversion, convert, set_conversion,
                    to_absolute_temp)
from ._dim import dim, Dim
from . import _defs  # Sets up standard units.
