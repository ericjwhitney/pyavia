"""
Provides LiveProperty class which allows the addition of 'live properties'
to user classes.  This is for the case where you would immediately like to
call some class method once a particular attribute is updated (i.e.
__set__).  This can be used to update the internal state, etc.  This is also
useful if updating during every read (i.e. __get__) is computationally
expensive.

Calling del on the property deletes the corresponding internal
attribute and also calls the given method.  This is only useful if the
existence of the attribute is monitored by the update method.

Example
-------

We make a simple class with two properties x, y associated with internal
attributes _x, _y.  The class computes their sum and stores it in advance.
We want an update of either x or y to trigger a recompute.

>>> from live_property import LiveProperty
>>> # noinspection PyUnresolvedReferences
... class LiveSum:
...     def __init__(self):
...         self._x, self._y = 0, 0
...         self._update()
...
...     @property
...     def z(self):
...         return self._z
...
...     def _update(self):
...         self._z = self.x + self.y
...         print(f"z updated to {self._z}")
...
...     x = LiveProperty('_x', _update, "An x value.")
...     y = LiveProperty('_y', _update, "A y value.")

Now we setup an instance and change the properties:

>>> obj = LiveSum()
z updated to 0
>>> obj.x, obj.y = 7, 9  # obj._update() will be called twice.
z updated to 7
z updated to 16
>>> print(f"obj.z = {obj.z}")  # No update; z already computed.
obj.z = 16

Deleting the internal attribute also results in an _update() call.

>>> del obj.y
Traceback (most recent call last):
...
AttributeError: 'LiveSum' object has no attribute '_y'
"""
# Last updated: 1 December 2019 by Eric J. Whitney

from typing import Callable, Any


class LiveProperty:
    """A data descriptor designed to add 'live properties' to a class,
    where setting an attribute automatically produces a function call."""

    def __init__(self, internal_id: str, callback: Callable[[Any], None],
                 doc: str = None):
        """
        Constructs a data descriptor for making a class 'live property'.

        Args:
            internal_id: Name of the corresponding instance attribute to
            associate with the property.  Typically same as property name
            with leading underscore.
            callback: Instance method to call after setting property.
            doc: Optional docstring.
        """
        self.internal_id = internal_id
        self.callback = callback
        self.__doc__ = doc

    def __get__(self, instance, owner):
        if instance is None:
            raise AttributeError(f"Can only access {owner.__name__}."
                                 f"{self.internal_id} through an instance.")
        return getattr(instance, self.internal_id)

    def __set__(self, instance, value):
        setattr(instance, self.internal_id, value)
        self.callback(instance)

    def __delete__(self, instance):
        delattr(instance, self.internal_id)
        self.callback(instance)
