from unittest import TestCase

from live_property import LiveProperty


# Create a test class with two members x, y that computes their sum and stores
# it in advance.  Updating either x or y triggers a  recompute.

class LiveSum:
    def __init__(self):
        self._x, self._y = 0, 0
        self._update()

    @property
    def z(self):
        return self._z

    update_counter = 0

    def _update(self):
        self._z = self.x + self.y
        LiveSum.update_counter += 1

    x = LiveProperty('_x', _update, "Opt. docstr")
    y = LiveProperty('_y', _update)


# noinspection PyUnusedLocal
class WrongExample:
    def __init__(self):
        self._i = 2

    def _update(self):
        pass

    i = LiveProperty('_j', _update)  # Non-existent instance attribute.


class TestLiveProperty(TestCase):
    def test___init__(self):
        # Test basic functionality.
        self.assertEqual(LiveSum.update_counter, 0)
        obj = LiveSum()
        self.assertEqual(LiveSum.update_counter, 1)
        obj.x, obj.y = 7, 9  # obj._update() will be called twice.
        self.assertEqual(LiveSum.update_counter, 3)
        # noinspection PyUnusedLocal
        temp = obj.z  # Shouldn't call obj._update().
        self.assertEqual(LiveSum.update_counter, 3)

        # Check no cross-talk.
        obj2 = LiveSum()
        self.assertEqual(obj2.x, 0)
        self.assertEqual(obj2.y, 0)
        self.assertEqual(obj2.z, 0)
        obj2.y = 13
        obj2.x = 11
        self.assertEqual(obj2.z, 24)

        # Check can't use on non-existent attribute.
        with self.assertRaises(AttributeError):
            # noinspection PyStatementEffect
            WrongExample().i  # Property was assigned to _j.
