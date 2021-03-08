"""
The Switch class allows the writing of a ``switch ... case`` statement
in Python similar to what is provided in other languages.  Specifically:

- It is implemented as context manager allowing a discrete code block to
  be associated with the Switch containing multiple statements.

- Cases are written using ``if`` statements.

    - Strict ``if ... elif ...`` else clauses don't allow fall
      through and allow the ``else`` clause to be used as the `default
      case`.

    - Alternatively, softer ``if`` statements can be combined with the
      ``Switch.break_switch()`` class method which can be used to break
      from the current Switch context early.

Examples
--------

Here is a simple example of a Switch with strict ``if ... elif ...
else`` statement, with the else serving as the `default`:

>>> for x in range(5):
...     with Switch(x) as case:
...         if case(0):
...             print("Nothing.")
...         elif case(1, 2):  # Note multiple comparisons.
...             print("A few.")
...         else:  # Default clause.
...             print("Many.")
Nothing.
A few.
A few.
Many.
Many.

Here is an example of a function with two nested Switch statements.  The
nesting is not necessary but is included purely for illustration:

>>> # noinspection PyUnresolvedReferences
... def print_day_time(weekday, hour):
...     with Switch(weekday) as day_case:
...         with Switch(hour) as hr_case:  # Example showing nested Switch.
...             if hr_case(*range(12)):
...                 hr_str = "morning"
...                 Switch.break_switch()  # Breaks inner Switch.
...
...             if hr_case(12):  # If / elif / else for no fall-thru.
...                 hr_str = "midday"
...             elif hr_case(*range(13, 24)):
...                 hr_str = "evening"
...             else:
...                 raise RuntimeError("That's not a real time.")
...
...         day_str = "workday"
...         if day_case(0, 1, 2, 3, 4):
...             pass  # Example of fall thru.
...         if day_case(4):
...             day_str += " (and it's Friday, woohoo!)"
...         if day_case(5, 6):
...             day_str = "weekend"
...             Switch.break_switch()  # Breaks outer Switch.
...         if not day_case(*range(7)):  # Example of 'not' case.
...             raise RuntimeError("That's not a real day.")
...
...     print(f"It is {hr_str} on a {day_str}.")

Some trial values passed to the function:

>>> print_day_time(0, 7)  # Monday 07:00.
It is morning on a workday.
>>> print_day_time(2, 12)  # Wednesday 12:00.
It is midday on a workday.
>>> print_day_time(4, 19)  # Friday 19:00.
It is evening on a workday (and it's Friday, woohoo!).
>>> print_day_time(5, 9)  # Saturday 09:00
It is morning on a weekend.
>>> print_day_time(6, 24)  # Sunday, invalid time. Exception passes thru.
Traceback (most recent call last):
...
  File "<doctest Switch[0]>", line 13, in print_day_time
    raise RuntimeError("That's not a real time.")
RuntimeError: That's not a real time.
"""

# Last updated: 27 December 2019 by Eric J. Whitney.

__all__ = ['Switch']


class Switch:
    """
    The Switch class allows the writing of a ``switch ... case`` statement
    in Python
    """
    class _Break(Exception):
        """Internal use class used to exit Switch contexts early."""

    def __init__(self, value):
        """
        Establish a Switch context using the given `value`.
        Parameters
        ----------
        value : Any
            `Value` is used for later comparison in case statements.
        """
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == Switch._Break:
            return True
        # Otherwise return None; allow traceback.

    def __call__(self, *args):
        """
        Calling the object with one or more values serves as a `case`
        statement in the Switch context.

        Parameters
        ----------
        args : Any
            One or more arguments to compare with the Switch value.

        Returns
        -------
        bool :
            True / False result of ``self.value in *args``.
        """
        return self.value in args

    @classmethod
    def break_switch(cls):
        """Breaks out of the current Switch context."""
        raise cls._Break()
