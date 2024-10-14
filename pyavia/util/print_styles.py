from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np


# Written by Eric J. Whitney, February 2024.


# ======================================================================


class FormatStyle:
    """
    `FormatStyle` defines a base class for discrete formatting
    operations that can be applied to a string. The style may optionally
    incoporate features of a linked parent style.

    Derived classes provide their own implementation of the `apply(s)`
    method to accomplish the formatting.

    Parameters
    ----------
    parent : FormatStyle, Optional
        The parent style, if any.
    """

    def __init__(self, parent: FormatStyle = None):
        self.parent = parent

    # -- Public Methods ------------------------------------------------

    def apply(self, s: str) -> str:
        """
        Format or modify `s`, as well as applying any formatting from
        parent styles if desired.

        .. note::Derived classes should override this method.

        Returns
        -------
        str
            The formatted string. At the base level this returns `s`
            unchanged.
        """
        return s

    @property
    def level(self) -> int:
        """
        Returns the level of this `FormatStyle` object in the tree
        of styles.  The top (root) style has a level of 1.
        """
        if self.parent:
            return self.parent.level + 1
        else:
            return 1


# ----------------------------------------------------------------------

class PrintStyles:
    """
    `PrintStyles` holds an organised collection of `FormatStyle` objects
    that may include parent / child relationships.  These can be used to
    apply different styles when printing output in different contexts.

    Parameters
    ----------
    display_level : int, default = 10
        Lowest number style level to display.  The highest level is
        **1**. Setting ``display_level=0`` will suppress output.

    Examples
    --------
    As a first example, we setup `AddTabStyle` formatting at every
    level.  These chain together with parent styles to increase the
    level of indent:
    >>> fmt = PrintStyles(display_level=3)
    >>> fmt.add('indent0', AddTabStyle())  # Level 1.
    >>> fmt.add('indent1', AddTabStyle(), parent='indent0')  # Level 2.
    >>> fmt.add('indent2', AddTabStyle(), parent='indent1')  # Level 3.
    >>> fmt.add('indent3', AddTabStyle(), parent='indent2')  # Level 4.

    The `print` function can then be called with a specified level to
    apply the formatting:
    >>> fmt.print('indent0', "Hello world! Note no indentation at this level.")
    Hello world! Note no indentation at this level.

    >>> fmt.print('indent1', "This text is indented.")
        This text is indented.

    >>> fmt.print('indent2', "This text is indented twice.")
            This text is indented twice.

    >>> fmt.print('indent3', "This text is indented thrice, but won't "
    ...                      "appear due to the 'display_level' setting.")
    >>> fmt.print('indent1', "Dropping back to first indent.")
        Dropping back to first indent.

    >>> fmt.print('indent3', "Temporarily override 'display_level' to "
    ...                      "show this text.", display_level=4)
                Temporarily override 'display_level' to show this text.

    In the following example we don't chain parent styles, but include
    the level number in the format string instead.  First we define the
    format style:
    >>> class HeadingFormat(FormatStyle):
    ...     def apply(self, s: str) -> str:
    ...         return f"[Heading {self.level}] " + s

    These can be added to the existing `PrintStyles` collection:
    >>> fmt.display_level = 5
    >>> fmt.add('heading1', HeadingFormat())
    >>> fmt.add('heading2', HeadingFormat(), parent='heading1')
    >>> fmt.add('heading3', HeadingFormat(), parent='heading2')

    These then give the following printed output.

    >>> fmt.print('heading1', "This text uses 'heading1'.")
    [Heading 1] This text uses 'heading1'.

    >>> fmt.print('heading2', "This text uses 'heading2'.")
    [Heading 2] This text uses 'heading2'.

    >>> fmt.print('heading3', "This text uses 'heading3'.")
    [Heading 3] This text uses 'heading3'.

    A new style can be added at the top of the hierarchy:
    >>> fmt.add('heading0', HeadingFormat(), children='heading1')
    >>> fmt.print('heading0', "Added style 'heading0'.")
    [Heading 1] Added style 'heading0'.

    >>> fmt.print('heading1', "This text uses 'heading1'.")
    [Heading 2] This text uses 'heading1'.

    A new style can be inserted in the middle of the hierarchy:
    >>> fmt.add('heading1.5', HeadingFormat(), parent='heading1',
    ...         children='heading2')
    >>> fmt.print('heading0', "This text uses 'heading0'.")
    [Heading 1] This text uses 'heading0'.

    >>> fmt.print('heading1', "This text uses 'heading1'.")
    [Heading 2] This text uses 'heading1'.

    >>> fmt.print('heading1.5', "Inserted style 'heading1.5'.")
    [Heading 3] Inserted style 'heading1.5'.

    >>> fmt.print('heading2', "This text uses 'heading2'.")
    [Heading 4] This text uses 'heading2'.

    >>> fmt.print('heading3', "This text uses 'heading3'.")
    [Heading 5] This text uses 'heading3'.
    """

    def __init__(self, display_level: int = 10):
        # Future work: display_levels could be a sequence of discrete
        # levels to display.

        self.display_level = display_level
        self._styles: dict[str, FormatStyle] = {}  # Flat file.

    # -- Public Methods ------------------------------------------------

    def add(self, name: str, style: FormatStyle,
            parent: str = None, children: str | Sequence[str] = None):
        """
        Adds a new style to the collection of formatting styles.

        Parameters
        ----------
        name : str
            Name of added style.
        style : FormatStyle
            Style object.
        parent : str, Optional
            Name of parent style (to insert below).

            .. note::If the object `style` holds an existing reference
               to a parent `FormatStyle` (i.e. `style.parent` is not
               `None`), this will be overwritten.

        children : str or Sequence[str], Optional
            Name of one or more child style/s (to insert above).

            .. note::If the child object already references another
               parent `FormatStyle` this will be changed to reference
               the object `style`.

        Raises
        ------
        ValueError
            If `name` already exists or `parent` does not exist.

        Notes
        -----
        At present there are no strong protections against circular
        references in the style hierarchy.
        """
        if name in self._styles:
            raise ValueError(f"Print style '{name}' already defined.")

        # -- Setup Parent Linkage --------------------------------------

        if parent:
            try:
                parent_ref = self._styles[parent]
            except KeyError:
                raise ValueError(f"Parent print style '{parent}' not found.")

            style.parent = parent_ref

        else:
            style.parent = None

        # -- Setup Child Linkage ---------------------------------------

        if children:  # Not 'None', blank ('') or empty sequence.

            if isinstance(children, str):
                children = [children]

            for child in children:
                if parent == child:
                    raise ValueError(f"Child and parent print styles "
                                     f"cannot be the same: '{parent}'")
                # Future work: No strong protection against circular
                # references in the hierarchy.

                try:
                    child_ref = self._styles[child]
                except KeyError:
                    raise ValueError(f"Child print style '{child}' "
                                     f"not found.")

                child_ref.parent = style

        # -- Finalise --------------------------------------------------

        self._styles[name] = style

    def apply(self, name: str, s: str) -> str:
        """
        Apply format style `name` to string `s`, including any parent
        styles (where applicable).

        Parameters
        ----------
        name : str
            Name of format style to apply.
        s : str
            The string to format.

        Returns
        -------
        str
            The formatted string.

        Raises
        ------
        ValueError
            If `name` is not found.
        """
        try:
            style = self._styles[name]
        except KeyError:
            raise ValueError(f"Style '{name}' not found.")

        return style.apply(s)

    def get_level(self, level: int) -> dict[str, FormatStyle]:
        """
        Return all styles that are located at `level` in the hierarchy.

        Parameters
        ----------
        level : int
            Return styles of this level.

        Returns
        -------
        dict[str, FormatStyle]
            Dict containing names and FormatStyle objects corresponding
            to `level`.
        """
        return {k: v for k, v in self._styles.items()
                if v.level == level}

    def print(self, name: str | None, s: str = '', *args,
              display_level: int = None, **kwargs):
        """
        Print string `s` after applying formatting, if style `name` is
        at or above the `display_level`.

        Parameters
        ----------
        name : str
            Name of format style to apply, or `None` to bypass
            formatting.

            .. note::If `name` is not found, a warning is generated
               and `s` is printed without formatting.

        s : str, default = ''
            String to format.

        display_level : int
            If supplied, sets the `display_level` parameter for this
            print operation only.

        *args, **kwargs :
            Remaining positional and keyword arguments passed directly
            to `print` after `s`.
        """
        if name is None:
            print(s, *args, **kwargs)
            return

        try:
            style = self._styles[name]
        except KeyError:
            print(s, *args, **kwargs)
            warnings.warn(f"Format style '{name}' not found.")
            return

        if display_level is None:
            display_level = self.display_level

        if style.level <= display_level:
            s_fmt = style.apply(s)
            print(s_fmt, *args, **kwargs)


# ----------------------------------------------------------------------

class PrintStylesMixin:
    """
    Mixin class that adds a `PrintStyles` object and methods to a
    class.

    .. note::This mixin should appear first (leftmost) in the list of
       parent classes.

    Parameters
    ----------
    display_level : int, default = 10
        Lowest number style level to display.  The highest level is
        **1**. Setting ``display_level=0`` will suppress output.
    """

    def __init__(self, *args, display_level: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pstyles = PrintStyles(display_level=display_level)

    # -- Public Methods ------------------------------------------------

    @property
    def display_level(self):
        return self.__pstyles.display_level

    @display_level.setter
    def display_level(self, level: int):
        self.__pstyles.display_level = level

    @property
    def pstyles(self) -> PrintStyles:
        return self.__pstyles


# ======================================================================

# Standard format styles.

class AddTabStyle(FormatStyle):
    """
    If the style has a parent, prepends four spaces to the string
    before applying the parent style.
    """

    def apply(self, s: str) -> str:
        if self.parent:
            return self.parent.apply('    ' + s)
        else:
            return s


class AddDotStyle(FormatStyle):
    """
    If the style has a parent, prepends three dots and a space
    (``... ``) to the string before applying the parent style.
    """

    def apply(self, s: str) -> str:
        if self.parent:
            return self.parent.apply('... ' + s)
        else:
            return s


class SkipStyle(FormatStyle):
    """Applies the parent style, but adds no extra formatting."""

    def apply(self, s: str) -> str:
        if self.parent:
            s = self.parent.apply(s)
        return s


class LevelDotStyle(FormatStyle):
    """
    Outputs four dots (``....``) for each output level except the last,
    which outputs three dots and a space (``... ``).
    """

    def apply(self, s: str) -> str:
        if self.level > 1:
            return '....' * (self.level - 1) + '... ' + s
        else:
            return s


class LevelTabStyle(FormatStyle):
    r"""Outputs ('level' - 1) sets of four spaces before the string."""

    def apply(self, s: str) -> str:
        return '    ' * (self.level - 1) + s


# ======================================================================

def ruled_line(s: str, above: str | None = '-', below: str | None = '-',
               *, max_length: int | None = 72, min_length: int | None = 72,
               end: str | None = None) -> str:
    """
    Returns a string containing `s` with a optional ruled lines above
    and/or below.  The length of the ruled lines is the same as `s`
    unless modified by  `min_length` or `max_length`.

    Parameters
    ----------
    s : str
        String to print above and below the line.

    above, below : str, default = '-'
        Character/s to repeat to form the ruled line.  Multiple
        characters can be used to form a pattern.  If `None` or blank,
        the line is omitted.

    min_length, max_length : int, default = 72
        Minimum and maximum length of the ruled line.

        .. note::If `above` or `below` use two or more characters,
           some of the end characters may be truncated to honour the
           `min_length` and `max_length` requirements.


    end : str, default = ''
        Character to append to the end of the ruled line below `s`
        (or `s` if absent).  The default option of ``end=''`` is
        designed to operate with builtin `print` which appends a newline
        by default.

    Returns
    -------
    str
        Formatted string.

    Examples
    --------
    >>> print(ruled_line('HEADING', max_length=60))
    ------------------------------------------------------------
    HEADING
    ------------------------------------------------------------

    >>> print(ruled_line('ANOTHER HEADING', above='ABCD.', below='-+',
    ...                  min_length=11, max_length=13))
    ABCD.ABCD.ABC
    ANOTHER HEADING
    -+-+-+-+-+-+-
    """
    length = len(s)
    if min_length is not None:
        length = max(length, min_length)

    # Function to generate (possibly overlong) ruled lines of repeating
    # patterns then trim to a maximum length.
    def _make_ruled_line(pattern: str) -> str:
        n_repeat = math.ceil(length / len(pattern))
        line_str = pattern * n_repeat
        if max_length is not None:
            line_str = line_str[:max_length]

        return line_str

    result = s
    if above:
        result = _make_ruled_line(above) + '\n' + result

    if below:
        result = result + '\n' + _make_ruled_line(below)

    if end is not None:
        result += end

    return result


# ======================================================================

def rad2str(x: float, *, dp: int = 1, signed: bool = True) -> str:
    """
    Engineering-style fixed width string conversion for angles in
    radians that also converts to degrees, adding **°** symbol.
    """
    sgn = '+' if signed else ''
    return f"{np.rad2deg(x):{sgn}{dp + 5}.0{dp}f}°"


def val2str(x: float, *, dp: int = 1, signed: bool = True,
            commas: int = 1) -> str:
    """
    Engineering-style fixed width string conversion for (possibly
    large) floating point values.  Set `commas` > 0 to add comma
    thousands separator and give anticipated size of number (i.e.
    `x` > 1,000, 000 would require ``commas=2``)
    """
    sgn = ''
    width = 5 + dp + commas
    if signed:
        sgn = '+'
        width += 1
    comma = ',' if commas > 0 else ''
    return f"{x:{sgn}{width}{comma}.0{dp}f}"


def _lgsgn2str(x: float) -> str:
    """Pretty string generation for larger signed numbers."""
    return f"{x:+,.01f}"
