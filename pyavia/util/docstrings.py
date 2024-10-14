from collections.abc import Callable
from textwrap import dedent
from typing import TypeVar, Any, Type

# Written by Eric J. Whitney, April 2024.

# The following docstring decorator code is heavily borrowed from
# pandas / util /_decorators.py.  Pandas is distributed under the BSD
# 3-clause licence:

#   BSD 3-Clause License
#
#   Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda
#   Foundry, Inc. and PyData Development Team All rights reserved.
#
#   Copyright (c) 2011-2024, Open source contributors.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions
#   are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.

# ======================================================================

# Define generic type for use in decorator functions.  This allows us to
# preserve the signature of the function it decorates.
_DocT = TypeVar("_DocT", bound=Callable[..., Any] | Type)


def doc(*docstrs: None | str | _DocT, **params: object
        ) -> Callable[[_DocT], _DocT]:
    """
    A decorator to take docstring templates, concatenate them and
    perform string substitution on them.

    Parameters
    ----------
    *docstrs : None, str, or Callable
        The string / function docstring / docstring templates to be
        appended in order, immediately after the wrapped function
        docstring.

    **params: dict[str, str]
        Keyword-value pairs to be substituted in the docstring.

    Notes
    -----
    This decorator adds an attribute "_docstr_parts" to the
    wrapped callable to keep track of the original docstring template
    for future use.  For use as a template, it is saved as a string.
    Otherwise it is saved as callable, and later accessed via `__doc__`
    and dedent.

    Examples
    --------
    First we create a parent class method with a normal docstring that
    includes a placeholder.  Then we make two derived classes where we
    substitute the class name in the docstring.

    >>> class Parent:
    ...     @doc(class_="Parent")
    ...     def my_function(self):
    ...         \"""Apply my function to {class_}.\"""
    ...         pass

    >>> class ChildA(Parent):
    ...     @doc(Parent.my_function, class_="ChildA")
    ...     def my_function(self):
    ...         pass

    >>> class ChildB(Parent):
    ...     @doc(Parent.my_function, class_="ChildB")
    ...     def my_function(self):
    ...         pass

    Resulting docstrings:

    >>> print(Parent.my_function.__doc__)
    Apply my function to Parent.
    >>> print(ChildA.my_function.__doc__)
    Apply my function to ChildA.
    >>> print(ChildB.my_function.__doc__)
    Apply my function to ChildB.

    This can also be applied to class-level docstrings:

    >>> @doc(greeting="Hello!")
    ... class ParentClass:
    ...     \"""This is the parent class docstring: {greeting}
    ...     \"""
    ...     pass

    >>> @doc(ParentClass, "Added part unique to the child class.",
    ...      greeting="G'day!")
    ... class ChildClass:
    ...     pass

     Resulting docstrings:

    >>> print(ParentClass.__doc__)  # doctest: +NORMALIZE_WHITESPACE
    This is the parent class docstring: Hello!
    >>> print(ChildClass.__doc__)
    This is the parent class docstring: G'day!
    Added part unique to the child class.
    """

    def decorator(decorated: _DocT) -> _DocT:
        # For the given function or class, assemble the docstring
        # parts and the overall string.

        docstr_parts: list[str | Callable] = []
        if decorated.__doc__:
            docstr_parts.append(dedent(decorated.__doc__))

        for docstr in docstrs:
            if docstr is None:
                continue

            if hasattr(docstr, "_docstr_parts"):
                # noinspection PyProtectedMember
                docstr_parts.extend(docstr._docstr_parts)

            elif isinstance(docstr, str) or docstr.__doc__:
                docstr_parts.append(docstr)

        params_applied = [
            part.format(**params)
            if isinstance(part, str) and len(params) > 0 else part
            for part in docstr_parts]

        decorated.__doc__ = "".join([
            part
            if isinstance(part, str) else dedent(part.__doc__ or "")
            for part in params_applied])

        decorated._docstr_parts = docstr_parts
        return decorated

    return decorator

# ----------------------------------------------------------------------
