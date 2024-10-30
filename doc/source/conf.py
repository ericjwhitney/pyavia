# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from pyavia import __version__

# -- Path setup --------------------------------------------------------

# sys.path.insert(0, os.path.abspath('..'))  # Designed to be run from doc/ (?)
# sys.path.insert(0, os.path.abspath('../../'))  # Still required (?)

# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('../..'))  # Source rel. to conf.py.
# sys.path.insert(0, os.path.abspath('../pyavia'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyAvia'
copyright = '2024, Eric J. Whitney'  # noqa
author = 'Eric J. Whitney'
version = __version__  # Short X.Y version.
release = version  # Full version, including alpha/beta/rc tags.

# -- General configuration ---------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']
templates_path = ['_templates']
exclude_patterns = []

# TODO temp disable
# autodoc_default_options = {
#     'members': True,  # No more Sphinx Bug?
#     'special-members': True,  # "" "" Ditto?
#     'exclude-members': '__abstractmethods__, __dict__, __hash__, '
#                        '__module__, __slots__, __weakref__'}

# -- Options for HTML output -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'classic'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for Python Domain -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain

# add_module_names = False  # Use short names for modules. (???)
