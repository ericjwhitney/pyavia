# PyAvia

**PyAvia** is a collection of modules useful for common tasks in aerospace
engineering or engineering in general.  

The current version is **0.0.4** (development).

*Important points*:
- Development is intended to be organic.
- Modules presently included were either previously developed during my day-to-day 
  work as an engineer, or otherwise just out of curiosity.  If something
  seems like it might be useful to other people, I will try to include it.
- I don't intend on including anything that is particularly arcane, buggy,
  hard to use or not for the public domain.

## Warning

**CAVEAT COMPUTOR** - These modules are provided "as is", without warranty
of any kind.  They are intended to be read and/or used by people trained
in engineering and scientific methods who know how to verify results and
who can recognise incorrect values when they see them... *which will
happen frequently.*

## Installation
**PyAvia** is designed for Python >= 3.9 and is platform-agnostic.

NumPy may need to be installed prior to installing PyAvia (*because I may
 not have setuptools completely on my side*), 
 **[see this link](https://docs.scipy.org/doc/numpy/user/install.html)**.

Once NumPy is installed, install PyAvia using the typical pip installation
 methods, for example:
- Using a shell with pip directly and root access:
```console
$ pip install pyavia
```
- Using a windows prompt as a user only: 
```console
C:\> py -m pip install --user pyavia
```

## Documentation [![Documentation Status](https://readthedocs.org/projects/pyavia/badge/?version=latest)](https://pyavia.readthedocs.io/en/latest/?badge=latest)

Documentation can be found at
 **[ReadTheDocs](https://pyavia.readthedocs.io/en/latest/)**.
 
## Source Code 

The source code is available in the
**[Github repository](https://github.com/ericjwhitney/pyavia)**.

## License [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

PyAvia is provided free of charge for use under the conditions of the 
**[MIT license](http://opensource.org/licenses/mit-license.php)**.

See the 
**[license file](https://github.com/ericjwhitney/pyavia/blob/master/LICENSE)**
for more details.

**Copyright 2022 © Eric J. Whitney**