"""Compatibility import for :mod:`datasift.algorithms`."""

import sys
from datasift import algorithms as _implementation

sys.modules[__name__] = _implementation
