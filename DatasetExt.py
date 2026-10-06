"""Compatibility import for :mod:`datasift.datasets`."""

import sys
from datasift import datasets as _implementation

sys.modules[__name__] = _implementation
