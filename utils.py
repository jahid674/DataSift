"""Compatibility import for :mod:`datasift.utils`."""

import sys
from datasift import utils as _implementation

sys.modules[__name__] = _implementation
