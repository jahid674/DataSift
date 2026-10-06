"""Compatibility import for :mod:`datasift.influence`."""

import sys
from datasift import influence as _implementation

sys.modules[__name__] = _implementation
