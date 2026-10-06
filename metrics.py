"""Compatibility import for :mod:`datasift.metrics`."""

import sys
from datasift import metrics as _implementation

sys.modules[__name__] = _implementation
