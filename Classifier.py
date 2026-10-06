"""Compatibility import for :mod:`datasift.models`."""

import sys
from datasift import models as _implementation

sys.modules[__name__] = _implementation
