"""Compatibility import for :mod:`datasift.partitioning`."""

import sys
from datasift import partitioning as _implementation

sys.modules[__name__] = _implementation
