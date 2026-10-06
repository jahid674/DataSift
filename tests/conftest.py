"""Load the pre-refactor implementation as an independent regression oracle."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def original():
    names = [
        "utils",
        "metrics",
        "DatasetExt",
        "Misc",
        "influence",
        "Classifier",
        "Algorithms",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    modules = {}
    try:
        for name in names:
            source = Path(__file__).parent / "reference" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules
