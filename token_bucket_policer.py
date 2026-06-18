"""Compatibility wrapper for `token-bucket-policer.py`.

This lets test code import `token_bucket_policer` even though the main
implementation file uses a hyphen in its filename.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_IMPL_PATH = Path(__file__).with_name("token-bucket-policer.py")
_SPEC = importlib.util.spec_from_file_location("token_bucket_policer_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load implementation module from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Params = _MODULE.Params
Result = _MODULE.Result
TokenBucketSim = _MODULE.TokenBucketSim
sweep = _MODULE.sweep
parse_args = _MODULE.parse_args

__all__ = ["Params", "Result", "TokenBucketSim", "sweep", "parse_args"]
