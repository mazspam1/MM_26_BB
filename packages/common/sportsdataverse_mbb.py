"""
Load sportsdataverse.mbb without importing the top-level package.

The sportsdataverse __init__ imports CFB modules that load an XGBoost model
and can fail on newer xgboost versions. This helper bypasses __init__ so
we can still use the MBB endpoints.
"""

from __future__ import annotations

from importlib import import_module, util
import sys
import types

import structlog

logger = structlog.get_logger()


def load_mbb():
    """Return the sportsdataverse.mbb module, bypassing package __init__."""
    try:
        return import_module("sportsdataverse.mbb")
    except Exception as exc:
        logger.warning(
            "sportsdataverse import failed; attempting mbb-only import",
            error=str(exc),
        )
        sys.modules.pop("sportsdataverse", None)
        spec = util.find_spec("sportsdataverse")
        if spec is None or spec.submodule_search_locations is None:
            raise

        stub = types.ModuleType("sportsdataverse")
        stub.__path__ = list(spec.submodule_search_locations)
        sys.modules["sportsdataverse"] = stub

        return import_module("sportsdataverse.mbb")
