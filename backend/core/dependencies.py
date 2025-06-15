"""Safe dependency imports with version checking."""

import importlib
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def safe_import(module: str, min_version: Optional[str] = None) -> Optional[Any]:
    """Safely import module with optional version check."""
    try:
        mod = importlib.import_module(module)
        if min_version:
            version = getattr(mod, "__version__", "0.0.0")
            if version < min_version:
                logger.warning("%s version %s < %s", module, version, min_version)
        return mod
    except ImportError:
        logger.warning("Module %s not available", module)
        return None
