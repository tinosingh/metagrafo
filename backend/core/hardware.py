"""Hardware detection utilities."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """Determine optimal compute device with memory awareness."""
    if torch.cuda.is_available():
        try:
            mem_alloc = torch.cuda.memory_allocated()
            mem_total = torch.cuda.get_device_properties(0).total_memory
            if mem_alloc < 0.8 * mem_total:
                return "cuda"
        except RuntimeError as e:
            logger.warning("CUDA check failed: %s", str(e))
    return "cpu"
