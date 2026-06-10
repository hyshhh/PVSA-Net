# Copyright (c) OpenMMLab. All rights reserved.
"""Optional block-sparse flash attention entry for PVSA.

This module is intentionally a safe placeholder. It does not compile or load
any custom kernel by default, so importing the project keeps the same runtime
requirements as the regular ToppAttention path.
"""

from typing import Any


def is_topp_flash_available() -> bool:
    """Return whether the optional custom kernel is available."""
    return False


def topp_flash_attention(*args: Any, **kwargs: Any):
    """Placeholder for the future custom kernel implementation."""
    raise RuntimeError(
        'topp flash attention kernel is not implemented in this branch. '
        'Keep use_topp_flash=False or provide a custom kernel implementation.')
