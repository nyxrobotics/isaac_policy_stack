from __future__ import annotations

from typing import Any, Dict, Type

from .base import Term

_REGISTRY: Dict[str, Type[Term]] = {}

def register(func_name: str):
    def deco(cls: Type[Term]):
        _REGISTRY[func_name] = cls
        return cls
    return deco

def get_term_class(func_name: str) -> Type[Term]:
    if func_name in _REGISTRY:
        return _REGISTRY[func_name]
    # Fallback: some io_descriptors may store term name instead of function name.
    if func_name.lower() in _REGISTRY:
        return _REGISTRY[func_name.lower()]
    raise KeyError(f"No term registered for func '{func_name}'. Available: {sorted(_REGISTRY.keys())}")
