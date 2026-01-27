from __future__ import annotations

from typing import Callable, Dict, Type

from .base import TermBase

_REGISTRY: Dict[str, Type[TermBase]] = {}
_ALIASES: Dict[str, str] = {
    # IsaacLab obs term name 'actions' corresponds to our last_action term
    "actions": "last_action",
}


def register(func_name: str, *, aliases: list[str] | None = None) -> Callable[[Type[TermBase]], Type[TermBase]]:
    """Decorator to register a term class by its function name."""

    def deco(cls: Type[TermBase]) -> Type[TermBase]:
        _REGISTRY[func_name] = cls
        if aliases:
            for a in aliases:
                _ALIASES[a] = func_name
        return cls

    return deco


def get_term_class(func_name: str) -> Type[TermBase]:
    key = _ALIASES.get(func_name, func_name)
    if key not in _REGISTRY:
        raise KeyError(f"No term registered for func '{func_name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]