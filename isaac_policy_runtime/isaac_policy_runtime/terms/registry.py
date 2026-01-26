from __future__ import annotations

from typing import Any, Dict, Type

# func_name -> Term class
_REGISTRY: Dict[str, Type[Any]] = {}

# IsaacLab / export 名と runtime 名のズレを吸収する
_ALIAS: Dict[str, str] = {
    # IsaacLab observation name
    "actions": "last_action",
}


def register(func_name: str):
    """Decorator to register a Term class under a given function name."""

    def _decorator(cls: Type[Any]) -> Type[Any]:
        if func_name in _REGISTRY:
            raise KeyError(
                f"Term already registered for func '{func_name}' -> {_REGISTRY[func_name]}"
            )
        _REGISTRY[func_name] = cls
        return cls

    return _decorator


def get_term_class(func_name: str) -> Type[Any]:
    resolved = _ALIAS.get(func_name, func_name)
    if resolved not in _REGISTRY:
        raise KeyError(
            f"No term registered for func '{func_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[resolved]
