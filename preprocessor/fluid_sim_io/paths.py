from __future__ import annotations

from pathlib import Path, PureWindowsPath


def is_absolute_reference(path: str | Path) -> bool:
    text = str(path)
    return Path(text).is_absolute() or PureWindowsPath(text).is_absolute()


def resolve_sibling_path(base_path: str | Path, relative_or_absolute: str | Path) -> Path:
    reference = Path(relative_or_absolute)
    if is_absolute_reference(reference):
        return reference
    return (Path(base_path).parent / reference).resolve()


__all__ = ["is_absolute_reference", "resolve_sibling_path"]
