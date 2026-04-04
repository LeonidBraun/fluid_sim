from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _strip_line_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaping = False

    index = 0
    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            if index < len(text):
                result.append(text[index])
                index += 1
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaping = False

    index = 0
    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        result.append(char)
        index += 1

    return "".join(result)


def read_relaxed_json(path: str | Path) -> Any:
    text = _read_text_file(Path(path))
    return json.loads(_strip_trailing_commas(_strip_line_comments(text)))


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")
    return destination


__all__ = ["read_relaxed_json", "write_json"]
