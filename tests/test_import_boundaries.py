from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOTS = {"urllib", "http", "socket", "ssl", "requests", "httpx"}
FORBIDDEN_GATE_PREFIXES = (
    "matchlab.adapters",
    "matchlab.compiler",
    "matchlab.local_staging",
)
ORACLE_FORBIDDEN_PREFIXES = (
    "matchlab.guards.gate",
    "matchlab.text",
    "matchlab.matching",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_parts = _module_name(path).split(".")
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                if node.module:
                    base.extend(node.module.split("."))
                target = ".".join(base)
            else:
                target = node.module or ""
            if target:
                found.add(target)
            elif node.level:
                found.add(".".join(package_parts[: len(package_parts) - node.level + 1]))
            for alias in node.names:
                if alias.name == "*":
                    continue
                if target:
                    found.add(f"{target}.{alias.name}")
    return found


def _source_for_module(module: str) -> Path | None:
    if not module.startswith("matchlab."):
        return None
    relative = Path(*module.split("."))
    candidates = (ROOT / relative.with_suffix(".py"), ROOT / relative / "__init__.py")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _transitive_imports(paths: list[Path]) -> set[str]:
    visited: set[Path] = set()
    found: set[str] = set()
    pending = list(paths)
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        imported = _imports(path)
        found.update(imported)
        for module in imported:
            source = _source_for_module(module)
            if source is not None and source not in visited:
                pending.append(source)
    return found


def _has_prefix(imports: set[str], prefixes: tuple[str, ...]) -> set[str]:
    return {
        imported
        for imported in imports
        if any(imported == prefix or imported.startswith(prefix + ".") for prefix in prefixes)
    }


def test_gate_path_does_not_import_forbidden_modules_transitively() -> None:
    gate_dir = ROOT / "matchlab" / "guards"
    gate_files = [
        ROOT / "matchlab" / "schemas.py",
        ROOT / "matchlab" / "text.py",
        *sorted(gate_dir.glob("*.py")),
    ]
    imports = _transitive_imports(gate_files)
    forbidden = _has_prefix(imports, FORBIDDEN_GATE_PREFIXES)
    forbidden.update(
        imported
        for imported in imports
        if imported.split(".", 1)[0] in NETWORK_ROOTS
    )
    if forbidden:
        raise AssertionError(f"gate path imports forbidden modules: {sorted(forbidden)}")


def test_oracle_does_not_import_gate_or_matcher_transitively() -> None:
    oracle = ROOT / "matchlab" / "evals" / "oracle.py"
    if not oracle.exists():
        raise AssertionError("oracle.py does not exist")
    forbidden = _has_prefix(_transitive_imports([oracle]), ORACLE_FORBIDDEN_PREFIXES)
    if forbidden:
        raise AssertionError(f"oracle imports gate or matcher modules: {sorted(forbidden)}")


if __name__ == "__main__":
    test_gate_path_does_not_import_forbidden_modules_transitively()
    test_oracle_does_not_import_gate_or_matcher_transitively()
