import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
APPROVED_PUBLIC_IMPORTS = frozenset({"mitmproxy.http"})
EXCLUDED_PARTS = frozenset({".git", ".venv", "node_modules", "dist", "__pycache__"})


def project_python_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.suffix in {".py", ".pyi", ".pyw"}
        and not EXCLUDED_PARTS.intersection(path.parts)
    )


def _loader_aliases(tree: ast.AST) -> set[str]:
    aliases = {"__import__", "import_module"}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        name = alias.asname or alias.name
                        if name not in aliases:
                            aliases.add(name)
                            changed = True
            if isinstance(node, ast.ImportFrom) and node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        name = alias.asname or alias.name
                        if name not in aliases:
                            aliases.add(name)
                            changed = True
            if isinstance(node, ast.Assign | ast.AnnAssign):
                value = node.value
                is_loader = isinstance(value, ast.Name) and value.id in aliases
                is_loader = is_loader or (
                    isinstance(value, ast.Attribute)
                    and value.attr in {"import_module", "__import__"}
                )
                if is_loader:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
    return aliases


def _dynamic_import_name(node: ast.Call, loader_aliases: set[str]) -> tuple[bool, str | None]:
    if isinstance(node.func, ast.Name) and node.func.id in loader_aliases:
        pass
    elif (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"import_module", "__import__"}
    ):
        pass
    else:
        return False, None
    argument = node.args[0] if node.args else next(
        (keyword.value for keyword in node.keywords if keyword.arg == "name"),
        None,
    )
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return True, argument.value
    return True, None


def find_private_mitm_imports(source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    loader_aliases = _loader_aliases(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module == "mitmproxy":
            imported_names = [f"{node.module}.{alias.name}" for alias in node.names]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("mitmproxy.")
        ):
            imported_names = [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Call):
            is_loader, dynamic_name = _dynamic_import_name(node, loader_aliases)
            imported_names = []
            if is_loader:
                if dynamic_name is None:
                    violations.append(f"{filename}:{node.lineno}: unresolved dynamic import")
                elif dynamic_name not in APPROVED_PUBLIC_IMPORTS:
                    violations.append(f"{filename}:{node.lineno}: {dynamic_name}")
        else:
            imported_names = []
        for imported_name in imported_names:
            if (
                imported_name.startswith("mitmproxy")
                and imported_name not in APPROVED_PUBLIC_IMPORTS
            ):
                violations.append(f"{filename}:{node.lineno}: {imported_name}")
    return violations


def test_project_uses_only_approved_public_mitmproxy_imports() -> None:
    violations = [
        violation
        for path in project_python_files()
        for violation in find_private_mitm_imports(path.read_text(), str(path))
    ]
    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "from mitmproxy.tools import web",
        "from mitmproxy.tools import web as public_web",
        "import mitmproxy.proxy.layers as layers",
        "import importlib; importlib.import_module('mitmproxy.tools.web')",
        "from importlib import import_module; import_module('mitmproxy.addons.view')",
        "__import__('mitmproxy.proxy.layers')",
        "from importlib import import_module as load; load('mitmproxy.tools.web')",
        "loader = __import__; loader('mitmproxy.addons.view')",
    ],
)
def test_private_api_bypass_forms_are_rejected(source: str) -> None:
    assert find_private_mitm_imports(source)


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module as load; load(name)",
        "import builtins; loader = builtins.__import__; loader(module_name)",
        "from builtins import __import__ as loader; loader(name=module_name)",
    ],
)
def test_unresolved_dynamic_loader_calls_fail_closed(source: str) -> None:
    assert "unresolved dynamic import" in find_private_mitm_imports(source)[0]


def test_dynamic_loader_rejects_literals_outside_the_public_allow_list() -> None:
    source = "from importlib import import_module as load; load('json')"
    assert find_private_mitm_imports(source)


@pytest.mark.parametrize(
    "source",
    [
        "from mitmproxy import http",
        "from importlib import import_module as load; load('mitmproxy.http')",
        "import builtins; loader = builtins.__import__; loader('mitmproxy.http')",
    ],
)
def test_documented_public_import_remains_allowed(source: str) -> None:
    assert find_private_mitm_imports(source) == []
