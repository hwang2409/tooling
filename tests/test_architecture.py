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


def _dynamic_import_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"}:
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
        function_name = node.func.attr
    else:
        return None
    if (
        function_name
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def find_private_mitm_imports(source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
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
            dynamic_name = _dynamic_import_name(node)
            imported_names = [dynamic_name] if dynamic_name else []
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
    ],
)
def test_private_api_bypass_forms_are_rejected(source: str) -> None:
    assert find_private_mitm_imports(source)


def test_documented_public_import_remains_allowed() -> None:
    assert find_private_mitm_imports("from mitmproxy import http") == []
