import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_ROOT = ROOT / "src"
APPROVED_PUBLIC_IMPORTS = frozenset({"mitmproxy.http"})
FORBIDDEN_DYNAMIC_ROOTS = frozenset({"builtins", "importlib", "pkgutil", "pydoc", "runpy"})
FORBIDDEN_DYNAMIC_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "__loader__",
        "builtins",
        "import_module",
        "importlib",
        "load_module",
        "locate",
        "pkgutil",
        "pydoc",
        "resolve_name",
        "run_module",
        "runpy",
    }
)
FORBIDDEN_DYNAMIC_ATTRIBUTES = frozenset(
    {"__import__", "import_module", "load_module", "locate", "resolve_name", "run_module"}
)
FORBIDDEN_DYNAMIC_LITERAL_MARKERS = frozenset(
    {
        "__builtins__",
        "__import__",
        "__loader__",
        "builtins",
        "import_module",
        "importlib",
        "load_module",
        "pkgutil",
        "pydoc",
        "runpy",
    }
)
SUSPICIOUS_LOADER_CALL_NAMES = frozenset(
    {"dynamic_import", "dynamic_loader", "importer", "load", "loader"}
)
PRIVATE_MODULE_PREFIXES = (
    "mitmproxy.addons.view",
    "mitmproxy.proxy.layers",
    "mitmproxy.tools.web",
    "mitmweb",
)


def project_python_files() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*")
        if path.suffix in {".py", ".pyi", ".pyw"}
    )


def _is_approved_public_import(imported_name: str) -> bool:
    return any(
        imported_name == approved or imported_name.startswith(f"{approved}.")
        for approved in APPROVED_PUBLIC_IMPORTS
    )


def _is_private_module_literal(value: str) -> bool:
    return any(prefix in value for prefix in PRIVATE_MODULE_PREFIXES)


def _is_dynamic_import_literal(value: str) -> bool:
    return any(marker in value for marker in FORBIDDEN_DYNAMIC_LITERAL_MARKERS)


def find_import_boundary_violations(
    source: str,
    filename: str = "<source>",
) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    def reject(node: ast.AST, detail: str) -> None:
        violations.append(f"{filename}:{getattr(node, 'lineno', 0)}: {detail}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.name
                if imported_name.split(".", 1)[0] in FORBIDDEN_DYNAMIC_ROOTS:
                    reject(node, f"dynamic import machinery is forbidden: {imported_name}")
                elif imported_name.startswith("mitmproxy"):
                    if not _is_approved_public_import(imported_name):
                        reject(node, f"unapproved mitmproxy import: {imported_name}")
                elif imported_name == "mitmweb" or imported_name.startswith("mitmweb."):
                    reject(node, f"private mitmweb import: {imported_name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module.split(".", 1)[0] in FORBIDDEN_DYNAMIC_ROOTS:
                reject(node, f"dynamic import machinery is forbidden: {module}")
            elif module == "mitmproxy" or module.startswith("mitmproxy."):
                for alias in node.names:
                    imported_name = f"{module}.{alias.name}"
                    if not _is_approved_public_import(imported_name):
                        reject(node, f"unapproved mitmproxy import: {imported_name}")
            elif module == "mitmweb" or module.startswith("mitmweb."):
                reject(node, f"private mitmweb import: {module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DYNAMIC_NAMES:
            reject(node, f"dynamic import name is forbidden: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DYNAMIC_ATTRIBUTES:
            reject(node, f"dynamic import attribute is forbidden: {node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in SUSPICIOUS_LOADER_CALL_NAMES
        ):
            reject(node, f"unresolved dynamic loader call is forbidden: {node.func.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_dynamic_import_literal(node.value):
                reject(node, f"dynamic import literal is forbidden: {node.value}")
            elif _is_private_module_literal(node.value):
                reject(node, f"private module literal is forbidden: {node.value}")
    return violations


def test_project_uses_only_static_approved_public_mitmproxy_imports() -> None:
    violations = [
        violation
        for path in project_python_files()
        for violation in find_import_boundary_violations(path.read_text(), str(path))
    ]
    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "from mitmproxy.tools import web",
        "from mitmproxy.tools import web as public_web",
        "import mitmproxy.proxy.layers as layers",
        "import mitmweb.master",
        "PRIVATE_MODULE = 'mitmproxy.tools.web'",
        "PRIVATE_MODULE = 'mitmweb.state'",
    ],
)
def test_private_api_imports_and_literals_are_rejected(source: str) -> None:
    assert find_import_boundary_violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "import importlib",
        "from importlib import import_module",
        "import importlib; importlib.import_module('mitmproxy.http')",
        "from importlib import import_module as load; load('mitmproxy.http')",
        "__import__('mitmproxy.http')",
        "import_module(module_name)",
        "load(module_name)",
        "loader(module_name)",
        "import builtins; loader = builtins.__import__; loader('mitmproxy.http')",
        "from builtins import __import__ as loader; loader(name=module_name)",
        "(loader := __import__)('mitmproxy.http')",
        "loader, other = __import__, print; loader('mitmproxy.http')",
        "import builtins; getattr(builtins, '__import__')('mitmproxy.http')",
        "__builtins__['__import__']('mitmproxy.http')",
        "import importlib; importlib.__dict__['import_module']('mitmproxy.http')",
        "module = 'importlib'",
        "module = 'builtins'",
        "eval(\"__import__('mitmproxy.http')\")",
        "eval(\"importlib.import_module('mitmproxy.http')\")",
        "import pkgutil; pkgutil.resolve_name(module_name)",
        "from pkgutil import resolve_name; resolve_name(module_name)",
        "import pydoc; pydoc.locate(module_name)",
        "from pydoc import locate; locate(module_name)",
        "import runpy; runpy.run_module(module_name)",
        "from runpy import run_module; run_module(module_name)",
        "__loader__.load_module(module_name)",
        "loader = __loader__; loader.load_module(module_name)",
    ],
)
def test_all_dynamic_import_machinery_is_rejected(source: str) -> None:
    assert find_import_boundary_violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "from mitmproxy import http",
        "import mitmproxy.http",
        "from mitmproxy.http import HTTPFlow",
        "import json",
    ],
)
def test_ordinary_static_public_imports_remain_allowed(source: str) -> None:
    assert find_import_boundary_violations(source) == []
