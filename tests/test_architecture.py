import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
BANNED_PREFIXES = (
    "mitmproxy.tools.web",
    "mitmproxy.addons.view",
    "mitmproxy.proxy.layers",
    "mitmweb",
)


def test_source_does_not_import_private_mitmweb_modules() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported = node.module if isinstance(node, ast.ImportFrom) else None
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [imported] if imported else []
            for name in names:
                if name and any(
                    name == banned or name.startswith(f"{banned}.")
                    for banned in BANNED_PREFIXES
                ):
                    violations.append(f"{path}:{node.lineno}: {name}")
    assert violations == []
