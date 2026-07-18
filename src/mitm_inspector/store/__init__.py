"""Bounded memory and durable local retention boundaries."""

from typing import Any

__all__ = ["MemoryStore", "SQLiteFlowStorage", "default_storage_path"]


def __getattr__(name: str) -> Any:
    if name == "MemoryStore":
        from mitm_inspector.store.memory import MemoryStore

        return MemoryStore
    if name in {"SQLiteFlowStorage", "default_storage_path"}:
        from mitm_inspector.store.sqlite import SQLiteFlowStorage, default_storage_path

        values = {
            "SQLiteFlowStorage": SQLiteFlowStorage,
            "default_storage_path": default_storage_path,
        }
        return values[name]
    raise AttributeError(name)
