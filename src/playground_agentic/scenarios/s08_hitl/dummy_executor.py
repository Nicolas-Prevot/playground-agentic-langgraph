"""SQLite-backed simulation: the database row IS the effect, never an external refund."""

import sqlite3
from pathlib import Path

from playground_agentic.scenarios.s08_hitl.graph import PreparedAction


class SqliteDummyExecutor:
    """Atomic deduplication is possible here because receipt and fake effect share one DB."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS effects "
                "(key TEXT PRIMARY KEY, action TEXT NOT NULL, receipt TEXT NOT NULL, "
                "compensated INTEGER NOT NULL DEFAULT 0)"
            )

    def execute(self, key: str, action: PreparedAction) -> str:
        payload = action.model_dump_json()
        receipt = f"simulated:{action.action}:{key}"
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT action, receipt, compensated FROM effects WHERE key = ?", (key,)
            ).fetchone()
            if previous:
                if previous[0] != payload or previous[2]:
                    raise ValueError("Key already belongs to a different or compensated action")
                return previous[1]
            connection.execute(
                "INSERT INTO effects (key, action, receipt) VALUES (?, ?, ?)",
                (key, payload, receipt),
            )
        return receipt

    def compensate(self, key: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE effects SET compensated = 1 WHERE key = ?", (key,))

    def count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return connection.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
