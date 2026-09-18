"""
SqliteStore — 继承 LangGraph BaseStore，用 SQLite 实现长期记忆。
一张 store 表，(namespace, key) 作为主键，value 存 JSON。

只实现 batch（同步）和 abatch（异步委托同步），put/get/search/delete 由 BaseStore 自动提供。
"""

import asyncio
import sqlite3
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore, Item, SearchItem,
    PutOp, GetOp, SearchOp, ListNamespacesOp,
    Op, Result, MatchCondition,
)


class SqliteStore(BaseStore):
    """基于 SQLite 的 LangGraph 长期记忆后端。"""

    def __init__(self, db_path: str = "joker_store.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
        """)
        self._conn.commit()

    # === namespace 序列化 ===

    @staticmethod
    def _pack(ns: tuple[str, ...]) -> str:
        return ".".join(ns)

    @staticmethod
    def _unpack(s: str) -> tuple[str, ...]:
        return tuple(s.split("."))

    # === 核心：batch ===

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # 真异步：把同步 batch 丢进线程池，await 期间事件循环能去干别的
        return await asyncio.to_thread(self.batch, ops)

    # === 各 Op 的 SQL 实现 ===

    def _put(self, op: PutOp) -> None:
        ns = self._pack(op.namespace)
        if op.value is None:
            self._conn.execute(
                "DELETE FROM store WHERE namespace = ? AND key = ?",
                (ns, op.key)
            )
        else:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """INSERT INTO store (namespace, key, value, created_at, updated_at)
                   VALUES (?, ?, ?,
                     COALESCE((SELECT created_at FROM store WHERE namespace=? AND key=?), ?),
                     ?
                   ) ON CONFLICT(namespace, key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = excluded.updated_at""",
                (ns, op.key, json.dumps(op.value, ensure_ascii=False),
                 ns, op.key, now, now)
            )
        self._conn.commit()

    def _get(self, op: GetOp) -> Item | None:
        ns = self._pack(op.namespace)
        row = self._conn.execute(
            "SELECT namespace, key, value, created_at, updated_at FROM store WHERE namespace=? AND key=?",
            (ns, op.key)
        ).fetchone()
        if row is None:
            return None
        return Item(
            namespace=self._unpack(row[0]),
            key=row[1],
            value=json.loads(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = self._pack(op.namespace_prefix)
        if prefix:
            rows = self._conn.execute(
                "SELECT namespace, key, value, created_at, updated_at FROM store "
                "WHERE namespace = ? OR namespace LIKE ? "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (prefix, prefix + ".%", op.limit, op.offset)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT namespace, key, value, created_at, updated_at FROM store "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (op.limit, op.offset)
            ).fetchall()

        items: list[SearchItem] = []
        for row in rows:
            value = json.loads(row[2])
            if op.filter:
                if not all(value.get(k) == v for k, v in op.filter.items()):
                    continue
            items.append(SearchItem(
                namespace=self._unpack(row[0]),
                key=row[1],
                value=value,
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
            ))
        return items

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = self._conn.execute(
            "SELECT DISTINCT namespace FROM store ORDER BY namespace LIMIT ? OFFSET ?",
            (op.limit, op.offset)
        ).fetchall()
        namespaces = [self._unpack(r[0]) for r in rows]

        if op.match_conditions:
            filtered: list[tuple[str, ...]] = []
            for ns in namespaces:
                ok = True
                for cond in op.match_conditions:
                    if cond.match_type == "prefix":
                        if ns[:len(cond.path)] != cond.path:
                            ok = False
                    elif cond.match_type == "suffix":
                        if ns[-len(cond.path):] != cond.path:
                            ok = False
                if ok:
                    filtered.append(ns)
            namespaces = filtered

        if op.max_depth is not None:
            namespaces = [ns[:op.max_depth] for ns in namespaces]

        return namespaces
