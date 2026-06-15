from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import settings


@dataclass
class UserPreference:
    user_id: str
    home_location: str = ""
    transport_prefer: str = "AUTO"
    budget_level: str = "STANDARD"
    hotel_prefer: str = "COMFORT"
    interest_tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "home_location": self.home_location,
            "transport_prefer": self.transport_prefer,
            "budget_level": self.budget_level,
            "hotel_prefer": self.hotel_prefer,
            "interest_tags": self.interest_tags or [],
        }


class PreferenceStore:
    def __init__(self) -> None:
        settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = settings.sqlite_db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    home_location TEXT DEFAULT '',
                    transport_prefer TEXT DEFAULT 'AUTO',
                    budget_level TEXT DEFAULT 'STANDARD',
                    hotel_prefer TEXT DEFAULT 'COMFORT',
                    interest_tags TEXT DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get_user_pref(self, user_id: str) -> UserPreference:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            pref = UserPreference(user_id=user_id, interest_tags=[])
            self.save_user_pref(pref)
            return pref
        return UserPreference(
            user_id=row["user_id"],
            home_location=row["home_location"],
            transport_prefer=row["transport_prefer"],
            budget_level=row["budget_level"],
            hotel_prefer=row["hotel_prefer"],
            interest_tags=json.loads(row["interest_tags"] or "[]"),
        )

    def save_user_pref(self, pref: UserPreference) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences
                    (user_id, home_location, transport_prefer, budget_level, hotel_prefer, interest_tags, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    home_location=excluded.home_location,
                    transport_prefer=excluded.transport_prefer,
                    budget_level=excluded.budget_level,
                    hotel_prefer=excluded.hotel_prefer,
                    interest_tags=excluded.interest_tags,
                    updated_at=excluded.updated_at
                """,
                (
                    pref.user_id,
                    pref.home_location,
                    pref.transport_prefer,
                    pref.budget_level,
                    pref.hotel_prefer,
                    json.dumps(pref.interest_tags or [], ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def update_from_message(self, user_id: str, message: str) -> dict[str, Any]:
        pref = self.get_user_pref(user_id)
        changed: dict[str, Any] = {}

        if any(word in message for word in ["坐飞机", "飞机优先", "习惯坐飞机"]):
            pref.transport_prefer = "FLIGHT"
            changed["transport_prefer"] = "FLIGHT"
        elif any(word in message for word in ["坐高铁", "坐火车", "高铁优先", "喜欢火车"]):
            pref.transport_prefer = "TRAIN"
            changed["transport_prefer"] = "TRAIN"
        elif any(word in message for word in ["自驾", "开车"]):
            pref.transport_prefer = "DRIVING"
            changed["transport_prefer"] = "DRIVING"

        if any(word in message for word in ["省钱", "自费", "便宜", "经济"]):
            pref.budget_level = "ECONOMY"
            pref.hotel_prefer = "BUDGET"
            changed["budget_level"] = "ECONOMY"
            changed["hotel_prefer"] = "BUDGET"
        elif any(word in message for word in ["公司报销", "商务", "出差"]):
            pref.budget_level = "BUSINESS"
            pref.hotel_prefer = "BUSINESS"
            changed["budget_level"] = "BUSINESS"
            changed["hotel_prefer"] = "BUSINESS"
        elif any(word in message for word in ["豪华", "五星", "高端"]):
            pref.budget_level = "LUXURY"
            pref.hotel_prefer = "LUXURY"
            changed["budget_level"] = "LUXURY"
            changed["hotel_prefer"] = "LUXURY"

        home_match = re.search(r"(?:我是|我家在|常住|住在)([\u4e00-\u9fff]{2,4})", message)
        if home_match:
            pref.home_location = home_match.group(1)
            changed["home_location"] = pref.home_location

        tags = set(pref.interest_tags or [])
        for tag in ["历史", "美食", "购物", "亲子", "自然", "博物馆", "夜景"]:
            if tag in message:
                tags.add(tag)
        if tags != set(pref.interest_tags or []):
            pref.interest_tags = sorted(tags)
            changed["interest_tags"] = pref.interest_tags

        if changed:
            self.save_user_pref(pref)
        return {"preference": pref.to_dict(), "changed": changed}

    def save_conversation_state(self, thread_id: str, user_id: str, state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_state (thread_id, user_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    thread_id,
                    user_id,
                    json.dumps(state, ensure_ascii=False, default=str),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_conversation_state(self, thread_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT state_json FROM conversation_state WHERE thread_id = ?", (thread_id,)).fetchone()
        return json.loads(row["state_json"]) if row else {}

    def reset_conversation(self, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_state WHERE thread_id = ?", (thread_id,))


preference_store = PreferenceStore()
