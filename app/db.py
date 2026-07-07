"""SQLite repository for hospital data (default backend).

Local-first by design: the reviewer can clone and run with zero external
accounts. Set DATABASE_URL (or DB_BACKEND=postgres) to run the identical
interface against Supabase/PostgreSQL + pgvector instead — see db_postgres.py
and get_repository() at the bottom of this module.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS hospital_group (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name_en TEXT NOT NULL,
    name_ar TEXT NOT NULL,
    tagline_en TEXT,
    tagline_ar TEXT
);

CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    name_en TEXT NOT NULL,
    name_ar TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    address_en TEXT,
    address_ar TEXT,
    phone TEXT
);

CREATE TABLE IF NOT EXISTS specializations (
    id TEXT PRIMARY KEY,
    name_en TEXT NOT NULL,
    name_ar TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS doctors (
    id TEXT PRIMARY KEY,
    name_en TEXT NOT NULL,
    name_ar TEXT NOT NULL,
    title_en TEXT,
    title_ar TEXT,
    specialization_id TEXT NOT NULL REFERENCES specializations(id),
    branch_id TEXT NOT NULL REFERENCES branches(id)
);

CREATE TABLE IF NOT EXISTS protocols (
    id TEXT PRIMARY KEY,
    specialization_id TEXT NOT NULL REFERENCES specializations(id),
    triage TEXT NOT NULL DEFAULT 'routine',
    keywords_en TEXT NOT NULL DEFAULT '[]',
    keywords_ar TEXT NOT NULL DEFAULT '[]',
    content_en TEXT NOT NULL,
    content_ar TEXT NOT NULL,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS appointments (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    doctor_id TEXT NOT NULL REFERENCES doctors(id),
    status TEXT NOT NULL DEFAULT 'confirmed',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_appointments_thread ON appointments(thread_id);
"""

# The write-path joined view of one appointment, shared by all reads.
_APPOINTMENT_SELECT = """
SELECT a.id, a.thread_id, a.status, a.created_at,
       d.id AS doctor_id, d.name_en AS doctor_en, d.name_ar AS doctor_ar,
       s.id AS specialization_id, s.name_en AS specialty_en, s.name_ar AS specialty_ar,
       b.id AS branch_id, b.name_en AS branch_en, b.name_ar AS branch_ar
FROM appointments a
JOIN doctors d ON d.id = a.doctor_id
JOIN specializations s ON s.id = d.specialization_id
JOIN branches b ON b.id = d.branch_id
"""


def new_appointment_id() -> str:
    return "APT-" + uuid.uuid4().hex[:6].upper()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("aliases", "keywords_en", "keywords_ar"):
        if key in d and isinstance(d[key], str):
            d[key] = json.loads(d[key])
    return d


class Repository:
    """All reads the chatbot performs against hospital data."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Bookings arrived after the original schema: make sure the table
        # exists on databases seeded before it was introduced.
        self._conn.executescript(
            SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS appointments"):]
        )
        self._conn.commit()

    # ---------- setup ----------

    def create_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ---------- hospital group ----------

    def hospital_group(self) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM hospital_group WHERE id = 1").fetchone()
        return dict(row) if row else {}

    def hospital_label(self, branch: dict[str, Any] | None = None) -> str:
        """e.g. 'Al-Mashreq Medical Group - Cairo Branch'."""
        group = self.hospital_group()
        name = group.get("name_en", "")
        if branch:
            return f"{name} - {branch['name_en']} Branch"
        return name

    # ---------- lookups ----------

    def list_branches(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM branches ORDER BY id").fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_specializations(self, branch_id: str | None = None) -> list[dict[str, Any]]:
        if branch_id:
            # Specializations offered at a branch = those with at least one doctor there.
            rows = self._conn.execute(
                """SELECT DISTINCT s.* FROM specializations s
                   JOIN doctors d ON d.specialization_id = s.id
                   WHERE d.branch_id = ? ORDER BY s.id""",
                (branch_id,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM specializations ORDER BY id").fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_doctors(
        self,
        specialization_id: str | None = None,
        branch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """SELECT d.*, s.name_en AS specialty_en, s.name_ar AS specialty_ar,
                          b.name_en AS branch_en, b.name_ar AS branch_ar
                   FROM doctors d
                   JOIN specializations s ON s.id = d.specialization_id
                   JOIN branches b ON b.id = d.branch_id
                   WHERE 1=1"""
        params: list[str] = []
        if specialization_id:
            query += " AND d.specialization_id = ?"
            params.append(specialization_id)
        if branch_id:
            query += " AND d.branch_id = ?"
            params.append(branch_id)
        rows = self._conn.execute(query + " ORDER BY d.id", params).fetchall()
        return [dict(r) for r in rows]

    def get_branch(self, branch_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM branches WHERE id = ?", (branch_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def get_specialization(self, spec_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM specializations WHERE id = ?", (spec_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_protocols(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT p.*, s.name_en AS specialty_en, s.name_ar AS specialty_ar
               FROM protocols p JOIN specializations s ON s.id = p.specialization_id"""
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ---------- appointments (the booking write-path) ----------

    def create_appointment(self, thread_id: str, doctor_id: str) -> dict[str, Any]:
        appointment_id = new_appointment_id()
        self._conn.execute(
            "INSERT INTO appointments (id, thread_id, doctor_id, status, created_at) VALUES (?, ?, ?, 'confirmed', ?)",
            (appointment_id, thread_id, doctor_id, utc_now()),
        )
        self._conn.commit()
        return self.get_appointment(appointment_id)

    def get_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            _APPOINTMENT_SELECT + " WHERE a.id = ?", (appointment_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_appointments(self, thread_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            _APPOINTMENT_SELECT + " WHERE a.thread_id = ? ORDER BY a.created_at",
            (thread_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def cancel_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        """Cancel a confirmed appointment; returns the updated record, or
        None when it doesn't exist / is already cancelled."""
        cursor = self._conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ? AND status = 'confirmed'",
            (appointment_id,),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_appointment(appointment_id)

    # ---------- seeding ----------

    def seed(self, dataset: dict[str, Any], embeddings: list[bytes | None]) -> None:
        """Replace all hospital data with the dataset contents."""
        self.create_schema()
        conn = self._conn
        conn.execute("DELETE FROM protocols"); conn.execute("DELETE FROM doctors")
        conn.execute("DELETE FROM specializations"); conn.execute("DELETE FROM branches")
        conn.execute("DELETE FROM hospital_group")

        group = dataset["hospital_group"]
        conn.execute(
            "INSERT INTO hospital_group (id, name_en, name_ar, tagline_en, tagline_ar) VALUES (1, ?, ?, ?, ?)",
            (group["name_en"], group["name_ar"], group.get("tagline_en"), group.get("tagline_ar")),
        )
        for branch in dataset["branches"]:
            aliases = branch.get("aliases_en", []) + branch.get("aliases_ar", [])
            conn.execute(
                "INSERT INTO branches (id, name_en, name_ar, aliases, address_en, address_ar, phone) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (branch["id"], branch["name_en"], branch["name_ar"], json.dumps(aliases, ensure_ascii=False),
                 branch.get("address_en"), branch.get("address_ar"), branch.get("phone")),
            )
        for spec in dataset["specializations"]:
            aliases = spec.get("aliases_en", []) + spec.get("aliases_ar", [])
            conn.execute(
                "INSERT INTO specializations (id, name_en, name_ar, aliases) VALUES (?, ?, ?, ?)",
                (spec["id"], spec["name_en"], spec["name_ar"], json.dumps(aliases, ensure_ascii=False)),
            )
        for doctor in dataset["doctors"]:
            conn.execute(
                "INSERT INTO doctors (id, name_en, name_ar, title_en, title_ar, specialization_id, branch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doctor["id"], doctor["name_en"], doctor["name_ar"], doctor.get("title_en"),
                 doctor.get("title_ar"), doctor["specialization_id"], doctor["branch_id"]),
            )
        for protocol, blob in zip(dataset["protocols"], embeddings):
            conn.execute(
                """INSERT INTO protocols (id, specialization_id, triage, keywords_en, keywords_ar,
                                          content_en, content_ar, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (protocol["id"], protocol["specialization_id"], protocol.get("triage", "routine"),
                 json.dumps(protocol["keywords_en"], ensure_ascii=False),
                 json.dumps(protocol["keywords_ar"], ensure_ascii=False),
                 protocol["content_en"], protocol["content_ar"], blob),
            )
        conn.commit()

    def close(self) -> None:
        self._conn.close()


def get_repository(config=None):
    """Backend factory: SQLite by default; Supabase/Postgres when DATABASE_URL
    is set (or DB_BACKEND=postgres)."""
    from .config import get_config
    config = config or get_config()
    if config.db_backend == "postgres":
        from .db_postgres import PostgresRepository
        return PostgresRepository(config.database_url)
    return Repository(config.db_path)
