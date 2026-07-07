"""Supabase / PostgreSQL repository (pgvector) — same interface as db.Repository.

Activate by setting DATABASE_URL to a Postgres connection string (for Supabase:
Project → Connect → Connection String → Session pooler URI), then run
`python scripts/seed_db.py` once. The rest of the system is backend-agnostic.

Protocol embeddings are stored in a pgvector `vector` column. At the current
knowledge-base size retrieval stays in-process (identical scoring to the SQLite
backend); at scale the dense ranking moves into SQL with
`ORDER BY embedding <=> %s` — one method, same interface.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

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
    aliases JSONB NOT NULL DEFAULT '[]',
    address_en TEXT,
    address_ar TEXT,
    phone TEXT
);

CREATE TABLE IF NOT EXISTS specializations (
    id TEXT PRIMARY KEY,
    name_en TEXT NOT NULL,
    name_ar TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'
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
    keywords_en JSONB NOT NULL DEFAULT '[]',
    keywords_ar JSONB NOT NULL DEFAULT '[]',
    content_en TEXT NOT NULL,
    content_ar TEXT NOT NULL,
    embedding vector
);
"""

APPOINTMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    doctor_id TEXT NOT NULL REFERENCES doctors(id),
    status TEXT NOT NULL DEFAULT 'confirmed',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_appointments_thread ON appointments(thread_id);
"""

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


def _vector_literal(blob: bytes | None) -> str | None:
    if blob is None:
        return None
    values = np.frombuffer(blob, dtype=np.float32)
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


def _vector_to_blob(text: str | None) -> bytes | None:
    if not text:
        return None
    return np.asarray(json.loads(text), dtype=np.float32).tobytes()


class PostgresRepository:
    """Mirror of db.Repository backed by Supabase/PostgreSQL."""

    def __init__(self, database_url: str):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise RuntimeError(
                "The Postgres backend needs psycopg2: pip install psycopg2-binary"
            ) from exc
        self._psycopg2 = psycopg2
        self._conn = psycopg2.connect(database_url)
        self._conn.autocommit = False
        self._dict_cursor = psycopg2.extras.RealDictCursor
        # Bookings arrived after the original schema: ensure the table exists.
        # A restricted role may lack CREATE (table then ships via migration).
        try:
            with self._conn.cursor() as cur:
                cur.execute(APPOINTMENTS_SCHEMA)
            self._conn.commit()
        except Exception:
            self._conn.rollback()

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._conn.cursor(cursor_factory=self._dict_cursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        self._conn.rollback()  # release read snapshot
        return rows

    # ---------- setup ----------

    def create_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)
        self._conn.commit()

    def conn(self):
        return self._conn

    # ---------- hospital group ----------

    def hospital_group(self) -> dict[str, Any]:
        rows = self._query("SELECT * FROM hospital_group WHERE id = 1")
        return rows[0] if rows else {}

    def hospital_label(self, branch: dict[str, Any] | None = None) -> str:
        name = self.hospital_group().get("name_en", "")
        if branch:
            return f"{name} - {branch['name_en']} Branch"
        return name

    # ---------- lookups ----------

    def list_branches(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM branches ORDER BY id")

    def list_specializations(self, branch_id: str | None = None) -> list[dict[str, Any]]:
        if branch_id:
            return self._query(
                """SELECT DISTINCT s.* FROM specializations s
                   JOIN doctors d ON d.specialization_id = s.id
                   WHERE d.branch_id = %s ORDER BY s.id""",
                (branch_id,),
            )
        return self._query("SELECT * FROM specializations ORDER BY id")

    def list_doctors(
        self,
        specialization_id: str | None = None,
        branch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """SELECT d.*, s.name_en AS specialty_en, s.name_ar AS specialty_ar,
                        b.name_en AS branch_en, b.name_ar AS branch_ar
                 FROM doctors d
                 JOIN specializations s ON s.id = d.specialization_id
                 JOIN branches b ON b.id = d.branch_id
                 WHERE 1=1"""
        params: list[str] = []
        if specialization_id:
            sql += " AND d.specialization_id = %s"
            params.append(specialization_id)
        if branch_id:
            sql += " AND d.branch_id = %s"
            params.append(branch_id)
        return self._query(sql + " ORDER BY d.id", tuple(params))

    def get_branch(self, branch_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM branches WHERE id = %s", (branch_id,))
        return rows[0] if rows else None

    def get_specialization(self, spec_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM specializations WHERE id = %s", (spec_id,))
        return rows[0] if rows else None

    def list_protocols(self) -> list[dict[str, Any]]:
        rows = self._query(
            """SELECT p.id, p.specialization_id, p.triage, p.keywords_en, p.keywords_ar,
                      p.content_en, p.content_ar, p.embedding::text AS embedding,
                      s.name_en AS specialty_en, s.name_ar AS specialty_ar
               FROM protocols p JOIN specializations s ON s.id = p.specialization_id"""
        )
        for row in rows:
            row["embedding"] = _vector_to_blob(row["embedding"])
        return rows

    # ---------- appointments (the booking write-path) ----------

    def create_appointment(self, thread_id: str, doctor_id: str) -> dict[str, Any]:
        from .db import new_appointment_id, utc_now

        appointment_id = new_appointment_id()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO appointments (id, thread_id, doctor_id, status, created_at) VALUES (%s, %s, %s, 'confirmed', %s)",
                (appointment_id, thread_id, doctor_id, utc_now()),
            )
        self._conn.commit()
        return self.get_appointment(appointment_id)

    def get_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        rows = self._query(_APPOINTMENT_SELECT + " WHERE a.id = %s", (appointment_id,))
        return rows[0] if rows else None

    def list_appointments(self, thread_id: str) -> list[dict[str, Any]]:
        return self._query(
            _APPOINTMENT_SELECT + " WHERE a.thread_id = %s ORDER BY a.created_at",
            (thread_id,),
        )

    def cancel_appointment(self, appointment_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE appointments SET status = 'cancelled' WHERE id = %s AND status = 'confirmed'",
                (appointment_id,),
            )
            updated = cur.rowcount
        self._conn.commit()
        if not updated:
            return None
        return self.get_appointment(appointment_id)

    # ---------- seeding ----------

    def seed(self, dataset: dict[str, Any], embeddings: list[bytes | None]) -> None:
        self.create_schema()
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM protocols"); cur.execute("DELETE FROM doctors")
            cur.execute("DELETE FROM specializations"); cur.execute("DELETE FROM branches")
            cur.execute("DELETE FROM hospital_group")

            group = dataset["hospital_group"]
            cur.execute(
                "INSERT INTO hospital_group (id, name_en, name_ar, tagline_en, tagline_ar) VALUES (1, %s, %s, %s, %s)",
                (group["name_en"], group["name_ar"], group.get("tagline_en"), group.get("tagline_ar")),
            )
            for branch in dataset["branches"]:
                aliases = branch.get("aliases_en", []) + branch.get("aliases_ar", [])
                cur.execute(
                    "INSERT INTO branches (id, name_en, name_ar, aliases, address_en, address_ar, phone) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (branch["id"], branch["name_en"], branch["name_ar"], json.dumps(aliases, ensure_ascii=False),
                     branch.get("address_en"), branch.get("address_ar"), branch.get("phone")),
                )
            for spec in dataset["specializations"]:
                aliases = spec.get("aliases_en", []) + spec.get("aliases_ar", [])
                cur.execute(
                    "INSERT INTO specializations (id, name_en, name_ar, aliases) VALUES (%s, %s, %s, %s)",
                    (spec["id"], spec["name_en"], spec["name_ar"], json.dumps(aliases, ensure_ascii=False)),
                )
            for doctor in dataset["doctors"]:
                cur.execute(
                    "INSERT INTO doctors (id, name_en, name_ar, title_en, title_ar, specialization_id, branch_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (doctor["id"], doctor["name_en"], doctor["name_ar"], doctor.get("title_en"),
                     doctor.get("title_ar"), doctor["specialization_id"], doctor["branch_id"]),
                )
            for protocol, blob in zip(dataset["protocols"], embeddings):
                cur.execute(
                    """INSERT INTO protocols (id, specialization_id, triage, keywords_en, keywords_ar,
                                              content_en, content_ar, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                    (protocol["id"], protocol["specialization_id"], protocol.get("triage", "routine"),
                     json.dumps(protocol["keywords_en"], ensure_ascii=False),
                     json.dumps(protocol["keywords_ar"], ensure_ascii=False),
                     protocol["content_en"], protocol["content_ar"], _vector_literal(blob)),
                )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
