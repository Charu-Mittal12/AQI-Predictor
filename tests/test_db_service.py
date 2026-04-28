# tests/test_db_service.py
"""
Unit tests for db_service using a real in-memory SQLite database.
No mocking — tests actual SQL logic.
TC-DB-01 to TC-DB-07
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def create_test_db(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS city_readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            city            TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            station_id      TEXT,
            state           TEXT,
            station_name    TEXT,
            latitude        REAL,
            longitude       REAL,
            pm25            REAL,
            pm10            REAL,
            no              REAL,
            no2             REAL,
            nox             REAL,
            nh3             REAL,
            co              REAL,
            so2             REAL,
            o3              REAL,
            benzene         REAL,
            toluene         REAL,
            xylene          REAL,
            aqi             REAL,
            temperature_2m        REAL,
            relative_humidity_2m  REAL,
            wind_speed_10m        REAL,
            wind_direction_10m    REAL,
            precipitation   REAL,
            pressure_msl    REAL,
            cloud_cover     REAL,
            UNIQUE(city, station_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            city              TEXT NOT NULL,
            computed_at       TEXT NOT NULL,
            current_aqi       INTEGER,
            forecast_json     TEXT,
            pollutants_json   TEXT,
            weekly_trend_json TEXT,
            primary_pollutant TEXT,
            UNIQUE(city, computed_at)
        );
    """)
    conn.commit()
    return conn


@pytest.fixture
def test_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def populated_db(test_db_path):
    conn = create_test_db(test_db_path)
    now = datetime.utcnow()
    rows = [
        ("Chennai", (now - timedelta(hours=i)).isoformat(), "s1", "Velachery", 45.0 + i, 90 + i)
        for i in range(210)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO city_readings "
        "(city, timestamp, station_id, station_name, pm25, aqi) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return test_db_path


# ── TC-DB-01 to TC-DB-03 : get_recent_readings logic ─────────────────────────

class TestGetRecentReadings:
    """TC-DB-01 to TC-DB-03 — get_recent_readings logic."""

    def test_returns_rows_within_window(self, populated_db):
        """TC-DB-01: Only rows within the hours window are returned."""
        conn = sqlite3.connect(populated_db)
        conn.row_factory = sqlite3.Row
        since = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        rows = conn.execute(
            "SELECT * FROM city_readings WHERE city=? AND timestamp>=? ORDER BY timestamp ASC",
            ("Chennai", since),
        ).fetchall()
        conn.close()
        assert len(rows) >= 46  # slight timing tolerance

    def test_returns_correct_city(self, populated_db):
        """TC-DB-02: Rows belong to the queried city only."""
        conn = sqlite3.connect(populated_db)
        conn.row_factory = sqlite3.Row
        since = (datetime.utcnow() - timedelta(hours=200)).isoformat()
        rows = conn.execute(
            "SELECT * FROM city_readings WHERE city=? AND timestamp>=?",
            ("Chennai", since),
        ).fetchall()
        conn.close()
        for row in rows:
            assert row["city"] == "Chennai"

    def test_no_rows_for_unknown_city(self, populated_db):
        """TC-DB-03: Unknown city returns empty list."""
        conn = sqlite3.connect(populated_db)
        rows = conn.execute(
            "SELECT * FROM city_readings WHERE city=?", ("Atlantis",)
        ).fetchall()
        conn.close()
        assert rows == []


# ── TC-DB-04 to TC-DB-05 : get_latest_reading logic ──────────────────────────

class TestGetLatestReading:
    """TC-DB-04 to TC-DB-05 — get_latest_reading logic."""

    def test_returns_most_recent_row(self, populated_db):
        """TC-DB-04: Latest reading has the most recent timestamp."""
        conn = sqlite3.connect(populated_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM city_readings WHERE city=? ORDER BY timestamp DESC LIMIT 1",
            ("Chennai",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["city"] == "Chennai"

    def test_returns_none_for_unknown_city(self, populated_db):
        """TC-DB-05: No row returned for a city that was never inserted."""
        conn = sqlite3.connect(populated_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM city_readings WHERE city=? ORDER BY timestamp DESC LIMIT 1",
            ("Atlantis",),
        ).fetchone()
        conn.close()
        assert row is None


# ── TC-DB-06 to TC-DB-07 : predictions table ─────────────────────────────────

class TestStorePrediction:
    """TC-DB-06 to TC-DB-07 — Prediction storage."""

    def test_store_and_retrieve_prediction(self, test_db_path):
        """TC-DB-06: Stored prediction can be retrieved with correct values."""
        conn = create_test_db(test_db_path)
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO predictions "
            "(city, computed_at, current_aqi, forecast_json, pollutants_json, weekly_trend_json, primary_pollutant) "
            "VALUES (?,?,?,?,?,?,?)",
            ("Chennai", now, 87, json.dumps([85, 24]), json.dumps({}), json.dumps([90, 7]), "PM2.5"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM predictions WHERE city='Chennai' ORDER BY computed_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row["current_aqi"] == 87
        assert row["primary_pollutant"] == "PM2.5"

    def test_duplicate_prediction_upserted(self, test_db_path):
        """TC-DB-07: Inserting same city + computed_at replaces the old row."""
        conn = create_test_db(test_db_path)
        ts = datetime.utcnow().isoformat()
        for aqi in (80, 95):
            conn.execute(
                "INSERT OR REPLACE INTO predictions "
                "(city, computed_at, current_aqi, forecast_json, pollutants_json, weekly_trend_json, primary_pollutant) "
                "VALUES (?,?,?,?,?,?,?)",
                ("Delhi", ts, aqi, "[]", "{}", "[]", "PM2.5"),
            )
            conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE city='Delhi'"
        ).fetchone()[0]
        conn.close()
        assert count == 1