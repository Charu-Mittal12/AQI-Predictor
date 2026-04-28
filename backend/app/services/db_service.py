import sqlite3
import json
from datetime import datetime, timedelta
from app.utils.config import DB_PATH
from app.utils.logger import get_logger

log = get_logger(__name__)

CITY_READING_COLUMNS = (
    "city", "timestamp", "station_id", "state", "station_name",
    "latitude", "longitude",
    "pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3",
    "benzene", "toluene", "xylene", "aqi",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "pressure_msl", "cloud_cover",
)


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_city_readings_columns(conn):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(city_readings)").fetchall()}
    expected = {
        "station_id": "TEXT",
        "state": "TEXT",
        "station_name": "TEXT",
        "latitude": "REAL",
        "longitude": "REAL",
        "pm10": "REAL",
        "nox": "REAL",
        "benzene": "REAL",
        "toluene": "REAL",
        "xylene": "REAL",
    }
    for col, col_type in expected.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE city_readings ADD COLUMN {col} {col_type}")


def _migrate_city_readings_constraint(conn):
    """
    If city_readings still has old UNIQUE(city, timestamp), migrate to
    UNIQUE(city, station_id, timestamp) by recreating the table.
    Preserves all existing rows.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='city_readings'"
    ).fetchone()
    if not row:
        return  # table doesn't exist yet; init_db will create it correctly

    existing_ddl = (row[0] or "").replace(" ", "").replace("\n", "")
    if "UNIQUE(city,station_id,timestamp)" in existing_ddl:
        return  # already on new constraint

    log.info(
        "Migrating city_readings UNIQUE constraint: "
        "(city, timestamp) -> (city, station_id, timestamp)"
    )
    conn.executescript("""
        ALTER TABLE city_readings RENAME TO city_readings_old;

        CREATE TABLE city_readings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            city                TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            station_id           TEXT,
            state               TEXT,
            station_name         TEXT,
            latitude            REAL,
            longitude           REAL,
            pm25                REAL,
            pm10                REAL,
            no                  REAL,
            no2                 REAL,
            nox                 REAL,
            nh3                 REAL,
            co                  REAL,
            so2                 REAL,
            o3                  REAL,
            benzene             REAL,
            toluene             REAL,
            xylene              REAL,
            aqi                 REAL,
            temperature_2m       REAL,
            relative_humidity_2m  REAL,
            wind_speed_10m        REAL,
            wind_direction_10m    REAL,
            precipitation       REAL,
            pressure_msl         REAL,
            cloud_cover          REAL,
            UNIQUE(city, station_id, timestamp)
        );

        INSERT OR IGNORE INTO city_readings SELECT * FROM city_readings_old;
        DROP TABLE city_readings_old;
    """)
    conn.commit()
    log.info("city_readings migration completed successfully.")


def init_db():
    conn = get_connection()

    # Run migration first if table already exists with old constraint
    _migrate_city_readings_constraint(conn)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS city_readings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            city                TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            station_id           TEXT,
            state               TEXT,
            station_name         TEXT,
            latitude            REAL,
            longitude           REAL,
            pm25                REAL,
            pm10                REAL,
            no                  REAL,
            no2                 REAL,
            nox                 REAL,
            nh3                 REAL,
            co                  REAL,
            so2                 REAL,
            o3                  REAL,
            benzene             REAL,
            toluene             REAL,
            xylene              REAL,
            aqi                 REAL,
            temperature_2m       REAL,
            relative_humidity_2m  REAL,
            wind_speed_10m        REAL,
            wind_direction_10m    REAL,
            precipitation       REAL,
            pressure_msl         REAL,
            cloud_cover          REAL,
            UNIQUE(city, station_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            city             TEXT NOT NULL,
            computed_at      TEXT NOT NULL,
            current_aqi      INTEGER,
            forecast_json    TEXT,
            pollutants_json  TEXT,
            weekly_trend_json TEXT,
            primary_pollutant TEXT,
            UNIQUE(city, computed_at)
        );
    """)

    ensure_city_readings_columns(conn)
    conn.commit()
    conn.close()
    log.info("Database initialized.")


def init_ingestion_tables():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sensor_registry (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id           TEXT NOT NULL,
            city                TEXT NOT NULL,
            station_name         TEXT,
            openaq_location_id  INTEGER NOT NULL,
            openaq_location_name TEXT,
            sensor_id           INTEGER NOT NULL,
            parameter           TEXT NOT NULL,
            units               TEXT,
            last_fetched_hour   TEXT,
            last_successful_sync TEXT,
            UNIQUE(openaq_location_id, sensor_id)
        );

        CREATE TABLE IF NOT EXISTS hourly_measurements (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id        TEXT NOT NULL,
            city             TEXT NOT NULL,
            openaq_location_id INTEGER NOT NULL,
            sensor_id        INTEGER NOT NULL,
            parameter        TEXT NOT NULL,
            units            TEXT,
            raw_value        REAL,
            raw_unit         TEXT,
            measurement_time TEXT NOT NULL,
            value            REAL,
            ingested_at      TEXT NOT NULL,
            UNIQUE(sensor_id, measurement_time)
        );
    """)
    conn.commit()
    conn.close()
    log.info("Ingestion tables initialized.")


def ensure_all_tables():
    init_db()
    init_ingestion_tables()


def store_readings(city: str, data: dict):
    payload = dict(data)
    payload["city"] = city
    store_readings_batch([payload])


def store_readings_batch(rows: list[dict]):
    if not rows:
        return
    conn = get_connection()
    placeholders = ",".join(["?"] * len(CITY_READING_COLUMNS))
    sql = (
        f"INSERT OR REPLACE INTO city_readings "
        f"({','.join(CITY_READING_COLUMNS)}) VALUES ({placeholders})"
    )
    values = []
    for row in rows:
        values.append(
            tuple(
                datetime.utcnow().isoformat() if col == "timestamp" and row.get(col) is None
                else row.get(col)
                for col in CITY_READING_COLUMNS
            )
        )
    conn.executemany(sql, values)
    conn.commit()
    conn.close()


# ── FIX 1: location_id-aware query ────────────────────────────────────────────
def get_recent_readings(city: str, hours: int = 200, location_id: int | None = None) -> list[dict]:
    """
    Return rows for the given city within the last `hours` hours.
    If location_id is provided, restrict to station_id = 'openaq{location_id}'.
    """
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    conn = get_connection()

    if location_id is not None:
        station_id = f"openaq{location_id}"
        rows = conn.execute(
            """
            SELECT * FROM city_readings
            WHERE city = ? AND station_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (city, station_id, since),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM city_readings
            WHERE city = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (city, since),
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ── FIX 2: location_id-aware latest row ───────────────────────────────────────
def get_latest_reading(city: str, location_id: int | None = None) -> dict | None:
    """
    Return the most recent row for the given city.
    If location_id is provided, restrict to station_id = 'openaq{location_id}'.
    """
    conn = get_connection()

    if location_id is not None:
        station_id = f"openaq{location_id}"
        row = conn.execute(
            """
            SELECT * FROM city_readings
            WHERE city = ? AND station_id = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (city, station_id),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT * FROM city_readings
            WHERE city = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (city,),
        ).fetchone()

    conn.close()
    return dict(row) if row else None


def store_prediction(city: str, result: dict):
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO predictions
            (city, computed_at, current_aqi, forecast_json,
             pollutants_json, weekly_trend_json, primary_pollutant)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            city,
            datetime.utcnow().isoformat(),
            result.get("current_aqi"),
            json.dumps(result.get("forecast_24h", [])),
            json.dumps(result.get("pollutants", {})),
            json.dumps(result.get("weekly_trend", [])),
            result.get("primary_pollutant", "PM2.5"),
        ),
    )
    conn.commit()
    conn.close()


def get_latest_prediction(city: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM predictions WHERE city = ? ORDER BY computed_at DESC LIMIT 1",
        (city,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "city": city,
        "current_aqi": row["current_aqi"],
        "primary_pollutant": row["primary_pollutant"],
        "last_updated": row["computed_at"],
        "forecast_24h": json.loads(row["forecast_json"]),
        "pollutants": json.loads(row["pollutants_json"]),
        "weekly_trend": json.loads(row["weekly_trend_json"]),
    }


def cleanup_old_readings(hours: int = 200):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    conn = get_connection()
    conn.execute("DELETE FROM city_readings WHERE timestamp < ?", (cutoff,))
    conn.commit()
    conn.close()
    log.info(f"Cleaned up readings older than {hours}h.")


def cleanup_old_predictions(days: int = 7):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    conn = get_connection()
    conn.execute("DELETE FROM predictions WHERE computed_at < ?", (cutoff,))
    conn.commit()
    conn.close()
    log.info(f"Cleaned up predictions older than {days} days.")


def upsert_sensor_registry(rows: list[dict]):
    if not rows:
        return
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    sql = """
        INSERT INTO sensor_registry
            (station_id, city, station_name, openaq_location_id, openaq_location_name,
             sensor_id, parameter, units, last_fetched_hour, last_successful_sync)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(openaq_location_id, sensor_id) DO UPDATE SET
            station_id            = excluded.station_id,
            city                 = excluded.city,
            station_name          = excluded.station_name,
            openaq_location_name = excluded.openaq_location_name,
            parameter            = excluded.parameter,
            units                = excluded.units,
            last_successful_sync = excluded.last_successful_sync
    """
    values = [
        (
            row.get("station_id"),
            row.get("city"),
            row.get("station_name"),
            row.get("openaq_location_id"),
            row.get("openaq_location_name"),
            row.get("sensor_id"),
            row.get("parameter"),
            row.get("units"),
            row.get("last_fetched_hour"),
            now,
        )
        for row in rows
    ]
    conn.executemany(sql, values)
    conn.commit()
    conn.close()


def get_sensor_registry() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sensor_registry ORDER BY city, station_id, parameter"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_measurement_time(sensor_id: int) -> str | None:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT measurement_time FROM hourly_measurements
        WHERE sensor_id = ?
        ORDER BY measurement_time DESC LIMIT 1
        """,
        (sensor_id,),
    ).fetchone()
    conn.close()
    return row["measurement_time"] if row else None


def insert_hourly_measurements(rows: list[dict]) -> int:
    if not rows:
        return 0
    conn = get_connection()
    sql = """
        INSERT OR IGNORE INTO hourly_measurements
            (station_id, city, openaq_location_id, sensor_id, parameter, units,
             raw_value, raw_unit, measurement_time, value, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    now = datetime.utcnow().isoformat()
    values = [
        (
            row.get("station_id"),
            row.get("city"),
            row.get("openaq_location_id"),
            row.get("sensor_id"),
            row.get("parameter"),
            row.get("units"),
            row.get("raw_value"),
            row.get("raw_unit"),
            row.get("measurement_time"),
            row.get("value"),
            now,
        )
        for row in rows
    ]
    cur = conn.executemany(sql, values)
    conn.commit()
    inserted = cur.rowcount if cur.rowcount is not None else len(values)
    conn.close()
    return inserted


def update_last_fetched(sensor_id: int, measurement_time: str):
    conn = get_connection()
    conn.execute(
        """
        UPDATE sensor_registry
        SET last_fetched_hour = ?, last_successful_sync = ?
        WHERE sensor_id = ?
        """,
        (measurement_time, datetime.utcnow().isoformat(), sensor_id),
    )
    conn.commit()
    conn.close()


def cleanup_old_hourly_measurements(days: int = 10):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    conn = get_connection()
    conn.execute("DELETE FROM hourly_measurements WHERE measurement_time < ?", (cutoff,))
    conn.commit()
    conn.close()
    log.info(f"Cleaned up hourly_measurements older than {days} days.")