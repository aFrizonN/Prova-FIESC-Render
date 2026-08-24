"""
Database management module for SQLite persistence layer.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import logging

import config

logger = logging.getLogger(__name__)


def get_db_connection() -> sqlite3.Connection:
    """Returns an active SQLite connection with row factory enabled."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes SQLite database tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Historical Sensor Readings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL,
        fault TEXT NOT NULL,
        fault_category TEXT NOT NULL,
        is_problem INTEGER NOT NULL,
        rpm REAL,
        temperature_c REAL,
        temperature_f REAL,
        z_rms_velocity_mm_s REAL,
        x_rms_velocity_mm_s REAL,
        z_peak_acceleration_g REAL,
        x_peak_acceleration_g REAL,
        z_peak_vel_comp_freq_hz REAL,
        x_peak_vel_comp_freq_hz REAL,
        z_rms_acceleration_g REAL,
        x_rms_acceleration_g REAL,
        z_kurtosis REAL,
        x_kurtosis REAL,
        z_crest_factor REAL,
        x_crest_factor REAL,
        z_peak_velocity_mm_s REAL,
        x_peak_velocity_mm_s REAL,
        z_high_freq_rms_accel_g REAL,
        x_high_freq_rms_accel_g REAL,
        total_vibration_energy REAL
    );
    """)

    # Table 2: Technical Documents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fault_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fault_category TEXT NOT NULL UNIQUE,
        document_name TEXT NOT NULL,
        document_path TEXT NOT NULL,
        title TEXT NOT NULL,
        text_content TEXT,
        chunk_count INTEGER DEFAULT 0,
        uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 3: Prescriptive Recommendations History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id INTEGER,
        fault_category TEXT NOT NULL,
        confidence REAL NOT NULL,
        similar_count INTEGER NOT NULL,
        prescription_text TEXT NOT NULL,
        has_document INTEGER NOT NULL,
        raw_event_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 4: Chat Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        event_context TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create indexes for fast query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_category ON sensor_readings(fault_category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_created ON sensor_readings(created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_problem ON sensor_readings(is_problem);")

    conn.commit()
    conn.close()
    logger.info("SQLite database tables and indexes initialized.")


def ingest_dataframe_to_db(df: pd.DataFrame, batch_size: int = 10000) -> None:
    """Ingests processed DataFrame into SQLite database efficiently."""
    conn = get_db_connection()
    
    # Check if table already has rows
    count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
    if count >= len(df):
        logger.info(f"sensor_readings already populated with {count:,} rows. Skipping ingest.")
        conn.close()
        return

    logger.info(f"Ingesting {len(df):,} records into SQLite...")
    cols_to_keep = [
        "id", "created_at", "fault", "fault_category", "is_problem",
        "rpm", "temperature_c", "temperature_f", "z_rms_velocity_mm_s",
        "x_rms_velocity_mm_s", "z_peak_acceleration_g", "x_peak_acceleration_g",
        "z_peak_vel_comp_freq_hz", "x_peak_vel_comp_freq_hz",
        "z_rms_acceleration_g", "x_rms_acceleration_g", "z_kurtosis",
        "x_kurtosis", "z_crest_factor", "x_crest_factor",
        "z_peak_velocity_mm_s", "x_peak_velocity_mm_s",
        "z_high_freq_rms_accel_g", "x_high_freq_rms_accel_g",
        "total_vibration_energy"
    ]
    
    insert_df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    insert_df["created_at"] = insert_df["created_at"].astype(str)
    insert_df["is_problem"] = insert_df["is_problem"].astype(int)

    insert_df.to_sql("sensor_readings", conn, if_exists="replace", index=False, chunksize=batch_size)
    
    # Re-apply indexes after replace
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_category ON sensor_readings(fault_category);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_created ON sensor_readings(created_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_problem ON sensor_readings(is_problem);")
    
    conn.commit()
    conn.close()
    logger.info("Database ingestion completed successfully.")


def log_prescription(
    event_id: Optional[int],
    fault_category: str,
    confidence: float,
    similar_count: int,
    prescription_text: str,
    has_document: bool,
    raw_event_json: Optional[dict] = None,
) -> int:
    """Logs a generated prescription to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO prescriptions_history 
    (event_id, fault_category, confidence, similar_count, prescription_text, has_document, raw_event_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id,
        fault_category,
        confidence,
        similar_count,
        prescription_text,
        1 if has_document else 0,
        json.dumps(raw_event_json) if raw_event_json else None
    ))
    presc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return presc_id


def log_chat_message(session_id: str, role: str, message: str, event_context: Optional[dict] = None) -> None:
    """Saves a conversation message to chat history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO chat_logs (session_id, role, message, event_context)
    VALUES (?, ?, ?, ?)
    """, (
        session_id,
        role,
        message,
        json.dumps(event_context) if event_context else None
    ))
    conn.commit()
    conn.close()


def get_kpis() -> Dict[str, Any]:
    """Retrieves top-level KPIs for the Streamlit dashboard."""
    conn = get_db_connection()
    total_events = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
    total_problems = conn.execute("SELECT COUNT(*) FROM sensor_readings WHERE is_problem = 1").fetchone()[0]
    total_docs = conn.execute("SELECT COUNT(*) FROM fault_documents").fetchone()[0]
    total_prescriptions = conn.execute("SELECT COUNT(*) FROM prescriptions_history").fetchone()[0]
    conn.close()
    return {
        "total_events": total_events,
        "total_problems": total_problems,
        "normal_ratio": ((total_events - total_problems) / (total_events + 1e-6)) * 100,
        "total_docs": total_docs,
        "total_prescriptions": total_prescriptions,
    }


def get_fault_distribution() -> pd.DataFrame:
    """Returns frequency of fault categories."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT fault_category, COUNT(*) as count, is_problem
        FROM sensor_readings
        GROUP BY fault_category
        ORDER BY count DESC
    """, conn)
    conn.close()
    return df


def get_time_series_faults() -> pd.DataFrame:
    """Returns daily counts of fault occurrences for timeline charts."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT substr(created_at, 1, 10) as date, fault_category, COUNT(*) as occurrences
        FROM sensor_readings
        WHERE is_problem = 1
        GROUP BY date, fault_category
        ORDER BY date ASC
    """, conn)
    conn.close()
    return df
