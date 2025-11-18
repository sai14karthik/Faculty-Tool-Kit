# backend/db.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "faculty_toolkit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint TEXT,
        payload TEXT,
        result TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def log_request(endpoint: str, payload: str, result: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO requests (endpoint, payload, result) VALUES (?, ?, ?)",
        (endpoint, payload, result),
    )
    conn.commit()
    conn.close()

def reset_requests():
    """Remove all logged requests (used to clear stats)"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM requests")
    conn.commit()
    conn.close()

# ensure DB exists on import
init_db()
