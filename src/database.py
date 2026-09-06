"""
HybridSense-X Persistent SQLite Database Manager
Provides persistent logging, telemetry, clause tracking, and analytics storage.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hybridsense.db"


def get_db_connection() -> sqlite3.Connection:
    """Establish connection to SQLite database with dict row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Main analysis log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                text TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                is_ambivalent BOOLEAN NOT NULL DEFAULT 0,
                prob_negative REAL,
                prob_neutral REAL,
                prob_positive REAL,
                prob_ambivalent REAL
            )
        """)

        # 2. Disentangled clauses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clause_breakdowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER,
                clause_index INTEGER NOT NULL,
                clause_text TEXT NOT NULL,
                polarity_score REAL NOT NULL,
                detected_polarity TEXT NOT NULL,
                discourse_marker TEXT,
                FOREIGN KEY (log_id) REFERENCES analysis_logs(id) ON DELETE CASCADE
            )
        """)

        # 3. User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER,
                suggested_label TEXT NOT NULL,
                user_comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (log_id) REFERENCES analysis_logs(id) ON DELETE CASCADE
            )
        """)

        conn.commit()


def log_analysis(
    text: str,
    predicted_class: int,
    predicted_label: str,
    confidence: float,
    model_used: str,
    latency_ms: float,
    probabilities: Optional[List[float]] = None,
    clauses: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Log an inference event into the SQLite database."""
    prob_neg = probabilities[0] if probabilities and len(probabilities) > 0 else None
    prob_neu = probabilities[1] if probabilities and len(probabilities) > 1 else None
    prob_pos = probabilities[2] if probabilities and len(probabilities) > 2 else None
    prob_amb = probabilities[3] if probabilities and len(probabilities) > 3 else None
    is_amb = 1 if predicted_class == 3 else 0

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_logs (
                text, predicted_class, predicted_label, confidence,
                model_used, latency_ms, is_ambivalent,
                prob_negative, prob_neutral, prob_positive, prob_ambivalent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text, predicted_class, predicted_label, confidence,
            model_used, latency_ms, is_amb,
            prob_neg, prob_neu, prob_pos, prob_amb
        ))
        log_id = cursor.lastrowid

        if clauses:
            for i, cl in enumerate(clauses):
                cursor.execute("""
                    INSERT INTO clause_breakdowns (
                        log_id, clause_index, clause_text, polarity_score,
                        detected_polarity, discourse_marker
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    log_id, i + 1, cl.get("text", ""),
                    cl.get("polarity_score", 0.0),
                    cl.get("detected_polarity", "Neutral"),
                    cl.get("marker", None)
                ))

        conn.commit()
        return log_id


def log_feedback(log_id: int, suggested_label: str, user_comment: str = "") -> None:
    """Record user feedback / validation rating for an inference."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback_logs (log_id, suggested_label, user_comment)
            VALUES (?, ?, ?)
        """, (log_id, suggested_label, user_comment))
        conn.commit()


def get_recent_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch the most recent analysis events."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, text, predicted_label, confidence, model_used, latency_ms
            FROM analysis_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_analytics_summary() -> Dict[str, Any]:
    """Compute aggregate analytics across all logged analyses."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM analysis_logs")
        total = cursor.fetchone()["total"]

        if total == 0:
            return {
                "total_inferences": 0,
                "sentiment_distribution": {},
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "model_usage": {},
                "ambivalent_count": 0
            }

        cursor.execute("""
            SELECT predicted_label, COUNT(*) as count
            FROM analysis_logs
            GROUP BY predicted_label
        """)
        distribution = {row["predicted_label"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT AVG(confidence) as avg_conf, AVG(latency_ms) as avg_lat
            FROM analysis_logs
        """)
        stats = cursor.fetchone()

        cursor.execute("""
            SELECT model_used, COUNT(*) as count
            FROM analysis_logs
            GROUP BY model_used
        """)
        models = {row["model_used"]: row["count"] for row in cursor.fetchall()}

        amb_count = distribution.get("Ambivalent", 0)

        return {
            "total_inferences": total,
            "sentiment_distribution": distribution,
            "avg_confidence": round(stats["avg_conf"] or 0.0, 4),
            "avg_latency_ms": round(stats["avg_lat"] or 0.0, 2),
            "model_usage": models,
            "ambivalent_count": amb_count
        }


# Initialize DB upon import
init_db()
