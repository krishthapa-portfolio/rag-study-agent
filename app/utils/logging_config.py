"""
Audit logging — every ingestion and query event gets a structured log line.
This is what you'd point to in an interview when asked "how would you debug
a bad answer in production" — you trace the audit log for that request_id.
"""
import logging
import json
import time
from pathlib import Path

Path("./data/logs").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("student_rag_agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("./data/logs/audit.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def audit_log(event_type: str, **fields):
    """
    Structured JSON log line. event_type examples:
    'ingestion_start', 'ingestion_blocked_injection', 'pii_masked',
    'query_received', 'retrieval_insufficient', 'query_answered'
    """
    record = {
        "timestamp": time.time(),
        "event": event_type,
        **fields,
    }
    logger.info(json.dumps(record))
