# FILE: a3d/logging_utils.py
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "t": int(time.time()*1000),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)

def setup_json_logging(level: int = logging.INFO) -> None:
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h]


def log_decode_events(path: str, axis: str, decoder: str, edges_count: int) -> None:
    """Durably record explicitly requested events; propagate write failures."""
    import csv
    import os

    os.makedirs(
        os.path.dirname(os.path.abspath(path)), exist_ok=True
    )
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([axis, decoder, edges_count])
        handle.flush()
        os.fsync(handle.fileno())
