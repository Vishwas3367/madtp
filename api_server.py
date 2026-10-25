"""Small standard-library API and static-file server for the MADTP prototype.

Serves the existing live prediction CSV to the frontend. It does not calculate
risk itself; live_madtp.py remains the source of prediction outputs.
"""
from __future__ import annotations
import csv
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "MADTP_Frontend" / "public"
RESULT_FILE = ROOT / "data" / "live_madtp_results.csv"
HOST = os.environ.get("MADTP_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("MADTP_API_PORT", "8000"))


def read_results():
    if not RESULT_FILE.exists() or RESULT_FILE.stat().st_size == 0:
        return []
    try:
        with RESULT_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {
                "heart_rate", "spo2", "current_risk_score", "risk_category",
                "twin_divergence_index", "rf_deterioration_probability",
                "ds18b20_surface_temperature_c", "dht22_ambient_temperature_c",
            }
            # Old experiment CSVs used different headers. Never present their
            # shifted columns as live measurements; wait for the live runner to
            # rotate the legacy file and write the stable schema.
            if not required.issubset(set(reader.fieldnames or [])):
                return []
            return list(reader)
    except (OSError, csv.Error):
        return []


def number(value):
    try:
        if value is None or value == "":
            return None
        n = float(value)
        return n if n == n and abs(n) != float("inf") else None
    except (TypeError, ValueError):
        return None


def clean_row(row):
    fields = [
        "timestamp_ms", "heart_rate", "spo2", "ds18b20_surface_temperature_c",
        "dht22_ambient_temperature_c", "dht22_humidity_percent",
        "mpu6050_accel_x", "mpu6050_accel_y", "mpu6050_accel_z",
        "mpu6050_gyro_x", "mpu6050_gyro_y", "mpu6050_gyro_z",
        "activity_heuristic", "heat_load", "exposure_time_minutes",
        "twin_divergence_index", "rf_prediction", "rf_deterioration_probability",
        "current_risk_score", "risk_category", "prediction_number", "received_utc",
    ]
    out = {}
    for key in fields:
        value = row.get(key)
        if key in {"risk_category", "received_utc"}:
            out[key] = value or None
        else:
            out[key] = number(value)
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            rows = read_results()
            self.send_json({"ok": True, "has_data": bool(rows), "rows": len(rows),
                            "source": str(RESULT_FILE.name)})
            return
        if path == "/api/latest":
            rows = read_results()
            self.send_json({"ok": True, "data": clean_row(rows[-1]) if rows else None})
            return
        if path == "/api/history":
            rows = read_results()[-100:]
            self.send_json({"ok": True, "data": [clean_row(row) for row in rows]})
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format, *args):
        # Keep the console readable; endpoint errors still appear in responses.
        if self.path.startswith("/api/"):
            print("[API] " + format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"MADTP dashboard/API: http://{HOST}:{PORT}")
    print(f"Serving prediction results from: {RESULT_FILE}")
    print("Start live_madtp.py in another terminal to populate live results.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nDashboard/API server stopped.")
    finally:
        server.server_close()
