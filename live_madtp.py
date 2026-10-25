import json
import time
import os
import warnings
import logging
import traceback
import csv
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from preprocessing.baseline import PersonalBaseline
from preprocessing.feature_extraction import FeatureExtractor
from madtp.multimodal_fusion import MultimodalFusion
from madtp.risk_prediction import MADTPRiskPredictor
from madtp.twin_divergence import TwinDivergenceAnalyzer
from ai.random_forest import RandomForestPredictor
from madtp.live_simulator import LiveSimulator

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MADTP-Server")

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

RAW_FILE = os.path.join(PROJECT_ROOT, "data", "live_madtp_serial_raw.csv")
RESULT_FILE = os.path.join(PROJECT_ROOT, "data", "live_madtp_results.csv")
RF_MODEL_FILE = os.path.join(PROJECT_ROOT, "ai", "random_forest_model.pkl")

HISTORY_SIZE = 300
BASELINE_DURATION_MINUTES = float(os.environ.get("MADTP_BASELINE_MINUTES", "1"))
MIN_BASELINE_SAMPLES = 1

HEART_RATE_MAX_AGE_SECONDS = 10.0
SPO2_MAX_AGE_SECONDS = 10.0

RESULT_COLUMNS = [
    "received_utc", "timestamp_ms", "heart_rate", "spo2",
    "ds18b20_surface_temperature_c", "dht22_ambient_temperature_c",
    "dht22_humidity_percent", "mpu6050_accel_x", "mpu6050_accel_y",
    "mpu6050_accel_z", "mpu6050_gyro_x", "mpu6050_gyro_y",
    "mpu6050_gyro_z", "ambient_temperature", "humidity",
    "activity_heuristic", "heat_load", "exposure_time_minutes",
    "twin_divergence_index", "rf_prediction",
    "rf_deterioration_probability", "current_risk_score",
    "risk_category", "prediction_number"
]

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def finite_number(value):
    if value is None:
        return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def get_first_existing_column(data, names):
    for name in names:
        if name in data.columns:
            return name
    return None

def append_record(path, record, columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = list(columns or record.keys())
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as handle:
                existing_header = next(csv.reader(handle), [])
        except (OSError, csv.Error):
            existing_header = []
        if existing_header != columns:
            legacy_path = path + ".legacy.csv"
            suffix = 1
            while os.path.exists(legacy_path):
                legacy_path = path + f".legacy_{suffix}.csv"
                suffix += 1
            shutil.move(path, legacy_path)
            exists = False
            logger.warning(f"Previous CSV schema differed; preserved old file as {legacy_path}")
    
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in columns})

def calculate_activity(history):
    recent = history[-10:]
    accel_magnitudes = [np.sqrt(row["accel_x"]**2 + row["accel_y"]**2 + row["accel_z"]**2) for row in recent]
    gyro_magnitudes = [np.sqrt(row["gyro_x"]**2 + row["gyro_y"]**2 + row["gyro_z"]**2) for row in recent]

    if len(accel_magnitudes) < 3:
        return 0.0

    accel_activity = float(np.std(accel_magnitudes)) / 2.0
    gyro_activity = float(np.std(gyro_magnitudes)) / 250.0
    combined_activity = 0.7 * accel_activity + 0.3 * gyro_activity
    return float(np.clip(combined_activity, 0.0, 1.0))

def validate_environment_record(record):
    required = [
        "ambient_temperature", "humidity", "body_temperature",
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    ]
    cleaned = {}
    for field in required:
        value = finite_number(record.get(field))
        if value is None:
            return None, f"Missing or invalid {field}"
        cleaned[field] = value

    if not 0 <= cleaned["humidity"] <= 100:
        return None, "Humidity outside 0-100%"
    if not -55 <= cleaned["body_temperature"] <= 125:
        return None, "DS18B20 sensor reading outside supported sensor range"
    if not -50 <= cleaned["ambient_temperature"] <= 100:
        return None, "Ambient temperature outside input range"
    return cleaned, None

def valid_heart_rate(value):
    value = finite_number(value)
    return value if value is not None and 35 <= value <= 220 else None

def valid_spo2(value):
    value = finite_number(value)
    return value if value is not None and 70 <= value <= 100 else None

def load_random_forest():
    if not os.path.exists(RF_MODEL_FILE):
        raise FileNotFoundError(f"Saved Random Forest model not found: {RF_MODEL_FILE}")
    predictor = RandomForestPredictor()
    predictor.load_model(RF_MODEL_FILE)
    logger.info(f"Loaded trained model: {RF_MODEL_FILE}")
    return predictor

# ============================================================
# GLOBAL STATE FOR FLASK
# ============================================================

app = Flask(__name__)

# Suppress standard Werkzeug request logs to keep terminal focused on metrics
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.setLevel(logging.ERROR)

rf_predictor = load_random_forest()

state = {
    "history": [],
    "start_monotonic": time.monotonic(),
    "valid_count": 0,
    "prediction_count": 0,
    "baseline": None,
    "baseline_samples": [],
    "baseline_started_at": None,
    "last_valid_hr": None,
    "last_valid_hr_at": None,
    "last_valid_spo2": None,
    "last_valid_spo2_at": None,
    "last_hr_sequence": None,
    "last_spo2_sequence": None,
    "last_baseline_sequence_pair": None
}

# ============================================================
# FLASK API ENDPOINT
# ============================================================

@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    global state
    
    try:
        record = request.get_json()
        if not record or not isinstance(record, dict):
            return jsonify({"error": "Invalid JSON"}), 400

        # 1. Log Raw JSON
        append_record(RAW_FILE, {
            "received_utc": datetime.now(timezone.utc).isoformat(),
            "raw_json": json.dumps(record),
        })

        # 2. Validate Environment Data
        cleaned, error = validate_environment_record(record)
        if error:
            logger.info(f"Waiting for valid sensor data: {error}")
            return jsonify({"status": "waiting", "message": error}), 200

        now_monotonic = time.monotonic()
        finger_detected = record.get("finger_detected") is True
        sensor_ok = record.get("max_ok", record.get("max30102_ok")) is True

        # 3. Handle Complete Finger Removal
        if not finger_detected or not sensor_ok:
            state["last_valid_hr"] = state["last_valid_hr_at"] = None
            state["last_valid_spo2"] = state["last_valid_spo2_at"] = None
            state["last_hr_sequence"] = None
            state["last_spo2_sequence"] = None

        # 4. Sequence & Freshness Tracking with Carry-Forward
        incoming_hr = valid_heart_rate(record.get("heart_rate"))
        incoming_spo2 = valid_spo2(record.get("spo2"))
        hr_sequence = record.get("heart_rate_sequence")
        spo2_sequence = record.get("spo2_sequence")

        if finger_detected and sensor_ok:
            if incoming_hr is not None:
                state["last_valid_hr"] = incoming_hr
                state["last_valid_hr_at"] = now_monotonic
                if hr_sequence is not None:
                    state["last_hr_sequence"] = hr_sequence
            elif state["last_valid_hr"] is not None and now_monotonic - state["last_valid_hr_at"] <= HEART_RATE_MAX_AGE_SECONDS:
                incoming_hr = state["last_valid_hr"]

            if incoming_spo2 is not None:
                state["last_valid_spo2"] = incoming_spo2
                state["last_valid_spo2_at"] = now_monotonic
                if spo2_sequence is not None:
                    state["last_spo2_sequence"] = spo2_sequence
            elif state["last_valid_spo2"] is not None and now_monotonic - state["last_valid_spo2_at"] <= SPO2_MAX_AGE_SECONDS:
                incoming_spo2 = state["last_valid_spo2"]

        hr_fresh = (
            state["last_valid_hr"] is not None and state["last_valid_hr_at"] is not None
            and now_monotonic - state["last_valid_hr_at"] <= HEART_RATE_MAX_AGE_SECONDS
        )
        spo2_fresh = (
            state["last_valid_spo2"] is not None and state["last_valid_spo2_at"] is not None
            and now_monotonic - state["last_valid_spo2_at"] <= SPO2_MAX_AGE_SECONDS
        )

        if not finger_detected or not sensor_ok or not hr_fresh or not spo2_fresh:
            missing = []
            if not finger_detected: missing.append("finger not detected")
            if not sensor_ok: missing.append("MAX30102 unavailable")
            if not hr_fresh: missing.append("no fresh heart_rate")
            if not spo2_fresh: missing.append("no fresh spo2")
            logger.info(f"Waiting for usable physiological data: {', '.join(missing)}")
            return jsonify({
              "status": "waiting",
              "risk_category": "WAITING",
              "server_alert": False,
              "message": ", ".join(missing)
            }), 200

        # 5. Build Row
        state["valid_count"] += 1
        elapsed_minutes = (now_monotonic - state["start_monotonic"]) / 60.0

        row = {
            "time": time.time(),
            "heart_rate": float(state["last_valid_hr"]),
            "spo2": float(state["last_valid_spo2"]),
            "body_temperature": cleaned["body_temperature"],
            "ambient_temperature": cleaned["ambient_temperature"],
            "humidity": cleaned["humidity"],
            "activity": 0.0,
            "heat_load": 0.0,
            "exposure_time": elapsed_minutes,
            "accel_x": cleaned["accel_x"],
            "accel_y": cleaned["accel_y"],
            "accel_z": cleaned["accel_z"],
            "gyro_x": cleaned["gyro_x"],
            "gyro_y": cleaned["gyro_y"],
            "gyro_z": cleaned["gyro_z"],
        }

        # 6. Check Baseline advancement
        sequence_pair = (hr_sequence, spo2_sequence)
        fresh_both = (incoming_hr is not None and incoming_spo2 is not None)
        
        if state["baseline"] is None and fresh_both:
            state["baseline_samples"].append(row.copy())
            state["last_baseline_sequence_pair"] = sequence_pair
            if state["baseline_started_at"] is None:
                state["baseline_started_at"] = now_monotonic

        # 7. Update History & Feature Derivations
        state["history"].append(row)
        if len(state["history"]) > HISTORY_SIZE:
            state["history"] = state["history"][-HISTORY_SIZE:]

        row["activity"] = calculate_activity(state["history"])
        temperature_load = np.clip((row["ambient_temperature"] - 28.0) / 16.0, 0.0, 1.0)
        humidity_load = np.clip((row["humidity"] - 40.0) / 45.0, 0.0, 1.0)
        row["heat_load"] = float(np.clip(0.55 * temperature_load + 0.45 * humidity_load, 0.0, 1.0))

        state["history"][-1] = row
        if state["baseline"] is None and fresh_both and state["baseline_samples"]:
            state["baseline_samples"][-1] = row.copy()

        # 8. Baseline Collection Check
        baseline_elapsed_minutes = ((time.monotonic() - state["baseline_started_at"]) / 60.0 if state["baseline_started_at"] is not None else 0.0)
        
        if state["baseline"] is None and (len(state["baseline_samples"]) < MIN_BASELINE_SAMPLES or baseline_elapsed_minutes < BASELINE_DURATION_MINUTES):
            logger.info(f"Collecting personal baseline: {baseline_elapsed_minutes:.1f}/{BASELINE_DURATION_MINUTES:g} min | valid samples: {len(state['baseline_samples'])}")
            return jsonify({
              "status": "collecting_baseline",
              "risk_category": "CALIB",
              "activity_heuristic": float(row["activity"]),
              "server_alert": False
            }), 200

        # 9. ML Pipeline Processing
        if state["baseline"] is None:
            baseline_frame = pd.DataFrame(state["baseline_samples"]).copy()
            baseline_builder = PersonalBaseline(baseline_frame, baseline_minutes=len(baseline_frame))
            state["baseline"] = baseline_builder.generate()
            logger.info(f"Personal baseline frozen from {len(baseline_frame)} valid samples over {baseline_elapsed_minutes:.2f} minutes.")
            state["baseline_samples"] = []

            state["baseline"]["activity"] = state["baseline"]["environment"]["activity"]
            state["baseline"]["normal_heart_rate"] = state["baseline"]["heart_rate"]["mean"]
            state["baseline"]["normal_spo2"] = state["baseline"]["spo2"]["mean"]
            state["baseline"]["normal_body_temperature"] = state["baseline"]["body_temperature"]["mean"]

        frame = pd.DataFrame(state["history"])

        # Feature Extraction & Fusion
        feature_extractor = FeatureExtractor(frame.copy(), state["baseline"])
        featured = feature_extractor.extract()

        fusion = MultimodalFusion(featured)
        fused = fusion.fuse()

        for column in ["time", "heart_rate", "spo2", "body_temperature", "ambient_temperature", "humidity", "activity", "heat_load", "exposure_time"]:
            if column not in fused.columns:
                fused[column] = featured[column].to_numpy()

        # Digital Twin
        twin_baseline = state["baseline"].copy()
        twin_baseline["heart_rate"] = state["baseline"]["heart_rate"]["mean"]
        twin_baseline["spo2"] = state["baseline"]["spo2"]["mean"]
        twin_baseline["body_temperature"] = state["baseline"]["body_temperature"]["mean"]
        twin_baseline["activity"] = state["baseline"]["environment"]["activity"]

        divergence_analyzer = TwinDivergenceAnalyzer(fused.copy(), twin_baseline)
        divergence_result = divergence_analyzer.analyze()

        if isinstance(divergence_result, pd.DataFrame):
            divergence_frame = divergence_result
        elif isinstance(divergence_result, list):
            divergence_frame = pd.DataFrame(divergence_result)
        else:
            raise TypeError(f"Unexpected TwinDivergenceAnalyzer output: {type(divergence_result).__name__}")

        divergence_column = get_first_existing_column(divergence_frame, ["twin_divergence_index", "divergence_index", "twin_divergence"])
        fused = fused.copy()
        divergence_values = pd.to_numeric(divergence_frame[divergence_column], errors="coerce").to_numpy()
        fused["twin_divergence_index"] = divergence_values

        # Random Forest Prediction
        model_features = rf_predictor.feature_columns
        latest_features = fused[model_features].tail(1).copy()
        prediction = rf_predictor.predict(latest_features)

        # Risk Assessment
        latest_for_risk = fused.tail(1).copy()
        if "time" not in latest_for_risk.columns:
            latest_for_risk["time"] = row["time"]

        risk_predictor = MADTPRiskPredictor(latest_for_risk)
        risk_result = risk_predictor.predict()

        if isinstance(risk_result, pd.DataFrame):
            risk_row = risk_result.iloc[-1].to_dict()
        elif isinstance(risk_result, list):
            risk_row = risk_result[-1]
        elif isinstance(risk_result, dict):
            risk_row = risk_result

        rf_row = prediction.iloc[-1].to_dict() if isinstance(prediction, pd.DataFrame) else prediction
        probability = rf_row.get("rf_deterioration_probability")
        predicted_class = rf_row.get("rf_prediction")
        current_risk = risk_row.get("current_risk_score")
        risk_category = risk_row.get("risk_category", risk_row.get("current_risk_category", "UNKNOWN"))

        state["prediction_count"] += 1

        output = {
            "received_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_ms": record.get("timestamp_ms"),
            "heart_rate": row["heart_rate"],
            "spo2": row["spo2"],
            "ds18b20_surface_temperature_c": row["body_temperature"],
            "dht22_ambient_temperature_c": row["ambient_temperature"],
            "dht22_humidity_percent": row["humidity"],
            "mpu6050_accel_x": row["accel_x"],
            "mpu6050_accel_y": row["accel_y"],
            "mpu6050_accel_z": row["accel_z"],
            "mpu6050_gyro_x": row["gyro_x"],
            "mpu6050_gyro_y": row["gyro_y"],
            "mpu6050_gyro_z": row["gyro_z"],
            "ambient_temperature": row["ambient_temperature"],
            "humidity": row["humidity"],
            "activity_heuristic": row["activity"],
            "heat_load": row["heat_load"],
            "exposure_time_minutes": row["exposure_time"],
            "twin_divergence_index": fused["twin_divergence_index"].iloc[-1],
            "rf_prediction": predicted_class,
            "rf_deterioration_probability": probability,
            "current_risk_score": current_risk,
            "risk_category": risk_category,
            "prediction_number": state["prediction_count"],
        }

        # 10. Save Results
        append_record(RESULT_FILE, output, RESULT_COLUMNS)

        print("\n" + "-" * 45)
        print(" LIVE PREDICTION")
        print("-" * 45)
        print(f"HR: {row['heart_rate']:.0f} BPM | SpO2: {row['spo2']:.0f}%")
        print(f"RF prediction        : {predicted_class}")
        print(f"RF deterioration prob: {probability}")
        print(f"Twin divergence      : {output['twin_divergence_index']:.4f}")
        print(f"Current risk score   : {current_risk}")
        print(f"Risk category        : {risk_category}")
        print("-" * 45)

        # Trigger hardware buzzer/vibrator if AI predicts deterioration (1) or Risk Category is HIGH
        cat_upper = str(risk_category).upper()
        trigger_hw_alert = bool(
        predicted_class == 1 
        or cat_upper in ["HIGH", "CRITICAL"] 
        or (current_risk is not None and float(current_risk) >= 0.45)
        )

        return jsonify({
            "status": "success",
            "prediction": int(predicted_class) if predicted_class is not None else 0,
            "risk_category": cat_upper,
            "risk_score": float(current_risk) if current_risk is not None else 0.0,
            "activity_heuristic": float(row["activity"]),
            "twin_divergence": float(output["twin_divergence_index"]),
            "server_alert": trigger_hw_alert
        }), 200 

    except Exception as exc:
        logger.error(f"Pipeline stage failed for this sample: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    global state
    try:
        req_data = request.get_json()
        action = req_data.get("action", "CONTINUE_WORKING")
        
        # Check if the live history has collected any data yet
        if not state["history"]:
            return jsonify({"error": "No live data available for simulation yet."}), 400
            
        # Convert the live dictionary list into the required Pandas DataFrame
        current_df = pd.DataFrame(state["history"])
        
        # Run the projection using the live dataframe
        simulator = LiveSimulator(current_df)
        projection = simulator.simulate(action)
        
        return jsonify(projection), 200
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" MADTP — LIVE FLASK INGESTION SERVER")
    print("=" * 65)
    print(f"RF model    : {RF_MODEL_FILE}")
    print(f"Raw log     : {RAW_FILE}")
    print(f"Results     : {RESULT_FILE}")
    print(f"\nListening for ESP32 Wi-Fi JSON on: http://0.0.0.0:5050/api/sensor_data")
    print("Press Ctrl+C to stop.\n")
    
    app.run(host="0.0.0.0", port=5050, threaded=False)