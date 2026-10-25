"""MADTP project launcher. Use explicit modes to avoid accidentally running
training, evaluation, and dashboard scripts as if they were one live pipeline.
"""
from __future__ import annotations
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def audit() -> int:
    required = [
        "ai/random_forest_model.pkl", "api_server.py", "preprocessing/baseline.py",
        "preprocessing/feature_extraction.py", "madtp/multimodal_fusion.py",
        "madtp/risk_prediction.py", "madtp/twin_divergence.py",
        "digital_twin/twin_update.py", "live_madtp.py",
        "data/simulated_sensor_data.csv",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    print(f"MADTP project root: {ROOT}")
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("Core project files: OK")
    print("Python version:", sys.version.split()[0])
    print("Note: run `python main.py --mode live` only when the ESP32 is connected.")
    print("Note: run `python main.py --mode batch` only when you intentionally want the legacy multi-script batch workflow.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MADTP wearable project launcher")
    parser.add_argument("--mode", choices=["audit", "live", "batch"], default="audit",
                        help="audit files (default), run live ESP32 pipeline, or run legacy batch scripts")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode == "audit":
        return audit()
    if args.mode == "live":
        runpy.run_path(str(ROOT / "live_madtp.py"), run_name="__main__")
        return 0
    print("WARNING: legacy batch mode executes many scripts, including training/evaluation scripts.")
    print("It is not the live hardware pipeline and may overwrite generated CSV outputs.")
    confirm = input("Type RUN-BATCH to continue: ").strip()
    if confirm != "RUN-BATCH":
        print("Cancelled.")
        return 2
    runpy.run_path(str(ROOT / "main_legacy_batch.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
