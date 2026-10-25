import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_patent_plots(csv_path, label_prefix):
    print(f"\nAnalyzing {label_prefix.upper()} data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    time_col = "time" if "time" in df.columns else "exposure_time_minutes"
    risk_col = "final_madtp_score" if "final_madtp_score" in df.columns else "current_risk_score"
    rf_col = "rf_deterioration_probability"
    
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. RISK SCORE VS ADAPTIVE THRESHOLD PLOT
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col], df[risk_col], label="MADTP Score", color="#1f77b4")
    
    if "adaptive_threshold" in df.columns:
        plt.plot(df[time_col], df["adaptive_threshold"], label="Adaptive Threshold", linestyle="--", color="#ff7f0e")
    else:
        limit = 50.0 if df[risk_col].max() > 1.5 else 0.5
        plt.axhline(y=limit, color="#ff7f0e", linestyle="--", label="Safety Threshold")

    plt.xlabel("Time (minutes)")
    plt.ylabel("Risk Score")
    plt.title(f"MADTP Worker Risk Trend ({label_prefix.title()} Data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    risk_plot_path = os.path.join(OUTPUT_DIR, f"{label_prefix}_risk_plot.png")
    plt.savefig(risk_plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {risk_plot_path}")

    # 2. AI DETERIORATION PROBABILITY PLOT
    if rf_col in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df[time_col], df[rf_col], label="RF Deterioration Probability", color="#1f77b4")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Probability")
        plt.title(f"AI Future Deterioration Prediction ({label_prefix.title()} Data)")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        prediction_plot_path = os.path.join(OUTPUT_DIR, f"{label_prefix}_prediction_plot.png")
        plt.savefig(prediction_plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {prediction_plot_path}")

if __name__ == "__main__":
    # Generate charts for the extreme condition simulation
    if os.path.exists("data/madtp_final_results.csv"):
        generate_patent_plots("data/madtp_final_results.csv", "simulated")
        
    # Generate charts for the live ESP32 physical prototype
    if os.path.exists("data/live_madtp_results.csv"):
        generate_patent_plots("data/live_madtp_results.csv", "hardware")