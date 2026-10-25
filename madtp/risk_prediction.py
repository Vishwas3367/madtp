import pandas as pd
import numpy as np


class MADTPRiskPredictor:
    """
    MADTP V4 Production Risk Predictor
    
    Integrates bounded feature extraction, dynamic thresholds, and AI probabilities.
    Replaces static manual weights with the optimized V4 mathematical formulation.
    """

    def __init__(self, data):
        self.data = data.copy()

    # =====================================================
    # 1. VALIDATE INPUT FEATURES
    # =====================================================
    def validate_features(self):
        required_columns = [
            "time",
            "physiological_state",
            "environmental_state",
            "activity_exposure_state",
            "temporal_deterioration",
            "twin_divergence_index"
        ]

        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required risk features: {', '.join(missing_columns)}")

    # =====================================================
    # 2. V4 STRICT MATHEMATICAL BOUNDS
    # =====================================================
    @staticmethod
    def clip01(value):
        """Prevents sensor anomalies from causing mathematical overflow."""
        return float(np.clip(np.asarray(value, dtype=float), 0.0, 1.0))

    # =====================================================
    # 3. V4 COMPLETE RISK PIPELINE
    # =====================================================
    def predict(self):
        print("\n" + "=" * 60)
        print("          MADTP V4 CURRENT RISK ENGINE")
        print("=" * 60)

        self.validate_features()
        print("✓ Required risk features verified")

        # Safely check if the Random Forest model has already appended its probability
        has_ai = "rf_deterioration_probability" in self.data.columns
        
        results = []

        for _, row in self.data.iterrows():
            # 1. Extract and Bound Base Features
            P = self.clip01(row.get("physiological_state", 0.0))
            E = self.clip01(row.get("environmental_state", 0.0))
            A = self.clip01(row.get("activity_exposure_state", 0.0))
            T = self.clip01(row.get("temporal_deterioration", 0.0))
            D = self.clip01(row.get("twin_divergence_index", 0.0))
            AI = self.clip01(row.get("rf_deterioration_probability", 0.0) if has_ai else 0.0)

            # 2. V4 Bounded Interaction
            # Square root prevents collapse if one variable is low
            interaction = self.clip01(np.sqrt(P * E))

            # 3. V4 Adaptive Pressure
            adaptive_pressure = self.clip01(
                0.35 * P + 
                0.25 * E + 
                0.20 * A + 
                0.20 * D
            )

            # 4. V4 Adaptive Threshold
            # Threshold becomes stricter (lower) in hazardous environments
            adaptive_threshold = 0.50 - (0.08 * E) - (0.05 * A) - (0.04 * D)
            adaptive_threshold = float(np.clip(adaptive_threshold, 0.30, 0.55))

            # 5. V4 Final Score Blending
            if has_ai:
                final_score = self.clip01((0.85 * AI) + (0.15 * adaptive_pressure))
            else:
                final_score = adaptive_pressure

            risk_margin = final_score - adaptive_threshold

            # 6. Risk Categorization
            if final_score < 0.25:
                category = "NORMAL"
            elif final_score < 0.40:
                category = "WATCH"
            elif final_score < 0.60:
                category = "MODERATE"
            elif final_score < 0.85:
                category = "HIGH"
            else:
                category = "CRITICAL"

            # 7. Update Row with V4 Metrics
            updated_row = row.copy()
            updated_row["current_risk_score"] = final_score * 100  # Scale 0-100 for UI and existing systems
            updated_row["risk_margin"] = risk_margin
            updated_row["adaptive_threshold"] = adaptive_threshold
            updated_row["v4_adaptive_pressure"] = adaptive_pressure
            updated_row["v4_interaction"] = interaction
            updated_row["risk_category"] = category
            
            results.append(updated_row)

        self.data = pd.DataFrame(results)
        print("✓ V4 bounded features and interactions extracted")
        print("✓ V4 adaptive thresholds calculated")
        print("✓ V4 final risk score and categories generated")

        # 8. Risk Trend Calculation (for batch data arrays)
        if len(self.data) > 1:
            self.data["risk_change"] = self.data["current_risk_score"].diff().fillna(0.0)
            self.data["risk_direction"] = np.where(
                self.data["risk_change"] > 0.5, "INCREASING",
                np.where(self.data["risk_change"] < -0.5, "DECREASING", "STABLE")
            )
            print("✓ Risk trend velocity calculated")
        else:
            self.data["risk_change"] = 0.0
            self.data["risk_direction"] = "STABLE"

        print("\nCurrent risk prediction completed.")
        return self.data


# =========================================================
# TEST MODULE
# =========================================================
if __name__ == "__main__":
    input_file = "data/twin_divergence_data.csv"
    output_file = "data/risk_prediction_data.csv"

    try:
        # Load Twin divergence data
        data = pd.read_csv(input_file)
        print("\nInput divergence data loaded:")
        print(data.shape)

        # Create risk predictor
        predictor = MADTPRiskPredictor(data)

        # Calculate risk
        result = predictor.predict()

        # Display results
        display_columns = [
            "time",
            "physiological_state",
            "environmental_state",
            "twin_divergence_index",
            "v4_adaptive_pressure",
            "adaptive_threshold",
            "current_risk_score",
            "risk_category",
            "risk_direction"
        ]

        print("\n----------------------------------------")
        print("MADTP V4 CURRENT RISK RESULTS")
        print("----------------------------------------")
        print(result[display_columns].head(15))

        # Save result
        result.to_csv(output_file, index=False)
        print(f"\n✓ Risk prediction data saved to: {output_file}")

        # Summary
        print("\n----------------------------------------")
        print("V4 RISK SUMMARY")
        print("----------------------------------------")
        print(f"Minimum risk : {result['current_risk_score'].min():.2f}")
        print(f"Maximum risk : {result['current_risk_score'].max():.2f}")
        print(f"Average risk : {result['current_risk_score'].mean():.2f}")
        print("\nRisk categories:")
        print(result["risk_category"].value_counts())

    except FileNotFoundError:
        print("\nERROR: data/twin_divergence_data.csv was not found.")
        print("Run twin_divergence.py first.")