import pandas as pd
from madtp.risk_prediction import MADTPRiskPredictor

class LiveSimulator:
    """
    On-demand Future Simulation Engine.
    Projects worker risk +20 minutes into the future based on supervisor interventions.
    """
    
    def __init__(self, current_live_data):
        # Safely extract the absolute latest row of live sensor data
        if isinstance(current_live_data, pd.DataFrame):
            self.base_state = current_live_data.iloc[-1:].copy()
        else:
            self.base_state = pd.DataFrame([current_live_data])

    def apply_intervention(self, action):
        """Mathematically alters the future state based on the selected action."""
        simulated_state = self.base_state.copy()
        
        if action == "REST_COOLING":
            # Heart rate drops, activity goes to 0, environmental load drops
            simulated_state["heart_rate"] = max(70, simulated_state["heart_rate"].iloc[0] - 20)
            simulated_state["activity_exposure_state"] = 0.0
            simulated_state["environmental_state"] = max(0.0, simulated_state["environmental_state"].iloc[0] - 0.25)
            
        elif action == "HYDRATION":
            # SpO2 improves, body temp drops slightly, activity remains the same
            simulated_state["spo2"] = min(100, simulated_state["spo2"].iloc[0] + 2)
            simulated_state["physiological_state"] = max(0.0, simulated_state["physiological_state"].iloc[0] - 0.15)
            
        elif action == "CONTINUE_WORKING":
            # Physiological strain and temporal deterioration increase
            simulated_state["physiological_state"] = min(1.0, simulated_state["physiological_state"].iloc[0] + 0.20)
            simulated_state["temporal_deterioration"] = min(1.0, simulated_state["temporal_deterioration"].iloc[0] + 0.15)

        return simulated_state

    def simulate(self, action):
        # Apply the hypothetical changes
        future_state = self.apply_intervention(action)
        
        # Run the modified state through the live V4 Predictor
        predictor = MADTPRiskPredictor(future_state)
        result = predictor.predict()
        
        future_risk = result["current_risk_score"].iloc[0]
        future_category = result["risk_category"].iloc[0]
        
        return {
            "intervention": action,
            "projected_risk_score": float(future_risk),
            "projected_category": str(future_category),
            "recommendation": "Safe to return to work" if future_risk < 40.0 else "Extended rest required"
        }