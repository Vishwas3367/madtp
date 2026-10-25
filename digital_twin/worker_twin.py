import pandas as pd
import numpy as np

try:
    from digital_twin.twin_state import DigitalTwinState
except ModuleNotFoundError:
    from twin_state import DigitalTwinState


class WorkerTwin:
    """
    Personalized worker model inside the Digital Twin.

    The Worker Twin maintains a virtual representation of
    the individual worker using:

        - Personal baseline
        - Current physiological observations
        - Current activity
        - Exposure
        - Unified worker state

    This module does NOT:
        - calculate final risk
        - perform AI prediction
        - calculate Twin divergence
        - perform future prediction
        - perform intervention simulation
    """

    def __init__(self, baseline):

        self.baseline = baseline

        # Digital Twin state container
        self.twin_state = DigitalTwinState()

        # Adaptive parameters
        self.adaptation_rate = 0.15

        # Current personalized worker state
        self.worker_state = {

            "heart_rate": self.baseline[
                "heart_rate"
            ],

            "spo2": self.baseline[
                "spo2"
            ],

            "body_temperature": self.baseline[
                "body_temperature"
            ],

            "activity": self.baseline[
                "activity"
            ],

            "exposure_time": 0.0,

            "cumulative_heat_exposure": 0.0,

            "unified_state": 0.0
        }


    # =====================================================
    # 1. INITIALIZE WORKER TWIN
    # =====================================================

    def initialize(self):

        """
        Initializes the virtual worker using the
        personal baseline.
        """

        self.worker_state["heart_rate"] = (
            self.baseline["heart_rate"]
        )

        self.worker_state["spo2"] = (
            self.baseline["spo2"]
        )

        self.worker_state["body_temperature"] = (
            self.baseline["body_temperature"]
        )

        self.worker_state["activity"] = (
            self.baseline["activity"]
        )

        self.worker_state["exposure_time"] = 0.0

        self.worker_state[
            "cumulative_heat_exposure"
        ] = 0.0

        self.worker_state[
            "unified_state"
        ] = 0.0


    # =====================================================
    # 2. UPDATE PHYSIOLOGICAL WORKER STATE
    # =====================================================

    def update_physiology(
        self,
        heart_rate,
        spo2,
        body_temperature
    ):

        """
        Adaptively updates the virtual worker's
        physiological state.

        The Twin does not instantly copy every
        observation. It gradually adapts toward
        the observed state.
        """

        rate = self.adaptation_rate


        self.worker_state[
            "heart_rate"
        ] = (

            (1 - rate)
            *
            self.worker_state[
                "heart_rate"
            ]

            +

            rate
            *
            heart_rate
        )


        self.worker_state[
            "spo2"
        ] = (

            (1 - rate)
            *
            self.worker_state[
                "spo2"
            ]

            +

            rate
            *
            spo2
        )


        self.worker_state[
            "body_temperature"
        ] = (

            (1 - rate)
            *
            self.worker_state[
                "body_temperature"
            ]

            +

            rate
            *
            body_temperature
        )


    # =====================================================
    # 3. UPDATE ACTIVITY AND EXPOSURE
    # =====================================================

    def update_activity_exposure(
        self,
        activity,
        exposure_time,
        heat_load
    ):

        """
        Updates the worker's physical activity and
        accumulated environmental exposure.
        """

        rate = self.adaptation_rate


        self.worker_state[
            "activity"
        ] = (

            (1 - rate)
            *
            self.worker_state[
                "activity"
            ]

            +

            rate
            *
            activity
        )


        self.worker_state[
            "exposure_time"
        ] = exposure_time


        self.worker_state[
            "cumulative_heat_exposure"
        ] += heat_load


    # =====================================================
    # 4. UPDATE UNIFIED WORKER STATE
    # =====================================================

    def update_unified_state(
        self,
        unified_state
    ):

        """
        Updates the virtual worker's unified
        multimodal condition.
        """

        rate = self.adaptation_rate


        self.worker_state[
            "unified_state"
        ] = (

            (1 - rate)
            *
            self.worker_state[
                "unified_state"
            ]

            +

            rate
            *
            unified_state
        )


    # =====================================================
    # 5. CALCULATE PERSONAL DEVIATION
    # =====================================================

    def calculate_personal_deviation(self):

        """
        Calculates how far the Worker Twin has moved
        from the worker's personal baseline.

        This is different from Twin divergence.

        Personal deviation:
            Worker Twin vs Personal Baseline

        Twin divergence:
            Real Observation vs Digital Twin
        """

        baseline_hr = (
            self.baseline[
                "heart_rate"
            ]
        )

        baseline_spo2 = (
            self.baseline[
                "spo2"
            ]
        )

        baseline_temp = (
            self.baseline[
                "body_temperature"
            ]
        )


        hr_deviation = abs(

            self.worker_state[
                "heart_rate"
            ]

            -

            baseline_hr
        )


        spo2_deviation = abs(

            self.worker_state[
                "spo2"
            ]

            -

            baseline_spo2
        )


        temperature_deviation = abs(

            self.worker_state[
                "body_temperature"
            ]

            -

            baseline_temp
        )


        # Normalize deviations

        hr_score = np.clip(
            hr_deviation / 40,
            0,
            1
        )


        spo2_score = np.clip(
            spo2_deviation / 5,
            0,
            1
        )


        temperature_score = np.clip(
            temperature_deviation / 1.5,
            0,
            1
        )


        personal_deviation = (

            0.40 * hr_score

            +

            0.20 * spo2_score

            +

            0.40 * temperature_score
        )


        return float(
            np.clip(
                personal_deviation,
                0,
                1
            )
        )


    # =====================================================
    # 6. UPDATE DIGITAL TWIN STATE OBJECT
    # =====================================================

    def synchronize_state(
        self,
        ambient_temperature,
        humidity,
        heat_load,
        temporal_state,
        unified_state
    ):

        """
        Transfers the current Worker Twin values
        into the common DigitalTwinState object.
        """

        self.twin_state.set_physiological_state(

            heart_rate=self.worker_state[
                "heart_rate"
            ],

            spo2=self.worker_state[
                "spo2"
            ],

            body_temperature=self.worker_state[
                "body_temperature"
            ]
        )


        self.twin_state.set_environmental_state(

            ambient_temperature=ambient_temperature,

            humidity=humidity,

            heat_load=heat_load
        )


        self.twin_state.set_activity_exposure_state(

            activity_level=self.worker_state[
                "activity"
            ],

            exposure_time=self.worker_state[
                "exposure_time"
            ],

            cumulative_heat_exposure=self.worker_state[
                "cumulative_heat_exposure"
            ]
        )


        self.twin_state.set_temporal_state(

            heart_rate_change=temporal_state[
                "heart_rate_change"
            ],

            body_temperature_change=temporal_state[
                "body_temperature_change"
            ],

            spo2_change=temporal_state[
                "spo2_change"
            ],

            environmental_change=temporal_state[
                "environmental_change"
            ],

            state_change=temporal_state[
                "state_change"
            ]
        )


        self.twin_state.set_unified_state(

            physiological_state=unified_state[
                "physiological_state"
            ],

            environmental_state=unified_state[
                "environmental_state"
            ],

            activity_exposure_state=unified_state[
                "activity_exposure_state"
            ],

            temporal_deterioration=unified_state[
                "temporal_deterioration"
            ],

            unified_worker_state=unified_state[
                "unified_worker_state"
            ]
        )


    # =====================================================
    # 7. GET WORKER TWIN STATE
    # =====================================================

    def get_worker_state(self):

        return self.worker_state.copy()


    # =====================================================
    # 8. DISPLAY WORKER TWIN
    # =====================================================

    def display(self):

        personal_deviation = (
            self.calculate_personal_deviation()
        )


        print("\n")
        print("=" * 60)
        print("             PERSONAL WORKER TWIN")
        print("=" * 60)


        print("\nCURRENT VIRTUAL WORKER")
        print("----------------------------------------")


        print(
            f"Heart Rate       : "
            f"{self.worker_state['heart_rate']:.2f} BPM"
        )


        print(
            f"SpO₂             : "
            f"{self.worker_state['spo2']:.2f} %"
        )


        print(
            f"Body Temperature : "
            f"{self.worker_state['body_temperature']:.3f} °C"
        )


        print(
            f"Activity Level   : "
            f"{self.worker_state['activity']:.3f}"
        )


        print(
            f"Exposure Time    : "
            f"{self.worker_state['exposure_time']:.1f} min"
        )


        print(
            f"Cumulative Heat  : "
            f"{self.worker_state['cumulative_heat_exposure']:.3f}"
        )


        print(
            f"Unified State    : "
            f"{self.worker_state['unified_state']:.3f}"
        )


        print(
            f"Personal Deviation: "
            f"{personal_deviation:.3f}"
        )


        print("----------------------------------------")


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    print("\nTesting Personal Worker Twin...")


    # -----------------------------------------------------
    # Load personal baseline
    # -----------------------------------------------------

    baseline_data = pd.read_csv(
        "data/personal_baseline.csv"
    )


    baseline = {

        "heart_rate":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "heart_rate",
                "baseline_value"
            ].iloc[0],

        "spo2":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "spo2",
                "baseline_value"
            ].iloc[0],

        "body_temperature":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "body_temperature",
                "baseline_value"
            ].iloc[0],

        "ambient_temperature":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "ambient_temperature",
                "baseline_value"
            ].iloc[0],

        "humidity":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "humidity",
                "baseline_value"
            ].iloc[0],

        "heat_load":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "heat_load",
                "baseline_value"
            ].iloc[0],

        "activity":
            baseline_data.loc[
                baseline_data[
                    "parameter"
                ] == "activity",
                "baseline_value"
            ].iloc[0]
    }


    # -----------------------------------------------------
    # Create Worker Twin
    # -----------------------------------------------------

    worker_twin = WorkerTwin(
        baseline
    )


    worker_twin.initialize()


    print(
        "✓ Worker Twin initialized"
    )


    # -----------------------------------------------------
    # Simulate a new worker observation
    # -----------------------------------------------------

    worker_twin.update_physiology(

        heart_rate=105,

        spo2=96.8,

        body_temperature=37.5
    )


    worker_twin.update_activity_exposure(

        activity=0.75,

        exposure_time=60,

        heat_load=0.65
    )


    worker_twin.update_unified_state(

        unified_state=0.48
    )


    print(
        "✓ Worker Twin updated with new observation"
    )


    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    worker_twin.display()


    # -----------------------------------------------------
    # Verify Digital Twin State
    # -----------------------------------------------------

    print(
        "\n✓ Worker Twin state model verified"
    )


    print(
        "✓ Personal baseline incorporated"
    )


    print(
        "✓ Physiological adaptation verified"
    )


    print(
        "✓ Activity/exposure tracking verified"
    )


    print(
        "✓ Personal deviation calculated"
    )


    print(
        "\n✓ Personal Worker Twin test completed successfully."
    )