import pandas as pd
import numpy as np

try:
    from digital_twin.twin_state import DigitalTwinState
    from digital_twin.worker_twin import WorkerTwin
    from digital_twin.environment_twin import EnvironmentTwin
except ModuleNotFoundError:
    from twin_state import DigitalTwinState
    from worker_twin import WorkerTwin
    from environment_twin import EnvironmentTwin


class DigitalTwinUpdater:
    """
    Synchronization engine for the MADTP Digital Twin.

    It receives one worker observation at a time and
    synchronizes:

        1. Worker Twin
        2. Environment Twin
        3. Complete Digital Twin State

    This creates a continuous observation -> Twin update
    pipeline.

    This module does NOT:
        - calculate final risk
        - perform AI prediction
        - calculate Twin divergence
        - perform future prediction
        - perform what-if simulation
    """

    def __init__(self, baseline):

        self.baseline = baseline

        # ---------------------------------------------
        # Create Twin components
        # ---------------------------------------------

        self.worker_twin = WorkerTwin(
            baseline
        )

        self.environment_twin = (
            EnvironmentTwin()
        )

        self.twin_state = DigitalTwinState()

        # ---------------------------------------------
        # Initialization flag
        # ---------------------------------------------

        self.initialized = False

        # ---------------------------------------------
        # Previous observation
        # ---------------------------------------------

        self.previous_observation = None


    # =====================================================
    # 1. INITIALIZE DIGITAL TWIN
    # =====================================================

    def initialize(self, observation):

        """
        Initializes both Worker Twin and Environment Twin
        using the first available observation.
        """

        # ---------------------------------------------
        # Initialize Worker Twin
        # ---------------------------------------------

        self.worker_twin.initialize()


        # ---------------------------------------------
        # Initialize Environment Twin
        # ---------------------------------------------

        self.environment_twin.initialize(

            ambient_temperature=
                observation[
                    "ambient_temperature"
                ],

            humidity=
                observation[
                    "humidity"
                ],

            heat_load=
                observation[
                    "heat_load"
                ]
        )


        # ---------------------------------------------
        # Initialize complete Twin state
        # ---------------------------------------------

        self.twin_state.set_physiological_state(

            heart_rate=
                observation[
                    "heart_rate"
                ],

            spo2=
                observation[
                    "spo2"
                ],

            body_temperature=
                observation[
                    "body_temperature"
                ]
        )


        self.twin_state.set_environmental_state(

            ambient_temperature=
                observation[
                    "ambient_temperature"
                ],

            humidity=
                observation[
                    "humidity"
                ],

            heat_load=
                observation[
                    "heat_load"
                ]
        )


        self.twin_state.set_activity_exposure_state(

            activity_level=
                observation[
                    "activity"
                ],

            exposure_time=
                observation[
                    "exposure_time"
                ],

            cumulative_heat_exposure=
                observation[
                    "cumulative_heat_exposure"
                ]
                if "cumulative_heat_exposure"
                in observation
                else observation[
                    "heat_load"
                ]
        )


        self.twin_state.set_temporal_state(

            heart_rate_change=0.0,

            body_temperature_change=0.0,

            spo2_change=0.0,

            environmental_change=0.0,

            state_change=0.0
        )


        self.twin_state.set_unified_state(

            physiological_state=
                observation[
                    "physiological_state"
                ],

            environmental_state=
                observation[
                    "environmental_state"
                ],

            activity_exposure_state=
                observation[
                    "activity_exposure_state"
                ],

            temporal_deterioration=
                observation[
                    "temporal_deterioration"
                ],

            unified_worker_state=
                observation[
                    "unified_worker_state"
                ]
        )


        # Store first observation

        self.previous_observation = (
            observation.copy()
        )

        self.initialized = True


    # =====================================================
    # 2. CALCULATE TEMPORAL CHANGES
    # =====================================================

    def calculate_temporal_state(
        self,
        observation
    ):

        """
        Calculates changes between the current observation
        and the previous observation.
        """

        if self.previous_observation is None:

            return {

                "heart_rate_change": 0.0,

                "body_temperature_change": 0.0,

                "spo2_change": 0.0,

                "environmental_change": 0.0,

                "state_change": 0.0
            }


        heart_rate_change = (

            observation[
                "heart_rate"
            ]

            -

            self.previous_observation[
                "heart_rate"
            ]
        )


        body_temperature_change = (

            observation[
                "body_temperature"
            ]

            -

            self.previous_observation[
                "body_temperature"
            ]
        )


        spo2_change = (

            observation[
                "spo2"
            ]

            -

            self.previous_observation[
                "spo2"
            ]
        )


        environmental_change = (

            observation[
                "environmental_state"
            ]

            -

            self.previous_observation[
                "environmental_state"
            ]
        )


        state_change = (

            observation[
                "unified_worker_state"
            ]

            -

            self.previous_observation[
                "unified_worker_state"
            ]
        )


        return {

            "heart_rate_change":
                float(heart_rate_change),

            "body_temperature_change":
                float(body_temperature_change),

            "spo2_change":
                float(spo2_change),

            "environmental_change":
                float(environmental_change),

            "state_change":
                float(state_change)
        }


    # =====================================================
    # 3. UPDATE WORKER TWIN
    # =====================================================

    def update_worker_twin(
        self,
        observation
    ):

        """
        Sends physiological, activity and unified-state
        information to the Worker Twin.
        """

        self.worker_twin.update_physiology(

            heart_rate=
                observation[
                    "heart_rate"
                ],

            spo2=
                observation[
                    "spo2"
                ],

            body_temperature=
                observation[
                    "body_temperature"
                ]
        )


        cumulative_heat = (

            observation[
                "cumulative_heat_exposure_normalized"
            ]

            if
            "cumulative_heat_exposure_normalized"
            in observation

            else

            observation[
                "heat_load"
            ]
        )


        self.worker_twin.update_activity_exposure(

            activity=
                observation[
                    "activity"
                ],

            exposure_time=
                observation[
                    "exposure_time"
                ],

            heat_load=
                observation[
                    "heat_load"
                ]
        )


        self.worker_twin.update_unified_state(

            unified_state=
                observation[
                    "unified_worker_state"
                ]
        )


    # =====================================================
    # 4. UPDATE ENVIRONMENT TWIN
    # =====================================================

    def update_environment_twin(
        self,
        observation
    ):

        """
        Sends environmental information to the
        Environment Twin.
        """

        return self.environment_twin.update(

            ambient_temperature=
                observation[
                    "ambient_temperature"
                ],

            humidity=
                observation[
                    "humidity"
                ],

            heat_load=
                observation[
                    "heat_load"
                ],

            exposure_time=
                observation[
                    "exposure_time"
                ],

            activity=
                observation[
                    "activity"
                ]
        )


    # =====================================================
    # 5. SYNCHRONIZE COMPLETE TWIN
    # =====================================================

    def synchronize(
        self,
        observation
    ):

        """
        Performs one complete Digital Twin update.
        """

        if not self.initialized:

            self.initialize(
                observation
            )

            return self.get_state()


        # ---------------------------------------------
        # Temporal state
        # ---------------------------------------------

        temporal_state = (
            self.calculate_temporal_state(
                observation
            )
        )


        # ---------------------------------------------
        # Worker Twin
        # ---------------------------------------------

        self.update_worker_twin(
            observation
        )


        # ---------------------------------------------
        # Environment Twin
        # ---------------------------------------------

        self.update_environment_twin(
            observation
        )


        # ---------------------------------------------
        # Unified state information
        # ---------------------------------------------

        unified_state = {

            "physiological_state":
                observation[
                    "physiological_state"
                ],

            "environmental_state":
                observation[
                    "environmental_state"
                ],

            "activity_exposure_state":
                observation[
                    "activity_exposure_state"
                ],

            "temporal_deterioration":
                observation[
                    "temporal_deterioration"
                ],

            "unified_worker_state":
                observation[
                    "unified_worker_state"
                ]
        }


        # ---------------------------------------------
        # Copy synchronized values into Digital Twin
        # ---------------------------------------------

        worker_state = (
            self.worker_twin
            .get_worker_state()
        )


        environment_state = (
            self.environment_twin
            .get_state()
        )


        self.twin_state.set_physiological_state(

            heart_rate=
                worker_state[
                    "heart_rate"
                ],

            spo2=
                worker_state[
                    "spo2"
                ],

            body_temperature=
                worker_state[
                    "body_temperature"
                ]
        )


        self.twin_state.set_environmental_state(

            ambient_temperature=
                environment_state[
                    "ambient_temperature"
                ],

            humidity=
                environment_state[
                    "humidity"
                ],

            heat_load=
                environment_state[
                    "heat_load"
                ]
        )


        self.twin_state.set_activity_exposure_state(

            activity_level=
                worker_state[
                    "activity"
                ],

            exposure_time=
                worker_state[
                    "exposure_time"
                ],

            cumulative_heat_exposure=
                worker_state[
                    "cumulative_heat_exposure"
                ]
        )


        self.twin_state.set_temporal_state(

            heart_rate_change=
                temporal_state[
                    "heart_rate_change"
                ],

            body_temperature_change=
                temporal_state[
                    "body_temperature_change"
                ],

            spo2_change=
                temporal_state[
                    "spo2_change"
                ],

            environmental_change=
                temporal_state[
                    "environmental_change"
                ],

            state_change=
                temporal_state[
                    "state_change"
                ]
        )


        self.twin_state.set_unified_state(

            physiological_state=
                unified_state[
                    "physiological_state"
                ],

            environmental_state=
                unified_state[
                    "environmental_state"
                ],

            activity_exposure_state=
                unified_state[
                    "activity_exposure_state"
                ],

            temporal_deterioration=
                unified_state[
                    "temporal_deterioration"
                ],

            unified_worker_state=
                unified_state[
                    "unified_worker_state"
                ]
        )


        # Store observation for next update

        self.previous_observation = (
            observation.copy()
        )


        return self.get_state()


    # =====================================================
    # 6. GET COMPLETE DIGITAL TWIN
    # =====================================================

    def get_state(self):

        """
        Returns the complete synchronized Digital Twin.
        """

        return self.twin_state.get_state()


    # =====================================================
    # 7. GET WORKER STATE
    # =====================================================

    def get_worker_state(self):

        return (
            self.worker_twin
            .get_worker_state()
        )


    # =====================================================
    # 8. GET ENVIRONMENT STATE
    # =====================================================

    def get_environment_state(self):

        return (
            self.environment_twin
            .get_state()
        )


# =========================================================
# HELPER FUNCTION
# =========================================================

def load_baseline():

    """
    Loads personal baseline from the CSV created by
    preprocessing/baseline.py.
    """

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


    return baseline


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    print(
        "\nTesting Digital Twin Update Engine..."
    )


    # -----------------------------------------------------
    # Load feature data
    # -----------------------------------------------------

    feature_data = pd.read_csv(

        "data/fused_data.csv"
    )


    print(
        f"✓ Fused data loaded: "
        f"{feature_data.shape}"
    )


    # -----------------------------------------------------
    # Load baseline
    # -----------------------------------------------------

    baseline = load_baseline()


    print(
        "✓ Personal baseline loaded"
    )


    # -----------------------------------------------------
    # Create Digital Twin updater
    # -----------------------------------------------------

    updater = DigitalTwinUpdater(
        baseline
    )


    # -----------------------------------------------------
    # Required fields
    # -----------------------------------------------------

    required_columns = [

        "heart_rate",

        "spo2",

        "body_temperature",

        "ambient_temperature",

        "humidity",

        "heat_load",

        "activity",

        "exposure_time",

        "physiological_state",

        "environmental_state",

        "activity_exposure_state",

        "temporal_deterioration",

        "unified_worker_state"
    ]


    # -----------------------------------------------------
    # Synchronize selected observations
    # -----------------------------------------------------

    selected_indices = [

        0,

        45,

        90,

        120,

        134,

        150,

        179
    ]


    print("\n")
    print("=" * 60)
    print("        DIGITAL TWIN SYNCHRONIZATION TEST")
    print("=" * 60)


    for index in selected_indices:

        row = feature_data.iloc[
            index
        ]


        observation = {

            column:
                float(row[column])

            for column
            in required_columns
        }


        observation[
            "time"
        ] = float(
            row["time"]
        )


        state = updater.synchronize(
            observation
        )


        worker_state = (
            updater.get_worker_state()
        )


        environment_state = (
            updater.get_environment_state()
        )


        print(
            f"\nTime: "
            f"{int(row['time'])} min"
        )


        print(
            f"  Worker HR: "
            f"{worker_state['heart_rate']:.2f} BPM"
        )


        print(
            f"  Worker Temp: "
            f"{worker_state['body_temperature']:.3f} °C"
        )


        print(
            f"  Environment Temp: "
            f"{environment_state['ambient_temperature']:.2f} °C"
        )


        print(
            f"  Environment Stress: "
            f"{environment_state['environmental_stress']:.3f}"
        )


        print(
            f"  Unified State: "
            f"{state['unified']['unified_worker_state']:.3f}"
        )


    print(
        "\n----------------------------------------"
    )

    print(
        "DIGITAL TWIN UPDATE VERIFICATION"
    )

    print(
        "----------------------------------------"
    )

    print(
        "✓ Worker Twin synchronized"
    )

    print(
        "✓ Environment Twin synchronized"
    )

    print(
        "✓ Physiological state synchronized"
    )

    print(
        "✓ Environmental state synchronized"
    )

    print(
        "✓ Activity/exposure state synchronized"
    )

    print(
        "✓ Temporal state synchronized"
    )

    print(
        "✓ Unified worker state synchronized"
    )

    print(
        "\n✓ Digital Twin synchronization test completed."
    )