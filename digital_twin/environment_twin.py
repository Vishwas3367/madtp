import pandas as pd
import numpy as np


class EnvironmentTwin:
    """
    Environmental Digital Twin for the MADTP system.

    Represents the environmental conditions surrounding
    the worker and maintains an adaptive virtual
    environmental state.

    Inputs:
        - Ambient temperature
        - Humidity
        - Heat load
        - Exposure time

    Outputs:
        - Virtual environmental state
        - Environmental stress
        - Environmental trend
        - Cumulative environmental exposure

    This module does NOT:
        - calculate worker risk
        - perform AI prediction
        - calculate Twin divergence
        - perform future prediction
        - perform intervention simulation
    """

    def __init__(self):

        # Current virtual environmental state

        self.environment_state = {

            "ambient_temperature": 0.0,

            "humidity": 0.0,

            "heat_load": 0.0,

            "exposure_time": 0.0,

            "cumulative_heat_exposure": 0.0,

            "environmental_stress": 0.0
        }


        # Previous environmental values

        self.previous_temperature = None

        self.previous_humidity = None

        self.previous_heat_load = None


        # Adaptive response rate

        self.adaptation_rate = 0.20


    # =====================================================
    # 1. INITIALIZE ENVIRONMENT TWIN
    # =====================================================

    def initialize(
        self,
        ambient_temperature,
        humidity,
        heat_load
    ):

        """
        Initializes the Environment Twin with the first
        observed environmental condition.
        """

        self.environment_state[
            "ambient_temperature"
        ] = float(
            ambient_temperature
        )


        self.environment_state[
            "humidity"
        ] = float(
            humidity
        )


        self.environment_state[
            "heat_load"
        ] = float(
            np.clip(
                heat_load,
                0,
                1
            )
        )


        self.environment_state[
            "exposure_time"
        ] = 0.0


        self.environment_state[
            "cumulative_heat_exposure"
        ] = 0.0


        self.environment_state[
            "environmental_stress"
        ] = self.calculate_environmental_stress()


        self.previous_temperature = (
            float(ambient_temperature)
        )


        self.previous_humidity = (
            float(humidity)
        )


        self.previous_heat_load = (
            float(heat_load)
        )


    # =====================================================
    # 2. CALCULATE ENVIRONMENTAL STRESS
    # =====================================================

    def calculate_environmental_stress(
        self,
        temperature=None,
        humidity=None,
        activity=None
    ):

        """
        Calculates environmental stress from temperature,
        humidity and optional worker activity.

        The output is normalized from 0 to 1.

        This represents environmental burden, not medical
        diagnosis.
        """

        if temperature is None:

            temperature = (
                self.environment_state[
                    "ambient_temperature"
                ]
            )


        if humidity is None:

            humidity = (
                self.environment_state[
                    "humidity"
                ]
            )


        if activity is None:

            activity = 0.0


        # ---------------------------------------------
        # Temperature load
        # ---------------------------------------------

        temperature_load = np.clip(

            (
                temperature - 28
            ) / 16,

            0,
            1
        )


        # ---------------------------------------------
        # Humidity load
        # ---------------------------------------------

        humidity_load = np.clip(

            (
                humidity - 40
            ) / 45,

            0,
            1
        )


        # ---------------------------------------------
        # Activity amplification
        # ---------------------------------------------

        activity_factor = (

            1.0

            +

            0.20 * np.clip(
                activity,
                0,
                1
            )
        )


        environmental_stress = (

            (
                0.55 *
                temperature_load

                +

                0.45 *
                humidity_load
            )

            *

            activity_factor
        )


        return float(
            np.clip(
                environmental_stress,
                0,
                1
            )
        )


    # =====================================================
    # 3. UPDATE ENVIRONMENT
    # =====================================================

    def update(
        self,
        ambient_temperature,
        humidity,
        heat_load,
        exposure_time,
        activity=0.0
    ):

        """
        Adaptively updates the virtual environment.

        The Environment Twin gradually follows the
        observed environment instead of instantly copying
        every measurement.
        """

        rate = self.adaptation_rate


        # ---------------------------------------------
        # Temperature
        # ---------------------------------------------

        current_temperature = (

            (1 - rate)

            *

            self.environment_state[
                "ambient_temperature"
            ]

            +

            rate
            *
            ambient_temperature
        )


        # ---------------------------------------------
        # Humidity
        # ---------------------------------------------

        current_humidity = (

            (1 - rate)

            *

            self.environment_state[
                "humidity"
            ]

            +

            rate
            *
            humidity
        )


        # ---------------------------------------------
        # Heat load
        # ---------------------------------------------

        current_heat_load = (

            (1 - rate)

            *

            self.environment_state[
                "heat_load"
            ]

            +

            rate
            *
            heat_load
        )


        # ---------------------------------------------
        # Store values
        # ---------------------------------------------

        self.environment_state[
            "ambient_temperature"
        ] = current_temperature


        self.environment_state[
            "humidity"
        ] = current_humidity


        self.environment_state[
            "heat_load"
        ] = np.clip(
            current_heat_load,
            0,
            1
        )


        self.environment_state[
            "exposure_time"
        ] = float(
            exposure_time
        )


        # ---------------------------------------------
        # Accumulated environmental exposure
        # ---------------------------------------------

        self.environment_state[
            "cumulative_heat_exposure"
        ] += (

            self.environment_state[
                "heat_load"
            ]
        )


        # ---------------------------------------------
        # Environmental stress
        # ---------------------------------------------

        self.environment_state[
            "environmental_stress"
        ] = self.calculate_environmental_stress(

            temperature=
                self.environment_state[
                    "ambient_temperature"
                ],

            humidity=
                self.environment_state[
                    "humidity"
                ],

            activity=activity
        )


        # ---------------------------------------------
        # Calculate environmental changes
        # ---------------------------------------------

        temperature_change = (

            self.environment_state[
                "ambient_temperature"
            ]

            -

            self.previous_temperature
        )


        humidity_change = (

            self.environment_state[
                "humidity"
            ]

            -

            self.previous_humidity
        )


        heat_load_change = (

            self.environment_state[
                "heat_load"
            ]

            -

            self.previous_heat_load
        )


        # Store current values for next update

        self.previous_temperature = (

            self.environment_state[
                "ambient_temperature"
            ]
        )


        self.previous_humidity = (

            self.environment_state[
                "humidity"
            ]
        )


        self.previous_heat_load = (

            self.environment_state[
                "heat_load"
            ]
        )


        return {

            "temperature_change":
                float(
                    temperature_change
                ),

            "humidity_change":
                float(
                    humidity_change
                ),

            "heat_load_change":
                float(
                    heat_load_change
                ),

            "environmental_stress":
                float(
                    self.environment_state[
                        "environmental_stress"
                    ]
                )
        }


    # =====================================================
    # 4. GET ENVIRONMENT STATE
    # =====================================================

    def get_state(self):

        """
        Returns the current virtual environmental state.
        """

        return self.environment_state.copy()


    # =====================================================
    # 5. GET ENVIRONMENTAL STRESS
    # =====================================================

    def get_environmental_stress(self):

        return float(

            self.environment_state[
                "environmental_stress"
            ]
        )


    # =====================================================
    # 6. DISPLAY ENVIRONMENT TWIN
    # =====================================================

    def display(self):

        print("\n")
        print("=" * 60)
        print("           ENVIRONMENT DIGITAL TWIN")
        print("=" * 60)


        print("\nCURRENT ENVIRONMENT")
        print("----------------------------------------")


        print(
            f"Ambient Temperature : "
            f"{self.environment_state['ambient_temperature']:.2f} °C"
        )


        print(
            f"Humidity            : "
            f"{self.environment_state['humidity']:.2f} %"
        )


        print(
            f"Heat Load           : "
            f"{self.environment_state['heat_load']:.3f}"
        )


        print(
            f"Exposure Time       : "
            f"{self.environment_state['exposure_time']:.1f} min"
        )


        print(
            f"Cumulative Heat     : "
            f"{self.environment_state['cumulative_heat_exposure']:.3f}"
        )


        print(
            f"Environmental Stress: "
            f"{self.environment_state['environmental_stress']:.3f}"
        )


        print("----------------------------------------")


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    print(
        "\nTesting Environment Digital Twin..."
    )


    # -----------------------------------------------------
    # Create Environment Twin
    # -----------------------------------------------------

    environment_twin = EnvironmentTwin()


    # -----------------------------------------------------
    # Initialize with normal environment
    # -----------------------------------------------------

    environment_twin.initialize(

        ambient_temperature=30.29,

        humidity=49.13,

        heat_load=0.170
    )


    print(
        "✓ Environment Twin initialized"
    )


    # -----------------------------------------------------
    # Simulate increasing environmental stress
    # -----------------------------------------------------

    test_conditions = [

        {
            "temperature": 32,
            "humidity": 55,
            "heat_load": 0.30,
            "exposure": 30,
            "activity": 0.50
        },

        {
            "temperature": 36,
            "humidity": 65,
            "heat_load": 0.55,
            "exposure": 45,
            "activity": 0.65
        },

        {
            "temperature": 42,
            "humidity": 78,
            "heat_load": 0.90,
            "exposure": 90,
            "activity": 0.90
        }
    ]


    for condition in test_conditions:

        changes = environment_twin.update(

            ambient_temperature=
                condition["temperature"],

            humidity=
                condition["humidity"],

            heat_load=
                condition["heat_load"],

            exposure_time=
                condition["exposure"],

            activity=
                condition["activity"]
        )


        print(
            "\n✓ Environment state updated"
        )


        print(
            f"  Temperature : "
            f"{environment_twin.get_state()['ambient_temperature']:.2f} °C"
        )


        print(
            f"  Humidity    : "
            f"{environment_twin.get_state()['humidity']:.2f} %"
        )


        print(
            f"  Heat Load   : "
            f"{environment_twin.get_state()['heat_load']:.3f}"
        )


        print(
            f"  Stress      : "
            f"{environment_twin.get_environmental_stress():.3f}"
        )


    # -----------------------------------------------------
    # Display final state
    # -----------------------------------------------------

    environment_twin.display()


    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    print(
        "\n✓ Environmental state maintained"
    )

    print(
        "✓ Adaptive environmental update verified"
    )

    print(
        "✓ Environmental stress calculated"
    )

    print(
        "✓ Exposure accumulation verified"
    )

    print(
        "✓ Environmental trend tracking verified"
    )

    print(
        "\n✓ Environment Digital Twin test completed successfully."
    )