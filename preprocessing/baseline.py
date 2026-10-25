import pandas as pd
import numpy as np


class PersonalBaseline:
    """
    Generates a personalized physiological baseline
    for the wearable worker.

    This module ONLY creates the baseline.
    It does not perform:
        - AI prediction
        - Digital Twin processing
        - risk calculation
        - future prediction
        - intervention simulation
    """

    def __init__(self, data, baseline_minutes=1):

        self.data = data.copy()
        self.baseline_minutes = baseline_minutes

        self.baseline = {}


    # =====================================================
    # 1. SELECT BASELINE PERIOD
    # =====================================================

    def select_baseline_period(self):

        """
        Uses the initial period of data as the worker's
        normal/reference condition.

        For the virtual prototype:
            first N valid samples = baseline period; callers must map elapsed time to samples
        """

        if len(self.data) < self.baseline_minutes:

            raise ValueError(
                "Not enough data to generate baseline."
            )

        baseline_data = self.data.iloc[
            :self.baseline_minutes
        ].copy()

        return baseline_data


    # =====================================================
    # 2. CALCULATE PHYSIOLOGICAL BASELINE
    # =====================================================

    def calculate_physiological_baseline(
        self,
        baseline_data
    ):

        self.baseline["heart_rate"] = {
            "mean":
                baseline_data["heart_rate"].mean(),

            "std":
                baseline_data["heart_rate"].std(),

            "minimum":
                baseline_data["heart_rate"].min(),

            "maximum":
                baseline_data["heart_rate"].max()
        }


        self.baseline["spo2"] = {
            "mean":
                baseline_data["spo2"].mean(),

            "std":
                baseline_data["spo2"].std(),

            "minimum":
                baseline_data["spo2"].min(),

            "maximum":
                baseline_data["spo2"].max()
        }


        self.baseline["body_temperature"] = {
            "mean":
                baseline_data[
                    "body_temperature"
                ].mean(),

            "std":
                baseline_data[
                    "body_temperature"
                ].std(),

            "minimum":
                baseline_data[
                    "body_temperature"
                ].min(),

            "maximum":
                baseline_data[
                    "body_temperature"
                ].max()
        }


    # =====================================================
    # 3. CALCULATE ENVIRONMENTAL BASELINE
    # =====================================================

    def calculate_environmental_baseline(
        self,
        baseline_data
    ):

        self.baseline["environment"] = {

            "ambient_temperature":
                baseline_data[
                    "ambient_temperature"
                ].mean(),

            "humidity":
                baseline_data[
                    "humidity"
                ].mean(),

            "heat_load":
                baseline_data[
                    "heat_load"
                ].mean(),

            "activity":
                baseline_data[
                    "activity"
                ].mean()
        }


    # =====================================================
    # 4. CREATE PERSONAL REFERENCE VALUES
    # =====================================================

    def create_reference_values(self):

        """
        Extracts the central personal values that
        downstream MADTP modules can directly use.
        """

        self.baseline["reference"] = {

            "normal_heart_rate":
                self.baseline[
                    "heart_rate"
                ]["mean"],

            "normal_spo2":
                self.baseline[
                    "spo2"
                ]["mean"],

            "normal_body_temperature":
                self.baseline[
                    "body_temperature"
                ]["mean"],

            "normal_ambient_temperature":
                self.baseline[
                    "environment"
                ]["ambient_temperature"],

            "normal_humidity":
                self.baseline[
                    "environment"
                ]["humidity"],

            "normal_heat_load":
                self.baseline[
                    "environment"
                ]["heat_load"],

            "normal_activity":
                self.baseline[
                    "environment"
                ]["activity"]
        }


    # =====================================================
    # 5. COMPLETE BASELINE GENERATION
    # =====================================================

    def generate(self):

        print("\n")
        print("=" * 60)
        print("        PERSONAL BASELINE GENERATION")
        print("=" * 60)


        # Select initial normal period

        baseline_data = (
            self.select_baseline_period()
        )

        print(
            f"\n✓ Baseline period: "
            f"first {self.baseline_minutes} samples"
        )


        # Physiological baseline

        self.calculate_physiological_baseline(
            baseline_data
        )

        print(
            "✓ Physiological baseline calculated"
        )


        # Environmental baseline

        self.calculate_environmental_baseline(
            baseline_data
        )

        print(
            "✓ Environmental baseline calculated"
        )


        # Reference values

        self.create_reference_values()

        print(
            "✓ Personal reference values created"
        )


        return self.baseline


    # =====================================================
    # 6. DISPLAY BASELINE
    # =====================================================

    def display_baseline(self):

        reference = self.baseline[
            "reference"
        ]


        print("\n")
        print("----------------------------------------")
        print("PERSONAL WORKER BASELINE")
        print("----------------------------------------")


        print(
            f"Normal Heart Rate       : "
            f"{reference['normal_heart_rate']:.2f} BPM"
        )

        print(
            f"Normal SpO₂             : "
            f"{reference['normal_spo2']:.2f} %"
        )

        print(
            f"Normal Body Temperature : "
            f"{reference['normal_body_temperature']:.3f} °C"
        )

        print(
            f"Normal Ambient Temp     : "
            f"{reference['normal_ambient_temperature']:.2f} °C"
        )

        print(
            f"Normal Humidity         : "
            f"{reference['normal_humidity']:.2f} %"
        )

        print(
            f"Normal Heat Load        : "
            f"{reference['normal_heat_load']:.3f}"
        )

        print(
            f"Normal Activity         : "
            f"{reference['normal_activity']:.3f}"
        )


        print("----------------------------------------")


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    input_file = (
        "data/filtered_data.csv"
    )


    try:

        # -----------------------------------------
        # Load filtered sensor data
        # -----------------------------------------

        data = pd.read_csv(
            input_file
        )


        print(
            "\nInput data loaded:"
        )

        print(
            data.shape
        )


        # -----------------------------------------
        # Create baseline generator
        # -----------------------------------------

        baseline_generator = PersonalBaseline(

            data,

            baseline_minutes=1
        )


        # -----------------------------------------
        # Generate baseline
        # -----------------------------------------

        baseline = (
            baseline_generator.generate()
        )


        # -----------------------------------------
        # Display baseline
        # -----------------------------------------

        baseline_generator.display_baseline()


        # -----------------------------------------
        # Save baseline
        # -----------------------------------------

        reference = baseline[
            "reference"
        ]


        baseline_output = pd.DataFrame({

            "parameter": [
                "heart_rate",
                "spo2",
                "body_temperature",
                "ambient_temperature",
                "humidity",
                "heat_load",
                "activity"
            ],

            "baseline_value": [

                reference[
                    "normal_heart_rate"
                ],

                reference[
                    "normal_spo2"
                ],

                reference[
                    "normal_body_temperature"
                ],

                reference[
                    "normal_ambient_temperature"
                ],

                reference[
                    "normal_humidity"
                ],

                reference[
                    "normal_heat_load"
                ],

                reference[
                    "normal_activity"
                ]
            ]
        })


        baseline_output.to_csv(
            "data/personal_baseline.csv",
            index=False
        )


        print(
            "\n✓ Baseline saved to:"
        )

        print(
            "data/personal_baseline.csv"
        )


    except FileNotFoundError:

        print(
            "\nERROR:"
        )

        print(
            "data/filtered_data.csv "
            "was not found."
        )

        print(
            "Run filtering.py first."
        )