import pandas as pd
import numpy as np


class FeatureExtractor:
    """
    Feature extraction module for MADTP.

    Converts filtered wearable data into meaningful
    physiological, environmental, activity and temporal
    features.

    This module does NOT:
        - perform AI prediction
        - create the Digital Twin
        - calculate final risk
        - perform future prediction
        - perform what-if simulation
    """

    def __init__(self, data, baseline):

        self.data = data.copy()

        self.baseline = baseline


    # =====================================================
    # 1. PHYSIOLOGICAL FEATURES
    # =====================================================

    def extract_physiological_features(self):

        normal_hr = self.baseline[
            "heart_rate"
        ]["mean"]

        normal_temp = self.baseline[
            "body_temperature"
        ]["mean"]

        normal_spo2 = self.baseline[
            "spo2"
        ]["mean"]


        # ---------------------------------------------
        # Deviation from personal baseline
        # ---------------------------------------------

        self.data["hr_deviation"] = (

            self.data["heart_rate"]
            - normal_hr
        )


        self.data["temperature_deviation"] = (

            self.data["body_temperature"]
            - normal_temp
        )


        self.data["spo2_deviation"] = (

            normal_spo2
            - self.data["spo2"]
        )


        # ---------------------------------------------
        # Percentage deviation
        # ---------------------------------------------

        self.data["hr_deviation_pct"] = (

            self.data["hr_deviation"]
            / normal_hr
            * 100
        )


        self.data["temperature_deviation_pct"] = (

            self.data["temperature_deviation"]
            / normal_temp
            * 100
        )


        self.data["spo2_deviation_pct"] = (

            self.data["spo2_deviation"]
            / normal_spo2
            * 100
        )


    # =====================================================
    # 2. ENVIRONMENTAL FEATURES
    # =====================================================

    def extract_environmental_features(self):

        # ---------------------------------------------
        # Temperature load
        # ---------------------------------------------

        self.data["temperature_load"] = (

            (
                self.data[
                    "ambient_temperature"
                ] - 28
            ) / 16
        ).clip(0, 1)


        # ---------------------------------------------
        # Humidity load
        # ---------------------------------------------

        self.data["humidity_load"] = (

            (
                self.data[
                    "humidity"
                ] - 40
            ) / 45
        ).clip(0, 1)


        # ---------------------------------------------
        # Combined environmental load
        # ---------------------------------------------

        self.data["environmental_load"] = (

            0.55 *
            self.data["temperature_load"]

            +

            0.45 *
            self.data["humidity_load"]
        ).clip(0, 1)


        # ---------------------------------------------
        # Environmental change
        # ---------------------------------------------

        self.data["ambient_temperature_change"] = (

            self.data[
                "ambient_temperature"
            ].diff()
        )


        self.data["humidity_change"] = (

            self.data[
                "humidity"
            ].diff()
        )


    # =====================================================
    # 3. ACTIVITY FEATURES
    # =====================================================

    def extract_activity_features(self):

        # ---------------------------------------------
        # Activity intensity
        # ---------------------------------------------

        self.data["activity_intensity"] = (

            self.data["activity"]
            .clip(0, 1)
        )


        # ---------------------------------------------
        # High activity indicator
        # ---------------------------------------------

        self.data["high_activity"] = (

            self.data["activity"]
            >= 0.70
        ).astype(int)


        # ---------------------------------------------
        # Cumulative activity exposure
        # ---------------------------------------------

        self.data["cumulative_activity"] = (

            self.data["activity"]
            .cumsum()
        )


    # =====================================================
    # 4. EXPOSURE FEATURES
    # =====================================================

    def extract_exposure_features(self):

        # ---------------------------------------------
        # Normalized exposure
        # ---------------------------------------------

        self.data["exposure_load"] = (

            self.data[
                "exposure_time"
            ] / 180
        ).clip(0, 1)


        # ---------------------------------------------
        # Heat exposure accumulation
        # ---------------------------------------------

        self.data["cumulative_heat_exposure"] = (

            self.data[
                "environmental_load"
            ].cumsum()
        )


        # Normalize cumulative exposure

        maximum = (

            self.data[
                "cumulative_heat_exposure"
            ].max()
        )


        if maximum > 0:

            self.data[
                "cumulative_heat_exposure_normalized"
            ] = (

                self.data[
                    "cumulative_heat_exposure"
                ]

                / maximum
            )

        else:

            self.data[
                "cumulative_heat_exposure_normalized"
            ] = 0


    # =====================================================
    # 5. TEMPORAL / TREND FEATURES
    # =====================================================

    def extract_temporal_features(self):

        # ---------------------------------------------
        # Physiological rate of change
        # ---------------------------------------------

        self.data["hr_rate_change"] = (

            self.data[
                "heart_rate"
            ].diff()
        )


        self.data["body_temp_rate_change"] = (

            self.data[
                "body_temperature"
            ].diff()
        )


        self.data["spo2_rate_change"] = (

            self.data[
                "spo2"
            ].diff()
        )


        # ---------------------------------------------
        # Short-term rolling trends
        # ---------------------------------------------

        self.data["hr_trend"] = (

            self.data[
                "heart_rate"
            ]

            -

            self.data[
                "heart_rate"
            ].rolling(
                window=5,
                min_periods=1
            ).mean()
        )


        self.data["temperature_trend"] = (

            self.data[
                "body_temperature"
            ]

            -

            self.data[
                "body_temperature"
            ].rolling(
                window=5,
                min_periods=1
            ).mean()
        )


        self.data["environment_trend"] = (

            self.data[
                "environmental_load"
            ]

            -

            self.data[
                "environmental_load"
            ].rolling(
                window=5,
                min_periods=1
            ).mean()
        )


    # =====================================================
    # 6. MULTIMODAL FEATURE GROUPS
    # =====================================================

    def create_feature_groups(self):

        """
        Creates normalized feature groups.

        These groups will later be supplied to
        multimodal_fusion.py.
        """

        # ---------------------------------------------
        # Physiological state
        # ---------------------------------------------

        self.data["physiological_state"] = (

            0.45 *
            np.clip(
                self.data[
                    "hr_deviation"
                ] / 45,
                0,
                1
            )

            +

            0.35 *
            np.clip(
                self.data[
                    "temperature_deviation"
                ] / 1.5,
                0,
                1
            )

            +

            0.20 *
            np.clip(
                self.data[
                    "spo2_deviation"
                ] / 5,
                0,
                1
            )
        ).clip(0, 1)


        # ---------------------------------------------
        # Environmental state
        # ---------------------------------------------

        self.data["environmental_state"] = (

            self.data[
                "environmental_load"
            ]
        )


        # ---------------------------------------------
        # Activity-exposure state
        # ---------------------------------------------

        self.data["activity_exposure_state"] = (

            0.50 *
            self.data[
                "activity_intensity"
            ]

            +

            0.50 *
            self.data[
                "exposure_load"
            ]
        ).clip(0, 1)


        # ---------------------------------------------
        # Temporal deterioration state
        # ---------------------------------------------

        hr_trend = np.clip(
            self.data[
                "hr_rate_change"
            ].fillna(0) / 5,
            -1,
            1
        )

        temp_trend = np.clip(
            self.data[
                "body_temp_rate_change"
            ].fillna(0) / 0.2,
            -1,
            1
        )

        environment_trend = np.clip(
            self.data[
                "ambient_temperature_change"
            ].fillna(0) / 3,
            -1,
            1
        )


        # Convert negative/positive trend
        # into a 0-1 deterioration scale.

        self.data["temporal_deterioration"] = (

            0.40 *
            ((hr_trend + 1) / 2)

            +

            0.35 *
            ((temp_trend + 1) / 2)

            +

            0.25 *
            ((environment_trend + 1) / 2)

        ).clip(0, 1)


    # =====================================================
    # 7. COMPLETE FEATURE EXTRACTION
    # =====================================================

    def extract(self):

        print("\n")
        print("=" * 60)
        print("        MULTIMODAL FEATURE EXTRACTION")
        print("=" * 60)


        # Physiological

        self.extract_physiological_features()

        print(
            "\n✓ Physiological features extracted"
        )


        # Environmental

        self.extract_environmental_features()

        print(
            "✓ Environmental features extracted"
        )


        # Activity

        self.extract_activity_features()

        print(
            "✓ Activity features extracted"
        )


        # Exposure

        self.extract_exposure_features()

        print(
            "✓ Exposure features extracted"
        )


        # Temporal

        self.extract_temporal_features()

        print(
            "✓ Temporal trend features extracted"
        )


        # Feature groups

        self.create_feature_groups()

        print(
            "✓ Multimodal feature groups created"
        )


        # Fill initial NaN values generated
        # by difference operations.

        self.data = (

            self.data
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


        print(
            "\nFeature extraction completed."
        )

        print(
            f"Total features available: "
            f"{len(self.data.columns)}"
        )


        return self.data


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    input_file = (
        "data/filtered_data.csv"
    )

    output_file = (
        "data/feature_data.csv"
    )


    try:

        # -----------------------------------------
        # Load filtered data
        # -----------------------------------------

        data = pd.read_csv(
            input_file
        )


        # -----------------------------------------
        # Load personal baseline
        # -----------------------------------------

        baseline_data = pd.read_csv(
            "data/personal_baseline.csv"
        )


        # Convert saved baseline CSV
        # into the structure required
        # by this module.

        baseline = {

            "heart_rate": {
                "mean":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "heart_rate",
                        "baseline_value"
                    ].iloc[0]
            },

            "spo2": {
                "mean":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "spo2",
                        "baseline_value"
                    ].iloc[0]
            },

            "body_temperature": {
                "mean":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "body_temperature",
                        "baseline_value"
                    ].iloc[0]
            },

            "environment": {

                "ambient_temperature":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "ambient_temperature",
                        "baseline_value"
                    ].iloc[0],

                "humidity":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "humidity",
                        "baseline_value"
                    ].iloc[0],

                "heat_load":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "heat_load",
                        "baseline_value"
                    ].iloc[0],

                "activity":
                    baseline_data.loc[
                        baseline_data[
                            "parameter"
                        ]
                        == "activity",
                        "baseline_value"
                    ].iloc[0]
            }
        }


        print(
            "\nInput data loaded:"
        )

        print(
            data.shape
        )


        # -----------------------------------------
        # Create feature extractor
        # -----------------------------------------

        extractor = FeatureExtractor(

            data,

            baseline
        )


        # -----------------------------------------
        # Extract features
        # -----------------------------------------

        feature_data = (
            extractor.extract()
        )


        # -----------------------------------------
        # Save feature dataset
        # -----------------------------------------

        feature_data.to_csv(
            output_file,
            index=False
        )


        # -----------------------------------------
        # Display important features
        # -----------------------------------------

        important_features = [

            "hr_deviation",

            "temperature_deviation",

            "spo2_deviation",

            "environmental_load",

            "activity_intensity",

            "exposure_load",

            "cumulative_heat_exposure_normalized",

            "hr_rate_change",

            "body_temp_rate_change",

            "physiological_state",

            "environmental_state",

            "activity_exposure_state",

            "temporal_deterioration"
        ]


        print(
            "\n----------------------------------------"
        )

        print(
            "IMPORTANT MADTP FEATURES"
        )

        print(
            "----------------------------------------"
        )


        print(
            feature_data[
                important_features
            ].head(10)
        )


        print(
            "\n✓ Feature data saved to:"
        )

        print(
            output_file
        )


    except FileNotFoundError as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )

        print(
            "\nMake sure you have already run:"
        )

        print(
            "1. data_simulator.py"
        )

        print(
            "2. data_cleaning.py"
        )

        print(
            "3. filtering.py"
        )

        print(
            "4. baseline.py"
        )