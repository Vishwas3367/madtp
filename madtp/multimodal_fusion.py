import pandas as pd
import numpy as np


class MultimodalFusion:
    """
    Multimodal feature fusion for the MADTP system.

    Combines four information domains:

        1. Physiological state
        2. Environmental state
        3. Activity / exposure state
        4. Temporal deterioration state

    Output:
        Unified adaptive worker state

    This module does NOT:
        - create the Digital Twin
        - calculate Twin divergence
        - perform AI prediction
        - perform future prediction
        - perform intervention simulation
    """

    def __init__(self, data):

        self.data = data.copy()

        self.required_features = [

            "physiological_state",

            "environmental_state",

            "activity_exposure_state",

            "temporal_deterioration"
        ]


    # =====================================================
    # 1. CHECK REQUIRED FEATURES
    # =====================================================

    def validate_features(self):

        missing_features = [

            feature

            for feature in self.required_features

            if feature not in self.data.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing required fusion features: "
                +
                ", ".join(missing_features)
            )


    # =====================================================
    # 2. NORMALIZE MODALITIES
    # =====================================================

    def normalize_modalities(self):

        """
        Ensures every modality is represented on a
        common 0-1 scale before fusion.
        """

        for feature in self.required_features:

            self.data[feature] = (

                pd.to_numeric(
                    self.data[feature],
                    errors="coerce"
                )

                .clip(0, 1)
            )


        self.data[
            self.required_features
        ] = (

            self.data[
                self.required_features
            ]

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )


    # =====================================================
    # 3. PHYSIOLOGICAL + ENVIRONMENTAL INTERACTION
    # =====================================================

    def calculate_physiological_environmental_interaction(
        self
    ):

        """
        Measures the combined effect of:

            physiological stress
                    ×
            environmental load

        A worker may have moderate physiological
        stress and moderate environmental stress
        individually, but their combination can
        represent a stronger condition.
        """

        self.data[
            "physiological_environment_interaction"
        ] = (

            self.data[
                "physiological_state"
            ]

            *

            self.data[
                "environmental_state"
            ]
        )


    # =====================================================
    # 4. ACTIVITY + ENVIRONMENT INTERACTION
    # =====================================================

    def calculate_activity_environment_interaction(
        self
    ):

        """
        Represents the additional burden created when
        physical activity occurs under environmental load.
        """

        self.data[
            "activity_environment_interaction"
        ] = (

            self.data[
                "activity_exposure_state"
            ]

            *

            self.data[
                "environmental_state"
            ]
        )


    # =====================================================
    # 5. TEMPORAL AMPLIFICATION
    # =====================================================

    def calculate_temporal_amplification(self):

        """
        Temporal deterioration is used as an adaptive
        modifier.

        If physiological/environmental conditions are
        getting worse, the unified state should increase.
        """

        base_state = (

            0.40 *
            self.data[
                "physiological_state"
            ]

            +

            0.30 *
            self.data[
                "environmental_state"
            ]

            +

            0.20 *
            self.data[
                "activity_exposure_state"
            ]
        )


        temporal_effect = (

            0.20 *
            self.data[
                "temporal_deterioration"
            ]
        )


        self.data[
            "temporal_amplification"
        ] = (

            base_state
            *

            temporal_effect
        )


    # =====================================================
    # 6. UNIFIED WORKER STATE
    # =====================================================

    def create_unified_worker_state(self):

        """
        Produces the unified state used by the next
        MADTP stages.

        The fusion contains:

            physiological condition
            environmental condition
            activity/exposure
            temporal deterioration
            cross-modal interactions
        """

        physiological = (

            0.30 *
            self.data[
                "physiological_state"
            ]
        )


        environmental = (

            0.25 *
            self.data[
                "environmental_state"
            ]
        )


        activity_exposure = (

            0.20 *
            self.data[
                "activity_exposure_state"
            ]
        )


        temporal = (

            0.10 *
            self.data[
                "temporal_deterioration"
            ]
        )


        physiological_environment = (

            0.10 *
            self.data[
                "physiological_environment_interaction"
            ]
        )


        activity_environment = (

            0.05 *
            self.data[
                "activity_environment_interaction"
            ]
        )


        self.data[
            "unified_worker_state"
        ] = (

            physiological

            +

            environmental

            +

            activity_exposure

            +

            temporal

            +

            physiological_environment

            +

            activity_environment
        )


        self.data[
            "unified_worker_state"
        ] = (

            self.data[
                "unified_worker_state"
            ]

            .clip(0, 1)
        )


    # =====================================================
    # 7. STATE CLASSIFICATION
    # =====================================================

    def classify_state(self):

        """
        Converts the unified numerical state into
        an interpretable condition.

        This is NOT the final risk classification.
        Final risk belongs to risk_prediction.py.
        """

        def classify(value):

            if value < 0.25:

                return "STABLE"

            elif value < 0.50:

                return "WATCH"

            elif value < 0.70:

                return "ELEVATED"

            elif value < 0.85:

                return "HIGH"

            else:

                return "SEVERE"


        self.data[
            "unified_state_class"
        ] = (

            self.data[
                "unified_worker_state"
            ]

            .apply(classify)
        )


    # =====================================================
    # 8. COMPLETE FUSION PIPELINE
    # =====================================================

    def fuse(self):

        print("\n")
        print("=" * 60)
        print("          MADTP MULTIMODAL FUSION")
        print("=" * 60)


        # ---------------------------------------------
        # Validate
        # ---------------------------------------------

        self.validate_features()

        print(
            "\n✓ Required modality features verified"
        )


        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------

        self.normalize_modalities()

        print(
            "✓ Modalities normalized"
        )


        # ---------------------------------------------
        # Cross-modal interactions
        # ---------------------------------------------

        self.calculate_physiological_environmental_interaction()

        print(
            "✓ Physiological-environment interaction calculated"
        )


        self.calculate_activity_environment_interaction()

        print(
            "✓ Activity-environment interaction calculated"
        )


        # ---------------------------------------------
        # Temporal effect
        # ---------------------------------------------

        self.calculate_temporal_amplification()

        print(
            "✓ Temporal amplification calculated"
        )


        # ---------------------------------------------
        # Unified state
        # ---------------------------------------------

        self.create_unified_worker_state()

        print(
            "✓ Unified worker state generated"
        )


        # ---------------------------------------------
        # Classification
        # ---------------------------------------------

        self.classify_state()

        print(
            "✓ Unified state classified"
        )


        print(
            "\nMultimodal fusion completed."
        )


        return self.data


# =========================================================
# TEST MODULE
# =========================================================

if __name__ == "__main__":

    input_file = (
        "data/feature_data.csv"
    )

    output_file = (
        "data/fused_data.csv"
    )


    try:

        # -----------------------------------------
        # Load feature data
        # -----------------------------------------

        data = pd.read_csv(
            input_file
        )


        print(
            "\nInput feature data loaded:"
        )

        print(
            data.shape
        )


        # -----------------------------------------
        # Create fusion engine
        # -----------------------------------------

        fusion = MultimodalFusion(
            data
        )


        # -----------------------------------------
        # Perform fusion
        # -----------------------------------------

        fused_data = fusion.fuse()


        # -----------------------------------------
        # Display important results
        # -----------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            "FUSED WORKER STATE"
        )

        print(
            "----------------------------------------"
        )


        display_columns = [

            "time",

            "physiological_state",

            "environmental_state",

            "activity_exposure_state",

            "temporal_deterioration",

            "physiological_environment_interaction",

            "activity_environment_interaction",

            "unified_worker_state",

            "unified_state_class"
        ]


        print(
            fused_data[
                display_columns
            ].head(10)
        )


        # -----------------------------------------
        # Save fused data
        # -----------------------------------------

        fused_data.to_csv(
            output_file,
            index=False
        )


        print(
            "\n✓ Fused data saved to:"
        )

        print(
            output_file
        )


    except FileNotFoundError:

        print(
            "\nERROR:"
        )

        print(
            "data/feature_data.csv "
            "was not found."
        )

        print(
            "Run feature_extraction.py first."
        )