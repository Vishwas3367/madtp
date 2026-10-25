import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# RANDOM FOREST PREDICTOR
# =========================================================

class RandomForestPredictor:
    """
    Random Forest prototype for MADTP.

    The model predicts whether the worker's condition
    is likely to deteriorate in the near future.

    Current training uses the virtual/simulated dataset.

    Later:
        Real wearable sensor data can replace the
        simulated data without changing the model
        interface.
    """

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=200,

            max_depth=8,

            min_samples_split=5,

            min_samples_leaf=2,

            random_state=42,

            class_weight="balanced"
        )

        self.feature_columns = [

            "physiological_state",

            "environmental_state",

            "activity_exposure_state",

            "temporal_deterioration",

            "twin_divergence_index",

            "hr_deviation",

            "spo2_deviation",

            "temperature_deviation",

            "heat_load",

            "activity",

            "exposure_time"
        ]

        self.trained = False


    # =====================================================
    # 1. CHECK FEATURES
    # =====================================================

    def validate_features(self, data):

        missing = [

            column

            for column in self.feature_columns

            if column not in data.columns
        ]

        if missing:

            raise ValueError(

                "Missing required features: "

                +

                ", ".join(missing)
            )


    # =====================================================
    # 2. CREATE FUTURE TARGET
    # =====================================================

    def create_target(self, data):

        """
        Creates a prototype future deterioration target.

        We look 5 samples ahead.

        If the future multimodal condition becomes
        sufficiently higher than the current condition,
        the sample is labelled as deterioration.

        0 = No predicted deterioration
        1 = Future deterioration

        This is a prototype target and will eventually
        be replaced by labels derived from real wearable
        observations.
        """

        data = data.copy()


        current_signal = (

            0.35 *
            data["physiological_state"]

            +

            0.25 *
            data["environmental_state"]

            +

            0.15 *
            data["activity_exposure_state"]

            +

            0.10 *
            data["temporal_deterioration"]

            +

            0.15 *
            data["twin_divergence_index"]
        )


        # Look 5 samples into the future

        future_signal = (
            current_signal.shift(-5)
        )


        # Future deterioration condition

        target = (

            future_signal

            >

            current_signal + 0.03
        )


        target = (
            target
            .fillna(False)
            .astype(int)
        )


        data[
            "future_deterioration"
        ] = target


        return data


    # =====================================================
    # 3. PREPARE FEATURES
    # =====================================================

    def prepare_features(self, data):

        X = data[
            self.feature_columns
        ].copy()


        X = (

            X

            .apply(
                pd.to_numeric,
                errors="coerce"
            )

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )


        return X


    # =====================================================
    # 4. TRAIN
    # =====================================================

    def train(self, data):

        self.validate_features(
            data
        )


        data = self.create_target(
            data
        )


        X = self.prepare_features(
            data
        )

        y = data[
            "future_deterioration"
        ]


        print(
            "\n----------------------------------------"
        )

        print(
            "TARGET DISTRIBUTION"
        )

        print(
            "----------------------------------------"
        )

        print(
            y.value_counts()
        )


        if y.nunique() < 2:

            raise ValueError(

                "Target contains only one class. "
                "The dataset does not contain enough "
                "variation for Random Forest training."
            )


        # -------------------------------------------------
        # Train/test split
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.25,

                random_state=42,

                stratify=y
            )
        )


        print(
            "\n✓ Training/testing data prepared"
        )


        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

        self.model.fit(

            X_train,

            y_train
        )


        self.trained = True


        print(
            "✓ Random Forest model trained"
        )


        # -------------------------------------------------
        # Test model
        # -------------------------------------------------

        predictions = (
            self.model.predict(
                X_test
            )
        )


        accuracy = accuracy_score(

            y_test,

            predictions
        )


        print(
            "\n========================================"
        )

        print(
            "     RANDOM FOREST MODEL RESULT"
        )

        print(
            "========================================"
        )


        print(
            f"Training samples : {len(X_train)}"
        )

        print(
            f"Testing samples  : {len(X_test)}"
        )

        print(
            f"Accuracy         : {accuracy:.3f}"
        )


        print(
            "\nClassification Report:"
        )

        print(
            classification_report(

                y_test,

                predictions,

                zero_division=0
            )
        )


        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(

                y_test,

                predictions
            )
        )


        return accuracy


    # =====================================================
    # 5. PREDICT
    # =====================================================

    def predict(self, data):

        if not self.trained:

            raise RuntimeError(

                "Random Forest model is not trained."
            )


        X = self.prepare_features(
            data
        )


        predictions = (
            self.model.predict(
                X
            )
        )


        probabilities = (
            self.model.predict_proba(
                X
            )
        )


        # Find probability corresponding to class 1

        positive_class_index = list(
            self.model.classes_
        ).index(1)


        result = data.copy()


        result[
            "rf_prediction"
        ] = predictions


        result[
            "rf_deterioration_probability"
        ] = (

            probabilities[
                :,
                positive_class_index
            ]
        )


        return result


    # =====================================================
    # 6. FEATURE IMPORTANCE
    # =====================================================

    def get_feature_importance(self):

        if not self.trained:

            raise RuntimeError(

                "Train the model before checking "
                "feature importance."
            )


        importance = pd.DataFrame({

            "feature":
                self.feature_columns,

            "importance":
                self.model.feature_importances_
        })


        importance = (

            importance

            .sort_values(

                "importance",

                ascending=False
            )

            .reset_index(
                drop=True
            )
        )


        return importance

    def load_model(self, path):
        """Load an existing trained model without retraining."""
        model = joblib.load(path)

        expected_features = len(self.feature_columns)
        actual_features = getattr(model, "n_features_in_", None)

        if actual_features != expected_features:
            raise ValueError(
                f"Model expects {actual_features} features, "
                f"but this predictor defines {expected_features}."
            )

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None:
            if list(model_features) != self.feature_columns:
                raise ValueError(
                    "Saved model feature names or order do not match "
                    "the predictor's feature_columns."
                )

        self.model = model
        self.trained = True
        print(f"Loaded trained model: {path}")
    # =====================================================
    # 7. SAVE MODEL
    # =====================================================

    def save_model(self, path):

        joblib.dump(

            self.model,

            path
        )


        print(
            f"\n✓ Random Forest model saved to:"
        )

        print(
            path
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n")

    print(
        "=" * 60
    )

    print(
        "       MADTP RANDOM FOREST PREDICTION"
    )

    print(
        "=" * 60
    )


    # -----------------------------------------------------
    # File paths
    # -----------------------------------------------------

    input_file = (
        "data/twin_divergence_data.csv"
    )

    output_file = (
        "data/rf_prediction_data.csv"
    )

    model_file = (
        "ai/random_forest_model.pkl"
    )


    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    try:

        data = pd.read_csv(
            input_file
        )

    except FileNotFoundError:

        print(
            "\nERROR:"
        )

        print(
            f"Input file not found:"
        )

        print(
            input_file
        )

        print(
            "\nRun this first:"
        )

        print(
            "python madtp/twin_divergence.py"
        )

        sys.exit(1)


    print(
        f"\n✓ Input data loaded: "
        f"{data.shape}"
    )


    # -----------------------------------------------------
    # Create predictor
    # -----------------------------------------------------

    predictor = (
        RandomForestPredictor()
    )


    print(
        "✓ Random Forest predictor initialized"
    )


    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    predictor.train(
        data
    )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    result = predictor.predict(
        data
    )


    print(
        "\n✓ Future deterioration predictions generated"
    )


    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance = (
        predictor.get_feature_importance()
    )


    print(
        "\n----------------------------------------"
    )

    print(
        "RANDOM FOREST FEATURE IMPORTANCE"
    )

    print(
        "----------------------------------------"
    )


    print(
        importance.to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # Display prediction results
    # -----------------------------------------------------

    display_columns = [

        "time",

        "twin_divergence_index",

        "physiological_state",

        "environmental_state",

        "temporal_deterioration",

        "rf_prediction",

        "rf_deterioration_probability"
    ]


    print(
        "\n----------------------------------------"
    )

    print(
        "RANDOM FOREST PREDICTIONS"
    )

    print(
        "----------------------------------------"
    )


    print(
        result[
            display_columns
        ].head(20).to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    result.to_csv(

        output_file,

        index=False
    )


    print(
        "\n✓ Prediction data saved to:"
    )

    print(
        output_file
    )


    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------

    predictor.save_model(
        model_file
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print(
        "\n----------------------------------------"
    )

    print(
        "RANDOM FOREST SUMMARY"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Average deterioration probability : "
        f"{result['rf_deterioration_probability'].mean():.3f}"
    )


    print(
        f"Maximum deterioration probability : "
        f"{result['rf_deterioration_probability'].max():.3f}"
    )


    print(
        f"Predicted deterioration samples   : "
        f"{result['rf_prediction'].sum()}"
    )


    print(
        "\n✓ Random Forest AI module completed successfully."
    )