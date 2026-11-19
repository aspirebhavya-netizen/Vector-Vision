
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("\nVECTOR VISION - DENGUE EARLY WARNING SYSTEM")
print("=" * 55)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

features = pd.read_csv(DATA_DIR / "dengue_features_train.csv")
labels = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
test_features = pd.read_csv(DATA_DIR / "dengue_features_test.csv")

print("\nDATA LOADED")
print("Training features:", features.shape)
print("Training labels:", labels.shape)
print("Test features:", test_features.shape)


# --------------------------------------------------
# 3. MERGE WEATHER AND DENGUE DATA
# --------------------------------------------------

df = pd.merge(
    features,
    labels,
    on=["city", "year", "weekofyear"],
    how="inner"
)

df["week_start_date"] = pd.to_datetime(df["week_start_date"])

df = df.sort_values(
    ["city", "week_start_date"]
).reset_index(drop=True)

test_features["week_start_date"] = pd.to_datetime(
    test_features["week_start_date"]
)

print("\nMERGED DATA:", df.shape)
print("Duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 4. CREATE THE NEXT-WEEK TARGET
# --------------------------------------------------

# This week's weather will be used to predict
# dengue cases in the next available weekly record.

df["future_cases"] = (
    df.groupby("city")["total_cases"].shift(-1)
)

# Remove the last record for each city because
# its next week's case count is not available.

df = df.dropna(subset=["future_cases"]).copy()
df["future_cases"] = df["future_cases"].astype(int)

print("\nNEXT-WEEK TARGET CREATED")
print("Usable records:", len(df))


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

# Extract month from the date.

df["month"] = df["week_start_date"].dt.month
test_features["month"] = test_features["week_start_date"].dt.month

# Weather, city and calendar information are features.
# Do NOT include total_cases or future_cases as inputs.

excluded_columns = [
    "week_start_date",
    "total_cases",
    "future_cases"
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

# Keep only columns that are available in test data too.
feature_columns = [
    col for col in feature_columns
    if col in test_features.columns
]

X_all = df[feature_columns].copy()

# Convert city into a categorical feature.
categorical_features = ["city"]

numeric_features = [
    col for col in feature_columns
    if col not in categorical_features
]

print("\nFEATURES:", len(feature_columns))


# --------------------------------------------------
# 6. CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# --------------------------------------------------

# We split by date, not randomly.
# Earlier weeks are used for training.
# Later weeks are used for validation.

unique_dates = sorted(df["week_start_date"].unique())

split_position = int(len(unique_dates) * 0.80)
split_position = max(1, min(split_position, len(unique_dates) - 1))

split_date = unique_dates[split_position]

train_mask = df["week_start_date"] < split_date
valid_mask = df["week_start_date"] >= split_date

X_train = X_all.loc[train_mask].copy()
X_valid = X_all.loc[valid_mask].copy()

cases_train = df.loc[train_mask, "future_cases"]
cases_valid = df.loc[valid_mask, "future_cases"]

cities_train = df.loc[train_mask, "city"]
cities_valid = df.loc[valid_mask, "city"]

print("\nCHRONOLOGICAL SPLIT")
print("Training records:", len(X_train))
print("Validation records:", len(X_valid))
print("Validation begins:", split_date)


# --------------------------------------------------
# 7. CREATE LOW, MEDIUM AND HIGH RISK LABELS
# --------------------------------------------------

# Risk thresholds are calculated separately for each city.
# They are based ONLY on training data.
#
# Low    = up to the city's training median
# Medium = above median, up to its 80th percentile
# High   = above its 80th percentile
#
# These are relative historical risk categories,
# not official medical or government alert thresholds.

thresholds = {}

threshold_data = pd.DataFrame({
    "city": cities_train,
    "cases": cases_train
})

for city, group in threshold_data.groupby("city"):
    thresholds[city] = {
        "low_limit": float(group["cases"].quantile(0.50)),
        "medium_limit": float(group["cases"].quantile(0.80))
    }


def make_risk_labels(cases, cities, threshold_values):
    results = []

    for case_count, city in zip(cases, cities):
        limits = threshold_values[city]

        if case_count <= limits["low_limit"]:
            results.append("Low")
        elif case_count <= limits["medium_limit"]:
            results.append("Medium")
        else:
            results.append("High")

    return pd.Series(results, index=cases.index)


y_train = make_risk_labels(
    cases_train, cities_train, thresholds
)

y_valid = make_risk_labels(
    cases_valid, cities_valid, thresholds
)

risk_order = ["Low", "Medium", "High"]

print("\nRISK CATEGORY COUNTS - TRAINING")
print(y_train.value_counts())


# --------------------------------------------------
# 8. DATA PREPROCESSING
# --------------------------------------------------

# Missing numerical values are filled with the median.
# Missing city values are filled with the most frequent city.
# City is converted into numerical columns.

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 9. CREATE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}


# --------------------------------------------------
# 10. TRAIN AND COMPARE MODELS
# --------------------------------------------------

trained_models = {}
results = {}

for model_name, classifier in models.items():

    print("\nTRAINING:", model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)

    macro_f1 = f1_score(
        y_valid,
        predictions,
        labels=risk_order,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(y_valid, predictions)

    high_recall = recall_score(
        y_valid,
        predictions,
        labels=["High"],
        average=None,
        zero_division=0
    )[0]

    trained_models[model_name] = pipeline

    results[model_name] = {
        "Macro F1": macro_f1,
        "Accuracy": accuracy,
        "High Risk Recall": high_recall
    }

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("High Risk Recall:", round(high_recall, 4))


# --------------------------------------------------
# 11. SELECT THE BEST MODEL
# --------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["Macro F1"]
)

best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)

comparison = pd.DataFrame(results).T
print("\nMODEL COMPARISON")
print(comparison)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv"
)


# --------------------------------------------------
# 12. DETAILED MODEL EVALUATION
# --------------------------------------------------

valid_predictions = best_model.predict(X_valid)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_valid,
    valid_predictions,
    labels=risk_order,
    zero_division=0
))

cm = confusion_matrix(
    y_valid,
    valid_predictions,
    labels=risk_order
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=risk_order
)

display.plot()
plt.title(f"{best_model_name} - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()


# --------------------------------------------------
# 13. FEATURE IMPORTANCE
# --------------------------------------------------

model_step = best_model.named_steps["model"]
preprocessor_step = best_model.named_steps["preprocessor"]

feature_names = preprocessor_step.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model_step.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance.head(15))

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Features - Vector Vision")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.show()


# --------------------------------------------------
# 14. SAVE THE TRAINED MODEL
# --------------------------------------------------

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "thresholds": thresholds,
    "risk_order": risk_order,
    "prediction_horizon": "Next available weekly record"
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "vector_vision_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY")


# --------------------------------------------------
# 15. PREDICT RISK FOR TEST DATA
# --------------------------------------------------

X_test = test_features[feature_columns].copy()

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)

prediction_output = test_features[
    ["city", "year", "weekofyear", "week_start_date"]
].copy()

prediction_output["predicted_risk"] = test_predictions

# Add the model's probability for each risk class.
model_classes = best_model.named_steps["model"].classes_

for index, class_name in enumerate(model_classes):
    prediction_output[
        f"probability_{class_name.lower()}"
    ] = test_probabilities[:, index]

prediction_output.to_csv(
    OUTPUT_DIR / "test_risk_predictions.csv",
    index=False
)

print("\nTEST DATA RISK PREDICTIONS")
print(prediction_output.head(10))

print("\nPREDICTIONS SAVED SUCCESSFULLY")


# --------------------------------------------------
# 16. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("VECTOR VISION PIPELINE COMPLETED")
print("=" * 55)

print("Selected model:", best_model_name)
print("Validation Macro F1:", round(
    results[best_model_name]["Macro F1"], 4
))
print("Validation Accuracy:", round(
    results[best_model_name]["Accuracy"], 4
))
print("Validation High Risk Recall:", round(
    results[best_model_name]["High Risk Recall"], 4
))

print("\nCheck these folders:")
print("Models:", MODEL_DIR)
print("Outputs:", OUTPUT_DIR)
