
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. PROJECT FILE LOCATIONS
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "models" / "vector_vision_model.pkl"
DATA_PATH = PROJECT_DIR / "dataset" / "dengue_features_test.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs"


# -----------------------------------------
# 2. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Vector Vision",
    page_icon="🦟",
    layout="wide"
)


# -----------------------------------------
# 3. LOAD TRAINED MODEL
# -----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    data = pd.read_csv(DATA_PATH)
    data["week_start_date"] = pd.to_datetime(
        data["week_start_date"]
    )
    data["month"] = data["week_start_date"].dt.month
    return data


# -----------------------------------------
# 4. DASHBOARD DESIGN
# -----------------------------------------

st.title("🦟 VECTOR VISION")
st.subheader("Weather-Driven Dengue Early Warning System")

st.write(
    "An experimental machine learning dashboard "
    "that estimates dengue risk using weather and "
    "historical patterns."
)

st.info(
    "These are experimental predictions, not official "
    "public-health alerts or medical diagnoses."
)

st.divider()


# -----------------------------------------
# 5. LOAD DATA AND MODEL
# -----------------------------------------

try:
    bundle = load_model()
    test_data = load_test_data()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()


# -----------------------------------------
# 6. CITY SELECTION
# -----------------------------------------

st.sidebar.header("Dashboard Filters")

cities = sorted(test_data["city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "Select City",
    cities
)

city_data = test_data[
    test_data["city"] == selected_city
].copy()


# -----------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------

X_test = city_data[feature_columns].copy()

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

prediction_data = city_data[
    ["city", "year", "weekofyear", "week_start_date"]
].copy()

prediction_data["Predicted Risk"] = predictions

model_classes = model.named_steps["model"].classes_

for index, class_name in enumerate(model_classes):
    prediction_data[
        f"{class_name} Probability"
    ] = probabilities[:, index] * 100


# -----------------------------------------
# 8. DASHBOARD SUMMARY
# -----------------------------------------

st.header("Dengue Risk Overview")

total_weeks = len(prediction_data)

high_risk_weeks = (
    prediction_data["Predicted Risk"] == "High"
).sum()

medium_risk_weeks = (
    prediction_data["Predicted Risk"] == "Medium"
).sum()

low_risk_weeks = (
    prediction_data["Predicted Risk"] == "Low"
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("City", selected_city.upper())
col2.metric("Weeks Analysed", total_weeks)
col3.metric("High Risk Weeks", int(high_risk_weeks))
col4.metric("Medium Risk Weeks", int(medium_risk_weeks))

st.divider()


# -----------------------------------------
# 9. RISK DISTRIBUTION CHART
# -----------------------------------------

st.header("Predicted Risk Distribution")

risk_counts = (
    prediction_data["Predicted Risk"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    risk_counts.index,
    risk_counts.values
)

ax.set_xlabel("Dengue Risk Category")
ax.set_ylabel("Number of Weeks")
ax.set_title("Predicted Dengue Risk by Category")

st.pyplot(fig)
plt.close(fig)


# -----------------------------------------
# 10. WEEKLY PREDICTIONS
# -----------------------------------------

st.header("Weekly Dengue Risk Predictions")

display_data = prediction_data.copy()

display_data["week_start_date"] = (
    display_data["week_start_date"].dt.strftime("%Y-%m-%d")
)

st.dataframe(
    display_data,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------------
# 11. SELECT A WEEK
# -----------------------------------------

st.header("Explore an Individual Prediction")

selected_index = st.selectbox(
    "Choose a weekly record",
    prediction_data.index,
    format_func=lambda index: (
        f"Week {prediction_data.loc[index, 'weekofyear']} - "
        f"{prediction_data.loc[index, 'week_start_date'].date()}"
    )
)

selected_record = prediction_data.loc[selected_index]

risk = selected_record["Predicted Risk"]

if risk == "High":
    st.error("HIGH DENGUE RISK")
elif risk == "Medium":
    st.warning("MEDIUM DENGUE RISK")
else:
    st.success("LOW DENGUE RISK")

st.write(
    "Prediction probabilities:"
)

probability_columns = [
    f"{class_name} Probability"
    for class_name in model_classes
]

st.dataframe(
    selected_record[probability_columns].to_frame(
        name="Probability (%)"
    ),
    use_container_width=True
)


# -----------------------------------------
# 12. DOWNLOAD PREDICTIONS
# -----------------------------------------

st.header("Download Results")

csv_data = prediction_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Risk Predictions CSV",
    data=csv_data,
    file_name=f"vector_vision_{selected_city}_predictions.csv",
    mime="text/csv"
)


# -----------------------------------------
# 13. PROJECT INFORMATION
# -----------------------------------------

st.divider()

st.header("About Vector Vision")

st.write(
    """
    Vector Vision uses historical dengue records and
    weather information to estimate dengue risk categories.

    Machine Learning Models:
    - Random Forest Classifier
    - Gradient Boosting Classifier

    Selected Model: Random Forest

    Risk categories are based on relative historical
    case-count thresholds used during model training.
    """
)

st.caption(
    "Vector Vision | Experimental Academic Project | 2026"
)
