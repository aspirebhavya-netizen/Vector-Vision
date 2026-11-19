
# 🦟 Vector Vision
## Weather-Driven Dengue Early Warning System

### About the Project
Vector Vision is a machine learning-based experimental early-warning system that predicts dengue risk categories using historical dengue cases and weather-related information.

The system analyses environmental factors such as temperature, humidity, rainfall, and vegetation indicators to estimate dengue risk as Low, Medium, or High.

### Problem Statement
Dengue outbreaks are influenced by environmental and climatic conditions. Identifying potential risk patterns early can help support public-health planning and awareness.

### Objectives
- Analyse historical dengue and weather data.
- Identify environmental patterns associated with dengue risk.
- Predict dengue risk categories using machine learning.
- Compare different classification algorithms.
- Display predictions through an interactive dashboard.

### Dataset
Dataset: DengAI: Predicting Disease Spread  
Source: DrivenData

The dataset includes:
- Historical dengue case counts.
- Temperature measurements.
- Relative humidity.
- Rainfall information.
- Vegetation indices (NDVI).
- City, year, and week information.

### Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

### Machine Learning Models
1. Random Forest Classifier
2. Gradient Boosting Classifier

Random Forest was selected as the current best model based on Macro F1-score and High Risk Recall on the validation data.

### Model Evaluation
The models were evaluated using:
- Accuracy
- Macro F1-score
- Precision
- Recall
- Confusion Matrix

### Application Features
- Interactive city selection.
- Dengue risk overview.
- Predicted risk distribution chart.
- Weekly risk predictions.
- Individual weekly prediction probabilities.
- CSV download of predictions.

### Project Structure

```text
IML project/
├── app/
│   └── app.py
├── dataset/
│   ├── dengue_features_train.csv
│   ├── dengue_labels_train.csv
│   ├── dengue_features_test.csv
│   └── submission_format.csv
├── models/
│   └── vector_vision_model.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   └── test_risk_predictions.csv
├── src/
│   └── train_model.py
└── README.md
```

### How to Run the Project

1. Install the required Python libraries.
2. Open the project folder in VS Code.
3. Train the model:

```bash
python src/train_model.py
```

4. Start the dashboard:

```bash
python -m streamlit run app/app.py
```

### Disclaimer
Vector Vision is an experimental academic project. Its predictions are not official dengue outbreak alerts, medical diagnoses, or substitutes for public-health surveillance.

### Future Enhancements
- Integrate real-time weather information.
- Include more geographical regions.
- Improve model performance and validation.
- Develop interactive geographical risk maps.
- Explore deployment for wider accessibility.

---
**Project: Vector Vision | 2026**
