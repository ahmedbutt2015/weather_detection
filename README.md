# Rainfall Prediction Classification Project

This project builds a machine learning classification system to predict rainfall occurrence (`RainTomorrow`) from historical weather features.

## Project Structure

- `Code.py`: Main file (recommended entry point)
- `rainfall_pipeline.py`: Full ML workflow implementation
- `cw2.py`: Compatibility entry point (calls `Code.py`)
- `weather_based_rain_prediction_dataset.csv`: Input dataset
- `outputs/`: Generated plots and model metrics
- `PROJECT_EXPLANATION.md`: Detailed methodology and result interpretation

## How to Run

Use Python 3 in this project folder:

```bash
python3 Code.py
```

This will:
1. Load and preprocess data
2. Train multiple classification models
3. Evaluate and compare model performance
4. Save all visualizations and a result table in `outputs/`

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM (RBF)

## Output Files

Main generated artifacts:

- `outputs/model_results.csv`
- `outputs/01_target_distribution.png`
- `outputs/02_correlation_heatmap.png`
- `outputs/03_numeric_distributions.png`
- `outputs/04_boxplots_by_class.png`
- `outputs/05_model_comparison.png`
- `outputs/06_roc_curves.png`
- `outputs/07_feature_importance_rf.png`
- `outputs/cm_*.png` (confusion matrix per model)
