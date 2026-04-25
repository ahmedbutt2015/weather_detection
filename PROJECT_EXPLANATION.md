# Project Explanation

## 1) What We Are Doing

The goal is to predict whether it will rain (`RainTomorrow`) using weather attributes such as:

- Temperature (`MinTemp`, `MaxTemp`)
- `Rainfall`
- `Humidity`
- `Pressure`
- `WindSpeed`, `WindGustSpeed`
- `Cloud`

This is a **binary classification** task where:

- `0` = No rain
- `1` = Rain

## 2) How We Did It

The full flow is implemented in `rainfall_pipeline.py` and executed through `Code.py`.

### Step A: Data Loading and Inspection

We read the CSV and print:

- shape
- column names
- missing values

### Step B: Target Handling

We detect target column automatically from common names (`RainTomorrow`, `Rain`, etc.).

Then we normalize target values:

- numeric values are kept as integers
- text labels like `Yes/No`, `True/False`, `Rain/No Rain` are converted to `1/0`

Rows with invalid target values are removed.

### Step C: Preprocessing

- Numerical columns: median imputation + standard scaling
- Categorical columns: mode imputation + one-hot encoding

This is done with a `ColumnTransformer`.

### Step D: Train/Test Split

- 80% train, 20% test
- `stratify=y` to preserve class ratio in both sets

### Step E: Model Training

We train 5 models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. SVM (RBF)

Each model is trained inside an sklearn `Pipeline` with preprocessing + model in one flow.

### Step F: Evaluation

For every model, we compute:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix

Metrics are saved to `outputs/model_results.csv`.

### Step G: Visualization

We save multiple distinct plots:

- target class distribution
- feature correlation heatmap
- numeric feature histograms
- feature boxplots by class
- model performance comparison chart
- combined ROC curves
- random-forest feature importance
- confusion matrix per model

## 3) What Is Happening At The End

At the end of execution:

1. All models finish training and evaluation.
2. A sorted model-comparison table is printed and saved.
3. All figures are saved in `outputs/`.

From the current run, key metrics were:

- Logistic Regression: Accuracy `0.9311`, ROC-AUC `0.9766`
- SVM (RBF): Accuracy `0.9830`, ROC-AUC `0.9984`
- Tree-based models (Decision Tree, Random Forest, Gradient Boosting): near-perfect scores on the test split

## 4) Important Interpretation Note

Near-perfect scores can happen when:

- the dataset is highly separable
- there is feature leakage
- data generation/collection naturally makes the target easy to infer

So these results are strong, but for academic rigor you should also consider:

- k-fold cross-validation
- leakage checks
- feature ablation experiments

## 5) How To Reproduce

Run:

```bash
python3 Code.py
```

Then inspect:

- `outputs/model_results.csv`
- all `.png` files in `outputs/`
