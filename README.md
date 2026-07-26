# Module 6: Wine Cultivar Model Serving

This project applies machine-learning model building and serving skills to a new multiclass classification problem.

The project uses the scikit-learn wine dataset to predict a wine cultivar from chemical measurements.
It compares a baseline Decision Tree classifier with a Random Forest ensemble, evaluates the selected model, saves a serving-ready artifact, and validates API-style prediction inputs.

## Project Question

Given the chemical measurements of a wine sample, can a machine-learning model predict which cultivar it belongs to?

## Technical Modification

The original Module 6 example trained and served a penguin species classifier. This custom project makes the following technical modifications:

- Replaces the penguin dataset with scikit-learn's wine dataset.
- Predicts one of three wine cultivars.
- Uses 13 numeric chemical measurements as features.
- Compares a Decision Tree baseline with a Random Forest ensemble.
- Adds five-fold stratified cross-validation.
- Generates a classification report and confusion matrix.
- Calculates Random Forest feature importance.
- Saves a serving-ready model bundle instead of saving only the estimator.
- Stores feature names, target names, validation ranges, and model metadata with the trained model.
- Includes an API-style prediction function with input validation.
- Writes major workflow results to `project.log`.

## Dataset

The project uses:

```python
from sklearn.datasets import load_wine
```

The dataset contains:

- 178 wine samples
- 13 numeric chemical features
- 3 target cultivars

The target classes are:

- `class_0`
- `class_1`
- `class_2`

## Features

The model uses the following features:

- alcohol
- malic_acid
- ash
- alcalinity_of_ash
- magnesium
- total_phenols
- flavanoids
- nonflavanoid_phenols
- proanthocyanins
- color_intensity
- hue
- od280/od315_of_diluted_wines
- proline

## Models

Two models are trained and compared:

1. `DecisionTreeClassifier`
2. `RandomForestClassifier`

The Decision Tree serves as the baseline model. The Random Forest is the ensemble model selected for the saved serving artifact.

## New Skills Applied

This project applies several new skills beyond the original example:

### Stratified Cross-Validation

Five-fold stratified cross-validation is used to provide a more reliable estimate of model performance than one train-test split alone.

### Model Interpretation

Random Forest feature importance is used to identify which chemical measurements contributed most strongly to predictions.

### Serving-Ready Model Bundle

The saved artifact contains:

- trained model
- feature names
- target names
- feature validation ranges
- dataset and model metadata
- held-out test accuracy
- mean cross-validation accuracy
- cross-validation standard deviation

### Input Validation

The custom prediction function checks for:

- missing features
- nonnumeric values
- values outside the observed training range

The function returns:

- predicted cultivar
- class probabilities
- validation warnings

## Project Structure

```text
ml-06-serving/
├── artifacts/
│   └── wine_model_bundle.joblib
├── docs/
│   └── index.md
├── notebooks/
│   └── module6_wine_serving_femi_rewritten.ipynb
├── project.log
├── README.md
└── pyproject.toml
```

## How to Run the Notebook

Open the project in VS Code and activate the project environment.

From the root project folder, start JupyterLab:

```powershell
uv run jupyter lab
```

Open:

```text
"https://github.com/Airfirm/ml-06-serving/"

notebooks/module6_wine_serving_femi_rewritten.ipynb

"https://github.com/Airfirm/ml-06-serving/blob/main/notebooks/ml-06_wine_serve_model_femi.ipynb"
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

The notebook should create or update:

```text
project.log
artifacts/wine_model_bundle.joblib
```

## Important Working-Directory Note

The notebook is designed to run from the `notebooks` folder.

It uses the parent directory as the project root:

```python
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent
```

If the notebook is moved or run from a different folder, update the path configuration before running it.

## Expected Workflow

The notebook performs the following steps:

1. Configure paths and logging.
2. Load the wine dataset.
3. Inspect the data and check data quality.
4. Split the data using stratification.
5. Train the Decision Tree and Random Forest models.
6. Compare held-out test accuracy.
7. perform five-fold cross-validation.
8. Generate a classification report.
9. Display a confusion matrix.
10. Calculate and chart feature importance.
11. Save the serving-ready model bundle.
12. Test a valid prediction request.
13. Test missing-feature validation.
14. Write a final summary to `project.log`.

## Model Artifact

The trained model bundle is saved to:

```text
artifacts/wine_model_bundle.joblib
```

The artifact can be loaded with:

```python
import joblib

bundle = joblib.load("artifacts/wine_model_bundle.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
```

## Logging

The notebook writes major workflow events and results to:

```text
project.log
```

The final log summary includes:

- dataset
- target
- feature count
- best held-out model
- Decision Tree accuracy
- Random Forest accuracy
- mean cross-validation accuracy
- cross-validation standard deviation
- most important feature
- saved artifact location

## Results

Run the notebook and record the actual values below.

- Decision Tree test accuracy: `Add result after running`
- Random Forest test accuracy: `Add result after running`
- Mean cross-validation accuracy: `Add result after running`
- Cross-validation standard deviation: `Add result after running`
- Most important feature: `Add result after running`

## Interpretation

The Random Forest is expected to perform well because it combines many decision trees and reduces dependence on one individual tree.
Cross-validation provides evidence about whether the model performs consistently across different subsets of the data.

Feature importance helps explain which chemical measurements are most useful for distinguishing the three wine cultivars.
The confusion matrix shows whether any cultivar is more difficult for the model to classify.

## Future Improvements

Possible future improvements include:

- Add a FastAPI service that loads `wine_model_bundle.joblib`.
- Create a Pydantic request model for stronger API validation.
- Add automated tests for valid and invalid prediction requests.
- Compare Random Forest with Gradient Boosting.
- Tune model hyperparameters with grid search.
- Add model-version information to the saved metadata.
- Save charts to an `artifacts/images/` folder.
- Add GitHub Actions for automated testing.

## Author

Femi

## License

This project is for educational use.
