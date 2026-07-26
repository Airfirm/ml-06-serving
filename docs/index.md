# Module 6 Custom Project

## Wine Cultivar Classification and Model Serving

This project demonstrates how to build, evaluate, save, and prepare a machine-learning model for serving.

The custom project predicts a wine cultivar from chemical measurements using the scikit-learn wine dataset.

## Business and Analytical Question

Given a wine sample's chemical measurements, can a machine-learning model identify its cultivar?

A reliable classifier could help demonstrate how measurable product characteristics can be used to categorize observations automatically.

## Project Overview

The project follows a complete machine-learning workflow:

1. Load and inspect the dataset.
2. Check missing values and duplicate rows.
3. Separate features from the target.
4. Create stratified training and testing sets.
5. Train a Decision Tree baseline.
6. Train a Random Forest ensemble.
7. Compare held-out test accuracy.
8. Use five-fold stratified cross-validation.
9. Review a classification report.
10. Visualize a confusion matrix.
11. Analyze feature importance.
12. Save a serving-ready model bundle.
13. Test API-style prediction input.
14. Log major results to `project.log`.

## Dataset

The project uses the wine dataset included with scikit-learn.

The dataset contains 178 observations, 13 numeric chemical features, and three target classes.

### Feature Variables

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

### Target Variable

The target is the wine cultivar.

This is a supervised multiclass classification problem.

## Technical Modification

The original example used penguin measurements to predict penguin species and saved only the trained estimator.

This project modifies that workflow by:

- applying the serving workflow to a different dataset and domain,
- comparing a baseline model with an ensemble model,
- adding stratified cross-validation,
- calculating feature importance,
- creating a reusable input-validation function,
- and saving a complete model bundle with metadata.

## Model Comparison

### Decision Tree

The Decision Tree provides a simple baseline. It is easy to interpret but can be sensitive to the particular training data.

### Random Forest

The Random Forest combines many decision trees. It is used as the selected serving model because ensemble methods can reduce the instability of a single tree.

## Evaluation

The notebook evaluates the models with:

- held-out test accuracy,
- five-fold stratified cross-validation,
- classification precision,
- classification recall,
- F1-score,
- confusion matrix.

Cross-validation is especially important because one train-test split may produce a result that is unusually high or low.

## Model Interpretation

The notebook calculates Random Forest feature importance.

This identifies the chemical measurements that contributed most strongly to the model's predictions.
The top features should be reviewed after running the notebook because their exact rankings come from the trained model.

## Serving-Ready Artifact

The notebook saves:

```text
artifacts/wine_model_bundle.joblib
```

The saved bundle contains:

- the trained Random Forest model,
- the required feature order,
- the target class names,
- minimum and maximum training values,
- model performance metadata,
- dataset information.

Saving these items together helps prevent serving errors caused by mismatched features or missing labels.

## Prediction Validation

The notebook includes a function that accepts a dictionary similar to a JSON request.

The function verifies that:

- all required features are present,
- all values can be converted to numbers,
- values are compared with the training-data range.

The prediction response includes:

- predicted cultivar,
- probability for each cultivar,
- warnings for values outside the training range.

## Logging

The notebook uses `datafun_toolkit` logging.

Major steps and results are written to:

```text
project.log
```

The log provides evidence that the notebook:

- loaded the dataset,
- trained both models,
- evaluated the models,
- performed cross-validation,
- saved the artifact,
- tested prediction validation,
- completed successfully.

## Run Instructions

From the project root, start JupyterLab:

```powershell
uv run jupyter lab
```

Open:

```text
notebooks/module6_wine_serving_femi_rewritten.ipynb
```

Then use:

```text
Kernel → Restart Kernel and Run All Cells
```

After the notebook completes, verify that these files exist:

```text
project.log
artifacts/wine_model_bundle.joblib
```

## Results

Complete this section with the values written by the notebook.

| Metric | Result |
|---|---:|
| Decision Tree test accuracy | Add after running |
| Random Forest test accuracy | Add after running |
| Mean cross-validation accuracy | Add after running |
| Cross-validation standard deviation | Add after running |
| Most important feature | Add after running |

## Insights

The completed results should be used to answer these questions:

- Which model performed better on the held-out test set?
- Did cross-validation support the held-out result?
- Which chemical measurements were most important?
- Did the model confuse any of the three cultivars?
- Were any prediction inputs outside the observed training range?

## Conclusion

This custom project applies Module 6 model-serving skills to a new multiclass classification problem.

The project goes beyond training a model by creating a reusable artifact that contains the model and the information needed to serve it safely.
The addition of cross-validation, feature importance, prediction probabilities, metadata, and input validation makes the workflow more complete and reliable.

## Future Work

The next step is to create a FastAPI service that loads the saved wine model bundle and exposes a `/predict` endpoint.

The service could also use a Pydantic request model to enforce required fields and numeric validation before the request reaches the prediction function.
