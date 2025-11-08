import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import xgboost as xgb

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("predictive-maintenance-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Load Data
Xtrain_path = "hf://datasets/Bhargavi329/vehicle-predictive-maintenance/X_train.csv"
Xtest_path = "hf://datasets/Bhargavi329/vehicle-predictive-maintenance/X_test.csv"
ytrain_path = "hf://datasets/Bhargavi329/vehicle-predictive-maintenance/y_train.csv"
ytest_path = "hf://datasets/Bhargavi329/vehicle-predictive-maintenance/y_test.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print("Predictive maintenance data loaded successfully.")

# Feature and Target Columns
numeric_features = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp', # Corrected column name
    'Coolant temp'
]

target_col = 'Engine Condition'

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Model Definition
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Hyperparameter Grid
param_grid = {
    'xgbclassifier__n_estimators': [150],
    'xgbclassifier__max_depth': [5],
    'xgbclassifier__learning_rate': [0.05],
    'xgbclassifier__subsample': [0.8],
    'xgbclassifier__colsample_bytree': [0.8],
    'xgbclassifier__reg_lambda': [0.5]
}

# Model Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Training and MLflow Logging
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log hyperparameter results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })


    # Print Detailed Classification Reports
    print("\n=== TRAINING PERFORMANCE ===")
    print(f"Accuracy:  {train_report['accuracy']:.4f}")
    print(f"Precision: {train_report['1']['precision']:.4f}")
    print(f"Recall:    {train_report['1']['recall']:.4f}")
    print(f"F1 Score:  {train_report['1']['f1-score']:.4f}")

    print("\n=== TEST PERFORMANCE ===")
    print(f"Accuracy:  {test_report['accuracy']:.4f}")
    print(f"Precision: {test_report['1']['precision']:.4f}")
    print(f"Recall:    {test_report['1']['recall']:.4f}")
    print(f"F1 Score:  {test_report['1']['f1-score']:.4f}")

    print("\n Confusion matrices and classification metrics logged successfully to MLflow!")

    # Compute confusion matrices
    train_cm = confusion_matrix(ytrain, y_pred_train)
    test_cm = confusion_matrix(ytest, y_pred_test)

    # Training confusion matrix
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay(train_cm).plot(cmap='Blues', values_format='d')
    plt.title("Training Confusion Matrix")
    plt.savefig("train_confusion_matrix.png")
    mlflow.log_artifact("train_confusion_matrix.png", artifact_path="plots")

    # Test confusion matrix
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay(test_cm).plot(cmap='Blues', values_format='d')
    plt.title("Test Confusion Matrix")
    plt.savefig("test_confusion_matrix.png")
    mlflow.log_artifact("test_confusion_matrix.png", artifact_path="plots")

    print("Confusion matrices logged successfully to MLflow!")


    # Threshold Optimization (Based on F1-Score)
    from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, classification_report

    # Get predicted probabilities for the positive class
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    # Compute precision-recall pairs
    precision, recall, thresholds = precision_recall_curve(ytest, y_pred_test_proba)

    # Compute F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the best threshold (highest F1)
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"\n Best Threshold based on F1-score: {best_threshold:.2f}")

    # Apply the best threshold to get new predictions
    y_pred_optimal = (y_pred_test_proba >= best_threshold).astype(int)

    # Evaluate metrics using the optimal threshold
    print("\n Updated Model Performance at Optimal Threshold:")
    print(f"Accuracy:  {accuracy_score(ytest, y_pred_optimal):.2f}")
    print(f"Precision: {precision_score(ytest, y_pred_optimal):.2f}")
    print(f"Recall:    {recall_score(ytest, y_pred_optimal):.2f}")
    print(f"F1-Score:  {f1_score(ytest, y_pred_optimal):.2f}")

    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(ytest, y_pred_optimal))

    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("optimized_f1", f1_score(ytest, y_pred_optimal))
    mlflow.log_metric("optimized_precision", precision_score(ytest, y_pred_optimal))
    mlflow.log_metric("optimized_recall", recall_score(ytest, y_pred_optimal))
    mlflow.log_metric("optimized_accuracy", accuracy_score(ytest, y_pred_optimal))

    # Save Model
    model_path = "best_predictive_maintenance_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally at: {model_path}")

    # ---------------------------
    # Upload to Hugging Face
    # ---------------------------
    repo_id = "Bhargavi329/predictive_maintenance_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository '{repo_id}' exists. Uploading model...")
    except RepositoryNotFoundError:
        print(f"Creating new repository '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
        create_pr=False
    )

    print(f"Model uploaded successfully to Hugging Face: {repo_id}")
