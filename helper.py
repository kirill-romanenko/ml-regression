import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_validate

import plots as p


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def calculate_classification_metrics(y_test, y_pred, y_probs=None, pos_label=1):
    """
    Calculate classification performance metrics

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    metrics = {
        "ROC AUC": roc_auc_score(y_test, y_probs) if y_probs is not None else None,
        "F1 Score": f1_score(y_test, y_pred, average="macro"),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "Accuracy": (y_pred == y_test).mean(),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }

    # Additional metrics for classification report
    metrics["Classification Report"] = {
        "Class": ["Positive", "Negative"],
        "Precision": [
            precision_score(y_test, y_pred, pos_label=1),
            precision_score(y_test, y_pred, pos_label=0),
        ],
        "Recall": [
            recall_score(y_test, y_pred, pos_label=1),
            recall_score(y_test, y_pred, pos_label=0),
        ],
    }

    if y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        metrics["ROC Curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

        precision, recall, thresholds = precision_recall_curve(
            y_test, y_probs, pos_label=pos_label
        )
        pr_auc = auc(recall, precision)
        baseline = np.mean(y_test == pos_label)  # Процент положительных примеров

        metrics["PR Curve"] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "baseline": baseline,
        }
        metrics["PR AUC"] = pr_auc

    return metrics


def evaluate_classification(
    y_test, y_pred, y_probs=None, model_name="Model", enable_plot=True
):
    """
    Evaluate classification performance with comprehensive metrics and visualizations

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class (required for ROC AUC)
    model_name : str, optional
        Name of the model for display purposes
    enable_plot : bool, optional
        Whether to display plots and detailed reports

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    # Calculate all metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_probs)

    if enable_plot:
        # Generate plots
        p.plot_classification_results(metrics, model_name)

        # Print detailed report
        p.print_classification_report(metrics, model_name)

    # Return metrics dictionary (excluding plot data for cleaner output)
    return {
        k: v
        for k, v in metrics.items()
        if k
        not in ["Confusion Matrix", "ROC Curve", "PR Curve", "Classification Report"]
    }


def calculate_regression_metrics(y_test, y_pred):
    """
    Calculate regression performance metrics

    Parameters:
    -----------
    y_test : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    dict: Dictionary of regression metrics
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    return metrics


def evaluate_regression(y_test, y_pred, model_name="Model", enable_plot=True):
    """
    Evaluate regression performance with metrics and plots

    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Model predictions
    model_name : str
        Model name
    enable_plot : bool
        Whether to show plots

    Returns:
    --------
    dict: Dictionary of regression metrics
    """
    metrics = calculate_regression_metrics(y_test, y_pred)

    if enable_plot:
        p.plot_regression_results(y_test, y_pred, model_name)
        p.print_regression_report(metrics, model_name)

    return metrics


def train_evaluate_model(
    model, model_name, X_train, y_train, X_test, y_test, seed=None, classification=True
):
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        if hasattr(model, "seed"):
            model.set_params(seed=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    if classification:
        y_probs = model.predict_proba(X_test)[:, 1]  # For ROC curve

        # Evaluate
        metrics = evaluate_classification(
            y_test=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            model_name=model_name,
            enable_plot=False,
        )
    else:
        metrics = evaluate_regression(
            y_test, y_pred, model_name=model_name, enable_plot=False
        )

    return metrics


def exp_mae(y_true, y_pred):
    return mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))


def exp_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))


def exp_mape(y_true, y_pred):
    return mean_absolute_percentage_error(np.expm1(y_true), np.expm1(y_pred))


def exp_r2(y_true, y_pred):
    return r2_score(np.expm1(y_true), np.expm1(y_pred))


def train_evaluate_model_cv(
    model,
    model_name,
    X,
    y,
    preprocessor=None,
    cv=5,
    seed=None,
    classification=True,
    use_log_target=False,
):
    """
    Train and evaluate a model using cross-validation and optional preprocessing.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        preprocessor: Preprocessing pipeline (e.g., StandardScaler, OneHotEncoder)
        cv: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        if hasattr(model, "seed"):
            model.set_params(seed=seed)

    # Create or extend pipeline with preprocessor and model
    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, append the model to it
        preprocessor.steps.append(("model", model))
        pipeline = preprocessor
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    else:
        # No preprocessor, just use the model
        pipeline = model

    if classification:
        # Scoring metrics for cross-validation (using macro averaging)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }
    else:
        # Handle regression metrics
        if use_log_target:
            scoring = {
                "mae": make_scorer(exp_mae, greater_is_better=False),
                "rmse": make_scorer(exp_rmse, greater_is_better=False),
                "mape": make_scorer(exp_mape, greater_is_better=False),
                "r2": make_scorer(exp_r2),
            }
        else:
            # Standard metrics (no log-transform)
            scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "mape": "neg_mean_absolute_percentage_error",
                "r2": "r2",
            }

    # Perform cross-validation on training data
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    if classification:
        metrics = {
            "ROC AUC": cv_results["test_roc_auc"].mean(),
            "F1 Score": cv_results["test_f1"].mean(),
            "Precision": cv_results["test_precision"].mean(),
            "Recall": cv_results["test_recall"].mean(),
            "Accuracy": cv_results["test_accuracy"].mean(),
            "PR AUC": cv_results["test_pr_auc"].mean(),
        }
    else:
        metrics = {
            "MAE": abs(cv_results["test_mae"].mean()),
            "RMSE": abs(cv_results["test_rmse"].mean()),
            "MAPE": abs(cv_results["test_mape"].mean()),
            "R2": cv_results["test_r2"].mean(),
        }

    # p.plot_classification_results(metrics, model_name)

    return metrics


def train_evaluate_models_cv(
    models: list,
    X,
    y,
    preprocessor=None,
    cv=5,
    seed=None,
    classification=True,
    use_log_target=False,
):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)
        if preprocessor is not None:
            current_preprocessor = clone(preprocessor)
        else:
            current_preprocessor = None

        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model,
            model_name,
            X,
            y,
            current_preprocessor,
            cv,
            seed,
            classification=classification,
            use_log_target=use_log_target,
        )

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap="RdBu_r", annot=True, fmt=".2f")
    plt.title("Model Evaluation Metrics Comparison")
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Train and evaluate multiple classification models, then display a heatmap of the metrics.

    Parameters:
    -----------
    models : list
        List of tuples containing (model_name, model_instance) where model_instance is a scikit-learn compatible classifier
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    preprocessor : Pipeline or Transformer, optional
        Preprocessing pipeline to apply to the data before training
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all evaluation metrics for all models
    """

    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed
        )

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap="RdBu_r", annot=True, fmt=".2f")
    plt.title("Model Evaluation Metrics Comparison")
    plt.tight_layout()
    plt.show()

    return metrics_df


def winsorize_outliers(df, column_name, lower_bound=None, upper_bound=None):
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df
