import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import plot_tree


def plot_phik(data, figsize=(12, 8), threshold=0.15):
    phik_matrix = data.phik_matrix()
    # Create a mask for values with value below threshold
    mask = phik_matrix < threshold
    # Apply mask to hide values below threshold
    phik_matrix_masked = phik_matrix.mask(mask)
    plt.figure(figsize=figsize)
    sns.heatmap(phik_matrix_masked, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.show()


def plot_hist_numeric(data, feature, figsize=(8, 4), x_min=None, x_max=None):
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]

    plt.figure(figsize=figsize)
    plt.grid()
    sns.histplot(filtered_data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


def plot_hist_categorical(data, feature, figsize=(4, 4)):
    category_counts = data[feature].value_counts()
    category_counts = category_counts.sort_values(ascending=False)
    plt.figure(figsize=figsize)
    plt.grid()
    sns.barplot(
        x=category_counts.values,
        y=category_counts.index,
        hue=category_counts.index,  # Add this
        palette="viridis",
        orient="h",
        legend=False,
    )  # Add this
    plt.title(f"Distribution of {feature}")
    plt.ylabel(feature)
    plt.xlabel("Frequency")
    plt.show()


def plot_all_categorical(data, cols=3, figsize=(5, 4), features=None):
    """
    Строит barplot для указанных категориальных переменных или всех категориальных,
    если список не указан.

    Parameters:
        data (pd.DataFrame): набор данных
        cols (int): количество графиков в строке
        figsize (tuple): размер одной ячейки подграфика
        features (list or None): список категориальных признаков для построения.
                                 Если None — берутся все категориальные.
    """

    if features is None:
        categorical_features = data.select_dtypes(
            include=["object", "category"]
        ).columns
    else:
        categorical_features = features

    if len(categorical_features) == 0:
        print("Нет категориальных признаков для отображения.")
        return

    n_features = len(categorical_features)
    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        ax = axes[i]

        category_counts = data[feature].value_counts().sort_values(ascending=False)

        sns.barplot(
            x=category_counts.values,
            y=category_counts.index,
            hue=category_counts.index,
            palette="viridis",
            orient="h",
            legend=False,
            ax=ax,
        )

        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel(feature)
        ax.grid()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_all_numeric(
    data, cols=3, figsize=(5, 4), features=None, x_min=None, x_max=None
):
    """
    Строит гистограммы для всех числовых признаков или заданного списка признаков.

    Parameters:
        data (pd.DataFrame): набор данных.
        cols (int): количество графиков в строке.
        figsize (tuple): размер одной ячейки (одного подграфика).
        features (list or None): список числовых признаков для отображения.
                                 Если None — берутся все числовые признаки.
        x_min (float or None): нижняя граница значений для фильтрации.
        x_max (float or None): верхняя граница значений для фильтрации.
    """

    if features is None:
        numeric_features = data.select_dtypes(include=["number"]).columns
    else:
        numeric_features = features

    if len(numeric_features) == 0:
        print("Нет числовых признаков для отображения.")
        return

    n_features = len(numeric_features)
    rows = int(np.ceil(n_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = np.array(axes).flatten()

    for i, feature in enumerate(numeric_features):
        ax = axes[i]

        filtered_data = data.copy()
        if x_min is not None:
            filtered_data = filtered_data[filtered_data[feature] >= x_min]
        if x_max is not None:
            filtered_data = filtered_data[filtered_data[feature] <= x_max]

        sns.histplot(filtered_data[feature], kde=True, ax=ax, color="steelblue")

        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_numeric_vs_categorical_boxplot(
    data: pd.DataFrame,
    numerical_feature: str,
    categorical_feature: str,
    figsize: tuple = (8, 6),
    title: str = None,
    palette: str = "Set2",
    showfliers: bool = True,
):
    """
    Create a boxplot comparing a numerical feature against a categorical one using Seaborn.

    Parameters:
    -----------
    data : pd.DataFrame
        The input dataframe
    numerical_feature : str
        Name of the numerical feature to plot
    categorical_feature : str
        Name of the categorical feature to group by
    figsize : tuple, optional
        Figure size (width, height), default is (8, 6)
    title : str, optional
        Custom title for the plot
    palette : str, optional
        Color palette for the boxplot, default is "Set2"
    showfliers : bool, optional
        Whether to show outliers, default is True

    Returns:
    --------
    None
        Displays the boxplot
    """
    # Validate input columns
    if numerical_feature not in data.columns:
        raise ValueError(
            f"Numerical feature '{numerical_feature}' not found in DataFrame"
        )
    if categorical_feature not in data.columns:
        raise ValueError(
            f"Categorical feature '{categorical_feature}' not found in DataFrame"
        )

    # Check if numerical feature is actually numeric
    if not pd.api.types.is_numeric_dtype(data[numerical_feature]):
        raise TypeError(f"Feature '{numerical_feature}' is not numeric")

    # Create the plot
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=data,
        x=categorical_feature,
        y=numerical_feature,
        palette=palette,
        showfliers=showfliers,
    )

    # Set title and labels
    if title is None:
        title = f"Distribution of {numerical_feature} by {categorical_feature}"
    plt.title(title)
    plt.xlabel(categorical_feature)
    plt.ylabel(numerical_feature)

    # Rotate x-axis labels if there are many categories
    plt.xticks(rotation=45 if len(data[categorical_feature].unique()) > 5 else 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_categorical_relationship(df, col1, col2):
    # Абсолютные значения
    count_crosstab = pd.crosstab(df[col1], df[col2])

    # Доли по строкам (внутри col1)
    row_prop = pd.crosstab(df[col1], df[col2], normalize="index")

    # Доли по столбцам (внутри col2)
    col_prop = pd.crosstab(df[col1], df[col2], normalize="columns")

    # Фигура с 3 подграфиками по горизонтали
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Абсолютные значения
    sns.heatmap(count_crosstab, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Абсолютные значения\n{col1} vs {col2}")
    axes[0].set_xlabel(col2)
    axes[0].set_ylabel(col1)

    # 2. Доли внутри col1 (по строкам)
    sns.heatmap(row_prop, annot=True, fmt=".2f", cmap="Greens", ax=axes[1])
    axes[1].set_title(f"Доли внутри {col1} (по строкам)")
    axes[1].set_xlabel(col2)
    axes[1].set_ylabel(col1)

    # 3. Доли внутри col2 (по столбцам)
    sns.heatmap(col_prop, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2])
    axes[2].set_title(f"Доли внутри {col2} (по столбцам)")
    axes[2].set_xlabel(col2)
    axes[2].set_ylabel(col1)

    plt.tight_layout()
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str = None,
    target_colors: dict = None,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    figsize: tuple = (8, 6),
):
    """
    Строит scatter plot зависимости между двумя числовыми переменными.
    При наличии бинарной таргетной переменной — точки окрашиваются по её значению.
    Позволяет задать ограничения на оси X и Y.

    :param df: pandas DataFrame
    :param x_col: Название числовой переменной по оси X
    :param y_col: Название числовой переменной по оси Y
    :param target_col: (опционально) Название бинарной переменной для окраски точек
    :param target_colors: (опционально) Словарь вида {значение_таргета: цвет}
    :param x_min: (опционально) Минимальное значение оси X
    :param x_max: (опционально) Максимальное значение оси X
    :param y_min: (опционально) Минимальное значение оси Y
    :param y_max: (опционально) Максимальное значение оси Y
    """
    # Проверка колонок
    for col in [x_col, y_col, target_col] if target_col else [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Проверка бинарного таргета
    if target_col is not None:
        unique_vals = sorted(df[target_col].dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Таргет '{target_col}' должен быть бинарным (2 уникальных значения)."
            )

        # Палитра
        if target_colors is None:
            palette = {unique_vals[0]: "blue", unique_vals[1]: "red"}
        else:
            if not all(val in target_colors for val in unique_vals):
                raise ValueError(
                    f"target_colors должен содержать оба значения таргета: {unique_vals}"
                )
            palette = target_colors

    # Построение графика
    plt.figure(figsize=figsize)
    if target_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=target_col, palette=palette)
        plt.legend(title=target_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color="blue")

    # Ограничения осей
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_trend(df, freq="ME"):
    """
    Визуализирует изменение средней цены домов по времени.

    Параметры:
    -----------
    df : pd.DataFrame
        Таблица с колонками 'date' и 'price'
    freq : str
        Частота агрегации:
            'D' — по дням,
            'W' — по неделям,
            'ME' — по месяцам
    """
    # Убедимся, что колонка с датой — datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Проверим частоту
    allowed = ["D", "W", "ME"]
    if freq not in allowed:
        raise ValueError(f"freq должно быть одним из {allowed}")

    # Группируем по выбранной частоте
    price_by_period = df.resample(freq, on="date")["price"].mean()

    # Определим подписи для удобства
    freq_label = {"D": "дням", "W": "неделям", "ME": "месяцам"}[freq]

    # Построим график
    plt.figure(figsize=(12, 6))
    plt.plot(price_by_period.index, price_by_period.values, marker="o", linewidth=2)
    plt.title(f"Динамика средней цены домов по {freq_label}")
    plt.xlabel("Дата")
    plt.ylabel("Средняя цена")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_classification_results(metrics, model_name="Model"):
    """
    Plot classification evaluation results

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_classification_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    # Создаем сетку графиков в зависимости от доступных метрик
    available_plots = []
    if "Confusion Matrix" in metrics:
        available_plots.append("Confusion Matrix")
    if "ROC Curve" in metrics:
        available_plots.append("ROC Curve")
    if "PR Curve" in metrics:
        available_plots.append("PR Curve")

    num_plots = len(available_plots)
    if num_plots == 0:
        print("No plot data available in metrics")
        return

    plt.figure(figsize=(5 * num_plots, 5))

    plot_index = 1

    # Plot 1: Confusion Matrix
    if "Confusion Matrix" in metrics:
        plt.subplot(1, num_plots, plot_index)
        sns.heatmap(
            metrics["Confusion Matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
        )
        plt.title(f"{model_name} - Confusion Matrix", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plot_index += 1

    # Plot 2: ROC Curve (if available)
    if "ROC Curve" in metrics:
        roc_data = metrics["ROC Curve"]
        plt.subplot(1, num_plots, plot_index)
        plt.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {metrics['ROC AUC']:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic", fontsize=14)
        plt.legend(loc="lower right")
        plot_index += 1

    # Plot 3: Precision-Recall Curve (if available)
    if "PR Curve" in metrics:
        pr_key = "PR Curve"
        pr_data = metrics[pr_key]

        pr_auc = metrics["PR AUC"]

        plt.subplot(1, num_plots, plot_index)
        plt.plot(
            pr_data["recall"],
            pr_data["precision"],
            color="darkgreen",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.2f})",
        )

        # Добавляем базовый уровень (случайный классификатор)
        if "baseline" in pr_data:
            plt.plot(
                [0, 1],
                [pr_data["baseline"], pr_data["baseline"]],
                color="navy",
                lw=2,
                linestyle="--",
                label=f"Baseline (AP = {pr_data['baseline']:.2f})",
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14)
        plt.legend(loc="lower left")
        plot_index += 1

    plt.tight_layout()
    plt.show()


def print_classification_report(metrics, model_name="Model"):
    """
    Print classification evaluation report

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_classification_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    # Create metrics table
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "ROC AUC",
                "PR AUC",
                "F1 Score",
                "Precision",
                "Recall",
                "Accuracy",
            ],
            "Value": [
                f"{metrics['ROC AUC']:.4f}"
                if metrics["ROC AUC"] is not None
                else "N/A",
                f"{metrics['PR AUC']:.4f}" if metrics["PR AUC"] is not None else "N/A",
                f"{metrics['F1 Score']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['Accuracy']:.4f}",
            ],
        }
    )

    # Classification report dataframe
    class_report_df = pd.DataFrame(metrics["Classification Report"])

    # Display results
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("=" * 60)

    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))

    print("\n\nCLASSIFICATION REPORT:")
    print(class_report_df.to_string(index=False))

    print("\n" + "=" * 60)


def plot_regression_results(y_test, y_pred, model_name="Model"):
    """
    Plot regression results and residuals

    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted target values
    metrics : dict
        Calculated regression metrics
    model_name : str
        Model name
    """
    plt.figure(figsize=(14, 6))

    # === 1. Scatter: y_true vs y_pred ===
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=y_test, y=y_pred, s=40, color="dodgerblue", edgecolor="k", alpha=0.7
    )
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2)
    plt.xlabel("Истинные значения", fontsize=12)
    plt.ylabel("Предсказанные значения", fontsize=12)
    plt.title(f"{model_name} — Истинные vs Предсказанные", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    # === 2. Residuals Plot ===
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.scatterplot(
        x=y_pred, y=residuals, s=40, color="purple", edgecolor="k", alpha=0.7
    )
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Предсказанные значения", fontsize=12)
    plt.ylabel("Остатки (y_true - y_pred)", fontsize=12)
    plt.title(f"{model_name} — График остатков", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def print_regression_report(metrics, model_name="Model"):
    """
    Print regression metrics in tabular form

    Parameters:
    -----------
    metrics : dict
        Dictionary of regression metrics
    model_name : str
        Model name
    """
    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "MAPE", "R2"],
            "Value": [
                f"{metrics['MAE']:.4f}",
                f"{metrics['RMSE']:.4f}",
                f"{metrics['MAPE']:.4f}",
                f"{metrics['R2']:.4f}",
            ],
        }
    )

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} REGRESSION EVALUATION".center(60))
    print("=" * 60)

    print("\nОсновные метрики регрессии:")
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 60)


def plot_feature_importance(
    model, X_train, X_test, top_n=None, figsize=(10, 6), sample_size=1000
):

    # if len(X_train) > sample_size:
    # X_train = X_train.sample(sample_size, random_state=42)

    if len(X_test) > sample_size:
        X_test = X_test.sample(sample_size, random_state=42)

    explainer = shap.Explainer(model, X_train)

    # Определяем, поддерживает ли explainer параметр check_additivity
    explainer_type = type(explainer).__name__
    if "Tree" in explainer_type:
        shap_values = explainer(X_test, check_additivity=False)
    else:
        shap_values = explainer(X_test)

    if shap_values.values.ndim == 3:
        shap.summary_plot(
            shap_values[:, :, 1],
            plot_type="bar",
            max_display=top_n,
            show=False,
            plot_size=figsize,
        )
    else:
        shap.summary_plot(
            shap_values,
            plot_type="bar",
            max_display=top_n,
            show=False,
            plot_size=figsize,
        )

    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def visualize_decision_tree(
    model, feature_names, class_names=None, figsize=(20, 10), max_depth=None
):
    """
    Visualize the decision tree structure.

    Parameters:
    - model: Trained DecisionTree model
    - feature_names: List of feature names
    - class_names: List of class names (for classification)
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=max_depth,
    )
    plt.title("Decision Tree Visualization")
    plt.show()


def plot_hyperparam_search_results(
    results,
    score_key="mean_test_score",
    title="Hyperparameter Tuning Results",
    xtick_step=5,
):
    """
    Generic plot function for hyperparameter search results from GridSearchCV, RandomizedSearchCV,
    BayesSearchCV, or any source with similar output.

    Args:
        results (dict or pd.DataFrame): Search results. Must contain 'params' and score_key.
        score_key (str): Key for the score column (default 'mean_test_score').
        title (str): Plot title.
        xtick_step (int): Frequency of x-axis labels.
    """
    # Normalize input
    if isinstance(results, dict):
        params = results.get("params")
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if "params" in results.columns:
            df = pd.DataFrame(results["params"].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError("results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={"index": "Set #"})

    # Best score
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    # Plot
    plt.figure(figsize=(12, 6))
    x = df["Set #"]
    y = df[score_key]
    plt.plot(x, y, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)

    # Clean x-ticks
    plt.xticks(ticks=x[::xtick_step])

    # Highlight best
    plt.plot(
        df.loc[best_idx, "Set #"], best_score, "ro", label=f"Best: {best_score:.4f}"
    )
    plt.annotate(
        f"Best\n{best_score:.4f}",
        xy=(df.loc[best_idx, "Set #"], best_score),
        xytext=(df.loc[best_idx, "Set #"], best_score + 0.02),
        arrowprops=dict(facecolor="red", shrink=0.05),
        ha="center",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def compare_metrics_heatmap(
    df1,
    df2,
    df1_name="DF1",
    df2_name="DF2",
    figsize=(8, 4),
    annot_fontsize=10,
    title="Comparison of ML Metrics (Δ%)",
):
    """
    Compare two DataFrames of ML metrics and plot a heatmap of their percentage differences.

    Parameters:
    - df1, df2: DataFrames containing metrics for ML algorithms (algorithms as index, metrics as columns)
    - df1_name, df2_name: Names to display for each DataFrame in the comparison
    - figsize: Size of the output figure
    - annot_fontsize: Font size for annotations in heatmap
    - title: Title for the plot

    Returns:
    - A matplotlib Figure object
    - The delta DataFrame showing percentage differences
    """

    # Определяем направление метрик
    # True = "чем больше — тем лучше", False = "чем меньше — тем лучше"
    direction = {
        "MAE": False,
        "RMSE": False,
        "MAPE": False,
        "R2": True,
    }

    # Копируем структуру
    delta = pd.DataFrame(index=df1.index, columns=df1.columns, dtype=float)

    for metric in df1.columns:
        if metric in direction:
            if direction[metric]:  # "больше = лучше"
                delta[metric] = (df2[metric] - df1[metric]) / df1[metric] * 100
            else:  # "меньше = лучше"
                delta[metric] = (df1[metric] - df2[metric]) / df1[metric] * 100
        else:
            # если метрика неизвестна — считать "больше = лучше" по умолчанию
            delta[metric] = (df2[metric] - df1[metric]) / df1[metric] * 100

    # Create a custom red-white-green colormap
    colors = ["#ff2700", "#ffffff", "#00b975"]  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        delta,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={"label": f"Δ% ({df2_name} vs {df1_name})"},
    )

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, delta
