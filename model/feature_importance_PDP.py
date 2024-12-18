import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import PartialDependenceDisplay


def plot_feature_importance_and_pdp(model_pipeline, X_train, y_train, top_n=5):
    """
    Plot feature importance for all features and generate PDPs for top N features.

    Parameters:
        model_pipeline: Trained pipeline including preprocessor and model
        X_train: Training data (raw)
        y_train: Training target labels
        top_n: Number of top features to plot PDPs
    """

    # Extract preprocessor and model from pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    model = model_pipeline.named_steps['classifier']

    # Transform training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    # Extract and sort feature importances
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    # All Features Importance Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[sorted_indices], y=[feature_names[i] for i in sorted_indices], orient='h')
    plt.title("Feature Importance - All Features")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    # Top 5 Features
    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    print(f"Top {top_n} Features:")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature} - Importance: {importances[sorted_indices[i]]:.4f}")

    # Partial Dependence Plots for Top Features
    num_classes = len(model.classes_)
    targets = model.classes_[:min(num_classes, 3)]  # Up to 3 target classes for PDPs
    n_rows = len(top_features)
    n_cols = len(targets)

    # PDP Subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.atleast_2d(axes)  # Ensure 2D array for axes handling

    for i, feature in enumerate(top_features):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            PartialDependenceDisplay.from_estimator(
                model,
                X_train_transformed,
                features=[feature],
                feature_names=feature_names,
                target=target,
                ax=ax,
                kind="average",
            )
            ax.set_title(f"PDP: {feature} (Class {target})")

    plt.tight_layout()
    plt.show()


def plot_glm(model_pipeline, X_train, y_train, top_n=5):
    """
    Plot feature coefficients (importance) for GLM and generate PDPs for the top N features.

    Parameters:
        model_pipeline: Trained pipeline including preprocessor and model
        X_train: Training data (raw)
        y_train: Training target labels
        top_n: Number of top features to plot PDPs
    """

    # Extract preprocessor and model from pipeline
    preprocessor = model_pipeline.named_steps['preprocessor']
    model = model_pipeline.named_steps['classifier']

    # Transform training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    # Extract coefficients and calculate importance (absolute values)
    coefficients = np.abs(model.coef_).mean(axis=0)
    sorted_indices = np.argsort(coefficients)[::-1]

    # All Features Coefficient Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=coefficients[sorted_indices], y=[feature_names[i] for i in sorted_indices], orient='h')
    plt.title("Feature Importance - GLM (Coefficients)")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Features")
    plt.show()

    # Top N Features
    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    print(f"Top {top_n} Features:")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature} - Coefficient: {coefficients[sorted_indices[i]]:.4f}")
