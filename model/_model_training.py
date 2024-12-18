
import numpy as np
from sklearn.model_selection import GridSearchCV
from _learning_curve import plot_learning_curve



def train_model(pipeline, params, X_train, y_train,model_name):
    """
    to find the best estimator by using grid search, output is the best model selected
    this function is suitable for both GLM and LGBM
    parameters
    -----------
    pipeline: model pipeline
    params: parameters for tuning the model
    X_train: training data for predictors
    y_train: training data for target
    model_name: str, the name of the model you want to train
    """
    print(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(pipeline, params, scoring="accuracy", cv=5, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Plotting Learning Curve for {model_name}...")
    plot_learning_curve(best_model, X_train, y_train, f"Learning Curve - {model_name}",cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    return best_model
