from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def evaluate_classification_predictions(df, outcome_column, *, preds_column=None, model=None, predictors=None, preprocessor=None):
    """
    Evaluate classification predictions against actual outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe used for evaluation
    outcome_column : str
        Name of the true outcome column
    preds_column : str, optional
        Name of the predictions column, by default None
    model :
        Fitted model, by default None
    predictors : list, optional
        List of predictors used for the model, required if model is provided
    preprocessor : Pipeline or ColumnTransformer
        Preprocessing pipeline used to transform the predictors

    Returns
    -------
    evals : pd.DataFrame
        DataFrame containing classification evaluation metrics
    confusion_matrix : np.ndarray
        Confusion matrix
    """
    evals = {}

    # Ensure either predictions or a model is provided
    assert preds_column or model, "Please either provide the column name of the pre-computed predictions or a model to predict from."

    # Generate predictions if model is provided
    if preds_column is None:
        X = df[predictors]
        if preprocessor:  # Apply preprocessing if available
            X = preprocessor.transform(X)
        preds = model.predict(X)
    else:
        preds = df[preds_column]

    y_true = df[outcome_column]

    # Compute standard metrics
    evals["accuracy"] = accuracy_score(y_true, preds)
    evals["precision"] = classification_report(y_true, preds, output_dict=True)['weighted avg']['precision']
    evals["f1_score"] = classification_report(y_true, preds, output_dict=True)['weighted avg']['f1-score']
    evals["cohen_kappa"] = cohen_kappa_score(y_true, preds)
    evals["mcc"] = matthews_corrcoef(y_true, preds)

    # Confusion matrix (return separately)
    cm = confusion_matrix(y_true, preds)

    return pd.DataFrame(evals, index=[0]).T, cm


def evaluation(df_test,X_test,best_model,name):
    """
    For displaying the plottings and matrix of model evaluation
    Parameters
    ---------
    df_test: pd.DataFrame
        the testing data
    X_test: pd.DataFrame
        Predictors' testing data
    best_model: the trained best model 
    name: str
        the model name
    """
    df_test["predictions"] = best_model.predict(X_test)
    eval_results, cm = evaluate_classification_predictions(
        df=df_test, outcome_column='Rating', preds_column='predictions', model=None, predictors=None
    )
    #confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")    
    plt.show()
    print(eval_results)

