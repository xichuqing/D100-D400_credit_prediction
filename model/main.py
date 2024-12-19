# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping
from splitting import create_sample_split
from feature_importance_PDP import plot_feature_importance_and_pdp, plot_glm
from _evaluation import evaluate_classification_predictions, evaluation
from _model_training import train_model
import sys
import os
from dalex import Explainer
import warnings
from pathlib import Path
root =Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from data._load_data import load_data
from data._cleaning import clean
from preprocessing import LogCap
from lightgbm.callback import early_stopping
# from warnings import simplefilter
# simplefilter(action='ignore')
warnings.filterwarnings("ignore")


# %%
# --------------------------------load the cleaned data--------------------------------#
# load the cleaned data to folder data and save as a parquet file
df = clean(load_data())


# %%
# --------------------------------preprocessing the data-----------------------------#

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Traverse to the root of the repository
repo_root = os.path.abspath(os.path.join(script_dir, '..'))  # Move up one level

# Construct the full path to the data file
file_path = os.path.join(repo_root, 'data', 'cleaned_data.parquet')
# Load the Parquet file
df = pd.read_parquet(file_path, engine='fastparquet')
# Print to verify the file path
print("File path:", file_path)
print("Data loaded successfully!")


# absolute_path = os.path.abspath('../data/cleaned_data.parquet')
# # Read the Parquet file
# df = pd.read_parquet(absolute_path, engine='fastparquet')

#split the data into testing and training
df = create_sample_split(df, 'ID', training_frac=0.8)
print(df.columns)
df_train = df[df["sample"] == "train"]
df_test = df[df["sample"] == "test"] 
predictors = ['Sector',
       'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
       'pretaxProfitMargin', 'grossProfitMargin',
       'operatingProfitMargin', 'returnOnAssets',
       'returnOnEquity', 'fixedAssetTurnover',
       'debtEquityRatio', 'debtRatio', 'effectiveTaxRate',
       'freeCashFlowOperatingCashFlowRatio',
       'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue',
       'enterpriseValueMultiple', 'operatingCashFlowPerShare',
       'operatingCashFlowSalesRatio', 'payablesTurnover']
target = 'Rating'
X_train = df_train[predictors]
y_train = df_train[target]
X_test = df_test[predictors]
y_test = df_test[target]

# Define categorical and numerical features
categoricals = ['Sector']  
numericals = list(set(predictors) - set(categoricals))


# %%
# ---------------------------------- Pipeline and Grid Parameters ---------------------------- #
preprocessor = ColumnTransformer(
    transformers=[
        ("num", LogCap(), numericals),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categoricals)
    ]
)

pipelines_and_params = {
    'GLM': {
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(multi_class="multinomial", solver="saga", max_iter=1000))
        ]),
        'params': {
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__max_iter": [500],
            "classifier__penalty": ["l1"],
            "classifier__l1_ratio": [0.1, 0.3]  
        }
    },
    'LGBM': {
        'pipeline': Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(objective="multiclass", n_jobs=-1))
        ]),
        'params': {
            'classifier__n_estimators': [100],
            'classifier__learning_rate': [0.01,0.05],
            'classifier__num_leaves': [15,31,45],
            'classifier__min_child_weight': [3,6],
            'classifier__lambda_l1': [0.1,0.3],
            
        }
    }
}


# %%
# -------------------------------------Model training-----------------------------------------------#
best_glm, best_glm_param = train_model(pipelines_and_params['GLM']['pipeline'], 
                       pipelines_and_params['GLM']['params'],
                       X_train,y_train, "GLM")

explainer = Explainer(best_glm, X_train, y_train, 
                      label="Multinomial Logistic regression")

# %%
best_lgbm, best_lgbm_param = train_model(pipelines_and_params['LGBM']['pipeline'], 
                        pipelines_and_params['LGBM']['params'], 
                        X_train,y_train,"LGBM")

explainer = Explainer(best_lgbm, X_train, y_train, 
                      label="LGBM")


# %%
# --------------------------------------Model evaluation---------------------------------------------#
evaluation1 = evaluation(df_test,X_test, best_glm,'GLM')
# %%
evaluation2 = evaluation(df_test,X_test, best_lgbm,'LGBM')


# %%
# ------------------------------------Feature selection and interpretation------------------------------#
# Feature importance plotting and Partial dependences plots for LGBM
print("LGBM Feature Importance and PDPs...")
plot_feature_importance_and_pdp(best_lgbm, X_train, y_train, top_n=5)
# %%
# Feature importance plotting  for GLM
print("GLM Feature Coefficients and PDPs...")
plot_glm(best_glm, X_train, y_train, top_n=5)


# %%
# -----------------------------------------------early stopping and the evaluation---------------------------------------#
# now we add the early stopping to see if it increase the performance

tuned_model = LGBMClassifier(objective="multiclass", n_jobs=-1, **best_lgbm_param)
early_stopping_model = tuned_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],  # Pass validation set
    eval_metric="multi_logloss",  # Evaluation metric
    callbacks =[early_stopping(stopping_rounds=20)]     # Use early stopping
)

evaluation3 = evaluation(df_test,X_test, early_stopping_model,'LGBM with early stopping')
# %%
