# Credit Rating Prediction using GLM and LGBM

Predicting corporate credit ratings is critical for assessing a company's financial stability and risk. This project uses machine learning models, specifically the Generalized Linear Model (GLM) and LightGBM (LGBM), to automate credit rating predictions based on financial indicators. Comparison of the models will be provided.

## Data source

The Dataset is from Source:https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/data
A list of 2029 credit ratings issued by major agencies such as Standard and Poors to big US firms (traded on NYSE or Nasdaq) from 2010 to 2016.

There are 30 features for every company of which 25 are financial indicators. They can be divided in:
1. Liquidity Measurement Ratios: currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding
2. Profitability Indicator Ratios: grossProfitMargin, operatingProfitMargin, pretaxProfitMargin, netProfitMargin, effectiveTaxRate, returnOnAssets, returnOnEquity, returnOnCapitalEmployed
3. Debt Ratios: debtRatio, debtEquityRatio
4. Operating Performance Ratios: assetTurnover
5. Cash Flow Indicator Ratios: operatingCashFlowPerShare, freeCashFlowPerShare, cashPerShare, operatingCashFlowSalesRatio, freeCashFlowOperatingCashFlowRatio


## Installation

You can install the environment using conda:

```bash
conda env create --file environment.yml
conda activate credit_rating 
```
Also, you can use pre-commit to check the code:

```bash
pre-commit run --all-files
```

## How to run the Model
1. Run the `main.py` in the model folder, it is the main analysis and the predicting for the data.
2. Clone this repository and you can directly run the `ead_cleaning.ipynb` for exploring the data
2. `_cleaning.py`,`_load_data.py`,`plotting.py`, contains functions for loading, cleaning and plotting the data
3. `_evaluation.py`,`_learning_curve.py`,`splitting.py`,`_model_training.py`,`feature_importance_PDP.py` and `_preprocessing.py` contains the functions needed in the `main.py`
4. `feature_engineering.py` contains a simple scikit-learn transformer named LogCap. You can run the unit test by running

```bash
pytest tests/test_feature_engineering.py
```



