# %%
import pandas as pd
import numpy as np
import sys
import os
from sklearn import preprocessing
from pathlib import Path
root =Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from preprocessing._preprocessing import encoder
from data import load_data


# %%

def clean(df_rating):
    # as we can seen in the eda part, the sample of default and the lowest risk are too small, thus we drop them
    df_rating = df_rating[df_rating['Rating']!='Lowest Risk'] # filter Lowest Risk
    df_rating = df_rating[df_rating['Rating']!='In Default']  # filter In Default
    df_rating.reset_index(inplace = True, drop=True) # reset index
    df = encoder(df_rating)
    # replace all the outliers
    df_rating_no_out = df.copy()
    for c in df_rating_no_out.columns[6:31]:
        q05 = df_rating_no_out[c].quantile(0.10)
        q95 = df_rating_no_out[c].quantile(0.90)
        iqr = q95 - q05 #Interquartile range
        fence_low  = q05-1.5*iqr
        fence_high = q95+1.5*iqr
        df_rating_no_out.loc[df_rating_no_out[c] > fence_high,c] = df_rating_no_out[c].quantile(0.25)
        df_rating_no_out.loc[df_rating_no_out[c] < fence_low,c] = df_rating_no_out[c].quantile(0.75)
    # apply the log transformation after normalize all the data to 0 to 1
    min_max_scaler = preprocessing.MinMaxScaler()
    for c in df_rating_no_out.columns[6:31]:
        df_rating[c] = min_max_scaler.fit_transform(df_rating[[c]].to_numpy())*1000
        df_rating_no_out[c] = df_rating_no_out[c].apply(lambda x: np.log10(x+0.01))
    # drop missing value
    df_rating_no_out = df_rating_no_out.dropna()
    # Define the new save path for the cleaned data
    output_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    cleaned_data_path = os.path.join(output_folder, "cleaned_data.parquet")
    df_rating_no_out.to_parquet(cleaned_data_path, index = False)


# %%
