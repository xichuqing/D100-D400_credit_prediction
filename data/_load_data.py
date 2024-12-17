import numpy as np
import pandas as pd

def load_data():
    """Load data from Kaggle
    Source:https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/data
    For simplicity, we download the data and put it in the data folder, which named 'corporate_rating.csv'
    Summary of transformations:
    1. create an id column for the id-based data splitting
    2. asign the 'Rating' according to the risk classification

    output: new csv file for corprorate rating data with a new id column
    """
    #load data
    df = pd.read_csv("corporate_rating.csv")
    #create an id column
    df["ID"] = range(1, len(df)+1)
    rating_dict = {'AAA':'Lowest Risk', 
               'AA':'Low Risk',
               'A':'Low Risk',
               'BBB':'Medium Risk', 
               'BB':'High Risk',
               'B':'High Risk',
               'CCC':'Highest Risk', 
               'CC':'Highest Risk',
               'C':'Highest Risk',
               'D':'In Default'}

    df.Rating = df.Rating.map(rating_dict)
    return df


