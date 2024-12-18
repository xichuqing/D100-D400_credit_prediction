# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
root =Path(__file__).resolve().parent.parent
sys.path.append(str(root))
print(root)



def count_of_rating(df):
    """
    plot the count of credit ratings
    Parameter
    ----------
    df: pd.DataFrame
    """
    ax = df['Rating'].value_counts().plot(kind='bar',
                                                figsize=(8,4),
                                                title="Count of Rating by Type",
                                                grid=True)
    return ax


def hist(df):
    """
    plot the histogram: range from -13 to 13 based on the summary statistics
    Parameter
    ----------
    df: pd.DataFrame
    """
    numerical_features = ['currentRatio', 'quickRatio', 'cashRatio']
    df[numerical_features].hist(bins=20, range = (-13,13),figsize=(10, 6))
    plt.suptitle('Distribution of Numerical Features')
    hist = plt.show()
    return hist


def pairwise(df):
    """
    plot the relations between all the numerical features by pairwising
    Parameter
    ----------
    df: pd.DataFrame
    """
    numerical_features = ['currentRatio', 'quickRatio', 'cashRatio']
    sns.pairplot(df, vars=numerical_features, hue='Rating', palette='bright', diag_kind='kde')
    pair = plt.show()
    return pair


def sector(df):
    """
    plot the rating by different sectros
    Parameter
    ----------
    df: pd.DataFrame
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='Sector', order=df['Sector'].value_counts().index, hue='Rating', palette='coolwarm')
    plt.title('Ratings by Sector')
    plt.xlabel('Count')
    plt.ylabel('Sector')
    plt.legend(title='Rating', loc='upper right')
    sector = plt.show()
    return sector


def boxplot_for_outlier(df):
    """
    plot the sample numerical features outliers by rating
    Parameter
    ----------
    df: pd.DataFrame
    """
    sns.boxplot(data=df, y='cashRatio', x='Rating', palette='coolwarm')
    plt.title('Cash Ratio Outliers by Rating')
    a = plt.show()
    sns.boxplot(data=df, y='currentRatio', x='Rating', palette='coolwarm')
    plt.title('Current Ratio Outliers by Rating')
    b = plt.show()
    sns.boxplot(data=df, y='quickRatio', x='Rating', palette='coolwarm')
    plt.title('Quick Ratio Outliers by Rating')
    c = plt.show()
    return a, b, c


def predictors_heatmap(df):     
    """
    Select only numerical columns, 
    Compute the correlation matrix, 
    Mask for the upper triangle and PLot the heatmap
        
    Parameter
    ----------
    df: pd.DataFrame
    """
    numerical_df = df.select_dtypes(include=['float64'])
    correlation_matrix = numerical_df.corr() 
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='Blues', fmt=".1f", cbar=True)
    plt.title('Correlation Heatmap (Lower Triangle)')
    heatmap = plt.show() 
    return heatmap


def correlation_map(df):
    """
    this function is to plot correlation between the variables in the dataset
    to be mentioned, the variables in the dataset should be encoded 
    using the function 'encoder' in the _preprocessing file
    Parameter
    ----------
    df: pd.DataFrame
    """
    numerical_df = df.select_dtypes(include=['float64','int64'])
    correlation_matrix = numerical_df.corr() 
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Correlation Heatmap (Lower Triangle)')
    correlation_map = plt.show() 
    first_column = correlation_matrix.iloc[:, 0]
    print(first_column)
    return correlation_map, correlation_matrix


def box_distribution(df):
    """
    this function is to plot the distribution of variables
    it is worth to mention, for better display, the outlier should be filtered out form the dataset
    Parameter
    ----------
    df: pd.DataFrame
    """
    figure, axes = plt.subplots(nrows=8, ncols=3, figsize=(20,44))
    i = 0 
    j = 0

    for c in df.columns[6:30]:
        sns.boxplot(x=df.Rating, y=df[c], palette="Set3", ax=axes[i, j])
        if j == 2:
            j=0
            i+=1
        else:
            j+=1    
