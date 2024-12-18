from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
root =Path(__file__).resolve().parent.parent
sys.path.append(str(root))


def encoder(df):
    """
    to encode all needed categorical variables
    higher risk with higher number
    Parameter
    -----------
    df: pd.DataFrame
    output: data frame with encoded categorical variables
    """
    le_sector = preprocessing.LabelEncoder()
    le_sector.fit(df.Sector)
    df.Sector = le_sector.transform(df.Sector)  # encode sector
    print("Sector encoding mapping:", dict(enumerate(le_sector.classes_)))
    rating_mapping = {
    'Lowest Risk': 1,
    'Low Risk': 2,
    'Medium Risk': 3,
    'High Risk': 4,
    'Highest Risk': 5,
    'In Default': 6
}
    # Apply mapping to encode 'Rating'
    df['Rating'] = df['Rating'].map(rating_mapping)
    
    return df
