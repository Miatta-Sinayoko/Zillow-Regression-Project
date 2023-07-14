# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import warnings
warnings.filterwarnings("ignore")

# Acquire
from env import host, user, password

# Create a function that retrieves the necessary connection URL.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection URL to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# Create function to retrieve zillow data
def get_zillow_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    filename = 'zillow.csv'

    # Verify if file exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    # Search database if file doesn't exist
    else:
        sql = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt,fips
            FROM properties_2017
            JOIN predictions_2017 USING (parcelid) 
            JOIN propertylandusetype USING (propertylandusetypeid)
            WHERE propertylandusetypeid = 261 AND transactiondate LIKE '2017%%';
            '''
        df = pd.read_sql(sql, get_connection('zillow'))
        df.to_csv(filename, index=False)
        return df

def prep_zillow_data(df):
    '''
    This function takes in the DataFrame from get_zillow_data
    and returns the DataFrame with preprocessing applied.
    '''
    # Drop null values
    df = df.dropna()

    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedroom',
        'bathroomcnt': 'bathroom',
        'calculatedfinishedsquarefeet': 'sqft',
        'taxvaluedollarcnt': 'home_value',
        'fips': 'county'
    })


    # Change the dtype for the necessary columns
    df['sqft'] = df.sqft.astype(int)
    df['county'] = df.county.astype(int).astype(str)

    # Replace the values for readability
    df = df.replace({'6037': 'Los Angeles', '6059': 'Orange', '6111': 'Ventura'})

    # Calculate IQR for removing outliers
    q3_bath, q1_bath = np.percentile(df.bathroom, [75, 25])
    iqr_bath = q3_bath - q1_bath

    q3_bed, q1_bed = np.percentile(df.bedroom, [75, 25])
    iqr_bed = q3_bed - q1_bed

    q3_sqft, q1_sqft = np.percentile(df.sqft, [75, 25])
    iqr_sqft = q3_sqft - q1_sqft

    q3_val, q1_val = np.percentile(df.home_value, [75, 25])
    iqr_val = q3_val - q1_val

    # Get rid of outliers
    # The 1.5 multiplied by the IQR is a standard deviation of 1.5. For a Poisson distribution, the standard deviation
    # equals the square root of the mean.
    df = df[~((df['bathroom'] < (q1_bath - 1.5 * iqr_bath)) | (df['bathroom'] > (q3_bath + 1.5 * iqr_bath)))]
    df = df[~((df['bedroom'] < (q1_bed - 1.5 * iqr_bed)) | (df['bedroom'] > (q3_bed + 1.5 * iqr_bed)))]
    df = df[~((df['sqft'] < (q1_sqft - 1.5 * iqr_sqft)) | (df['sqft'] > (q3_sqft + 1.5 * iqr_sqft)))]
    df = df[~((df['home_value'] < (q1_val - 1.5 * iqr_val)) | (df['home_value'] > (q3_val + 1.5 * iqr_val)))]

    return df

def split_data(df):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)

    # Split train_validate into train and validate datasets
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.churn)

    return train, validate, test

def min_max_scaler(X_train, X_validate, X_test):
    '''
    Scale the features in X_train, X_validate, and X_test using MinMaxScaler.

    Args:
        X_train (DataFrame): The training data.
        X_validate (DataFrame): The validation data.
        X_test (DataFrame): The test data.

    Returns:
        scaler (object): The MinMaxScaler object.
        X_train_scaled (DataFrame): The scaled training data.
        X_validate_scaled (DataFrame): The scaled validation data.
        X_test_scaled (DataFrame): The scaled test data.
    '''

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled
