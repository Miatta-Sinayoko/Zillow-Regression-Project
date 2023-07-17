# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

# Acquire
from env import host, user, password
##############################################   ACQUIRE     ##############################################

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
##############################################   CLEAN     ##############################################

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


    # Change dtype for columns needed
    df['sqft'] = df.sqft.astype(int)
    df['county'] = df.county.astype(int).astype(str)

    # Replace fips with county names streamlining
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
##############################################   SPLIT     ##############################################

def split_data(df):
    '''
    This function splits the clean zillow data 
    '''
    
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    # Split train_validate into train and validate datasets
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)

    return train, validate, test

##############################################  SCALE DATA    ##############################################

def scaled_zillow(train, validate, test):
    '''
    Scale the features in train, validate, and test using MinMaxScaler.

    Args:
        train (DataFrame): The training data.
        validate (DataFrame): The validation data.
        test (DataFrame): The test data.

    Returns:
        scaler (object): The MinMaxScaler object.
        train_scaled (DataFrame): The scaled training data.
        validate_scaled (DataFrame): The scaled validation data.
        test_scaled (DataFrame): The scaled test data.
    '''

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)
# scale
    train_scaled = pd.DataFrame( scaler.transform(train))
    validate_scaled = pd.DataFrame( scaler.transform(validate))
    test_scaled = spd.DataFrame( scaler.transform(test))

    train_scaled = train_scaled.rename(columns={0:'bathroom', 1:'bedroom', 2:'sqft', 3: 'home_value'})
    validate_scaled = validate_scaled.rename(columns={0:'bathroom', 1:'bedroom', 2:'sqft', 3:'home_value'})
    test_scaled = test_scaled.rename(columns={0:'bathroom', 1:'bedroom', 2:'sqft', 3: 'home_value'})
    
    return train_scaled, validate_scaled, test_scaled

##############################################   WRANGLE     ##############################################
def wrangle_zillow():
    '''This function acquires, cleans, and splits the zillow data.'''
    
    df = prep_zillow_data(get_zillow_data())

    train, validate, test = split_zillow(df)

    return train, validate, test
##############################################  MODEL SPLIT    ##############################################
def model_split(train, validate, test):
    '''This function splits the train, validate, test datasets from the target variable to prepare it for model.'''

    #train_validate, test = train_test_split(df, test_size = .2, random_state=311)

   #train, validate = train_test_split(train_validate, test_size = .25, random_state=311)
                                              


    X_train = train.drop(columns=['home_value', 'county'])

    y_train = train.home_value

    X_validate = validate.drop(columns=['home_value', 'county'])

    y_validate = validate.home_value

    X_test = test.drop(columns=['home_value', 'county'])

    y_test = test.home_value

    return X_train, y_train, X_validate, y_validate, X_test, y_test