import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
 ##############################################VISUALIZE##############################

# def plot_variable_pairs(df):
#     '''
#     Plots scatter plots of all pairs of continuous variables in the DataFrame.

#     Args:
#         df (pd.DataFrame): The DataFrame to plot.
#     '''

#     df = df.sample(n=10000)
#     cols = ['bedroom', 'bathroom', 'sqft']
#     target = 'home_value'

#     for col in cols:
        
#         sns.lmplot(data=df, x=col, y=target, hue='county')

#     return sns.lmplot(df, x = col, y = target, hue='country')

# def plot_categorical_and_continuous_vars(df, cat_var_col, con_var_col):
#     '''
#     Plots a strip plot, box plot, and swarm plot of a categorical variable vs. a continuous variable.

#     Args:
#         df (pd.DataFrame): The DataFrame to plot.
#         cat_var_col (str): The name of the categorical variable.
#         con_var_col (str): The name of the continuous variable.
#     '''

#     df = df.sample(n=1000)
#     fig, axs = plt.subplots(1, 3, figsize=(18, 8))

#     sns.stripplot(ax=axs[0], x=cat_var_col, y=con_var_col, data=df)
#     axs[0].set_title('Strip plot')

#     sns.boxplot(ax=axs[1], x=cat_var_col, y=con_var_col, data=df)
#     axs[1].set_title('Box plot')

#     sns.swarmplot(ax=axs[2], x=cat_var_col, y=con_var_col, data=df, s=1)
#     axs[2].set_title('Swarm plot')


# def ols_lasso_tweedie(X_train, X_validate, y_train, y_validate, metric_df):
#     '''
#     Fits and evaluates an OLS, LassoLars, and TweedieRegressor on the training and validation sets.

#     Args:
#         X_train (pd.DataFrame): The training set features.
#         X_validate (pd.DataFrame): The validation set features.
#         y_train (pd.DataFrame): The training set target.
#         y_validate (pd.DataFrame): The validation set target.
#         metric_df (pd.DataFrame): A DataFrame to store the model metrics.

#     Returns:
#         pd.DataFrame: The updated metric_df.
#     '''
#     # Make OLS model fit OLS model
#     lm = LinearRegression()
    
#     OLSmodel = lm.fit(X_train, y_train.value)
    
#     # Formulate predictions and save to y_train
#     y_train['value_pred_ols'] = lm.predict(X_train)
    
#     # Evaluate
#     rmse_train_ols = mean_squared_error(y_train.value, y_train.value_pred_ols) ** 0.5
   
#     # Predict validate
#     y_validate['value_pred_ols'] = lm.predict(X_validate)
     
#      # Evaluate RMSE for validate
#     rmse_validate_ols = mean_squared_error(y_validate.value,y_validate.value_pred_ols) ** 0.5
  
#     # Append metric
#     metric_df = metric_df.append({
#         'model': 'ols',
#         'RMSE_train': rmse_train_ols,
#         'RMSE_validate': rmse_validate_ols,
#         'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_ols)
#     }, ignore_index=True)
   
#     print(f"""
#         RMSE for OLS using LinearRegression
#         Training/In-Sample: {rmse_validate_ols}""")

   
#  # LassoLars model
# lars = LassoLars(alpha=0.03)
# Larsmodel = lars.fit(X_train, y_train.value)
# y_train['value_pred_lars'] = lars.predict(X_train)
# rmse_train_lars = mean_squared_error(y_train.value, y_train.value_pred_lars) ** .5
# y_validate['value_pred_lars'] = lars.predict(X_validate)
# rmse_validate_lars = mean_squared_error(y_validate.value, y_validate.value_pred_lars) ** .5
# metric_df = metric_df.append({
#     'model': 'lasso_alpha0.03',
#     'RMSE_train': rmse_train_lars,
#     'RMSE_validate': rmse_validate_lars,
#     'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_lars)    
# }, ignore_index=True)
# print(f"""RMSE for LassoLars
#     Training/In-Sample:  {rmse_train_lars} 
#     Validation/Out-of-Sample: {rmse_validate_lars}""")

# # TweedieRegressor model make and fit 
# tr = TweedieRegressor(power=1, alpha=1.0)

# Tweediemodel = tr.fit(X_train, y_train.value)

# # Formulate predictions and save to y_train
# y_train['value_pred_tweedie'] = tr.predict(X_train)

# # Evaluate
# rmse_train_tweedie = mean_squared_error(y_train.value, y_train.value_pred_tweedie) ** .5

# # Predict validate
# y_validate['value_pred_tweedie'] = tr.predict(X_validate)

# # Evaluate RMSE for  validate 
# rmse_validate_tweedie = mean_squared_error(y_validate.value, y_validate.value_pred_tweedie) ** .5

# #Append metric
# metric_df = metric_df.append({
#     'model': 'tweedie_power1_alpha1.0',
#     'RMSE_train': rmse_train_tweedie,
#     'RMSE_validate': rmse_validate_tweedie,
#     'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_tweedie)    
# }, ignore_index=True)

# print(f"""RMSE for TweedieRegressor
#     Training/In-Sample:  {rmse_train_tweedie} 
#     Validation/Out-of-Sample: {rmse_validate_tweedie}""")

# return y_train, y_validate, metric_df

    
    
    
def get_bedroom_graph(df):
    '''This function takes a sample of the train data and returns a lmplot between bedcount and home value.'''
    
    df = df.sample(n=1000)
    
    sns.lmplot(data=df, x='bedroom', y='home_value')
    
    plt.ylabel('Home Value')
    plt.xlabel('Number of Bedrooms')
    plt.title('Is there a relationship between the number of bedrooms and home value?')
    
    plt.show()


def get_bathroom_graph(df):
    '''This function takes a sample of the train data and returns a lmplot between bathcount and home value.'''
    
    df = df.sample(n=1000)
    
    sns.lmplot(data=df, x='bathroom', y='home_value')
    
    plt.ylabel('Home Value')
    plt.xlabel('Number of Bathrooms')
    plt.title('Is there a relationship between the number of bathrooms and home value?')
    
    plt.show()

def get_sqft_graph(df):
    '''This function takes a sample of the train data and returns a lmplot between square footage and home value.''' 

    df = df.sample( n = 1000) 

    sns.lmplot(df, x = 'sqft', y = 'home_value')

    plt.ylabel('Home Value')
    plt.xlabel('Total Square Feet')
    plt.title('Is Square Footage Related to Home Value?')

    plt.show()

def get_county_graph(df):
    '''This function takes a sample of the train data and returns swarmplot of county and home value.'''
    
    df = df.sample(n=1000)

    #stripplot
    sns.stripplot(data=df, x= 'county', y= 'home_value',  s=1, hue= 'county', legend=False)
    plt.title('Distribution of Home Values in Los Angeles, Orange, and Ventura Counties - Stripplot')
    plt.ylabel("Home Value")
    plt.show()


    #swarmplot
    sns.swarmplot(data=df, x= 'county', y= 'home_value',  s=1, hue= 'county', legend=False)
    plt.title('Distribution of Home Values in Los Angeles, Orange, and Ventura Counties - Swarmplot')
    plt.ylabel("Home Value")
    plt.show()


def plot_variable_pairs(df):
    
    df = df.sample(n = 10000)
    
    cols = ['bedroom', 'bathroom', 'sqft']
    
    target = 'home_value'
    
    for col in cols:
        
        sns.lmplot(df, x = col, y = target, hue='county')


def plot_categorical_and_continuous_vars(df, cat_var_col, con_var_col):
    '''This function graphs a swarmplot that shows the density of home values within each county.'''
    
    # sample the data to make the graph readable
    df = df.sample(n=1000)
    
    plt.figure(figsize=(15,8)) 

    fig, axs = plt.subplots(1,3, figsize=(18,8))
    
    sns.stripplot(ax=axs[0], x=cat_var_col, y=con_var_col, data=df)
    axs[0].set_title('stripplot')
    
    sns.boxplot(ax=axs[1], x=cat_var_col, y=con_var_col, data=df)
    axs[1].set_title('boxplot')

    plt.title('Distribution of Home Values in Los Angeles, Orange, and Ventura Counties')
    
    sns.swarmplot(x=cat_var_col, y=con_var_col, data=df, s=1, hue=cat_var_col)
    #axs[2].set_title('swarmplot')

    plt.show()



    ##############################################   MODEL     ##############################################

def get_baseline(y_train, y_validate):
    '''This function gets the baseline for modeling and creates a metric df '''
    
    # change the target to databases 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # add the mean baseline to the db
    value_pred_mean = y_train.value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean 

    # RMSE of baseline
    rmse_train = mean_squared_error(y_train.value, y_train.value_pred_mean) ** (.5)

    rmse_validate = mean_squared_error(y_validate.value, y_validate.value_pred_mean) ** (.5)

    # create a df to easily view results of models
    metric_df = pd.DataFrame(data = [
        {
            'model': "mean_baseline",
            'RMSE_train': rmse_train,
            'RMSE_validate': rmse_validate,
            'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_mean)
        }
    ])

    return y_train, y_validate, metric_df



def ols_lasso_tweedie(X_train, X_validate, y_train, y_validate, metric_df):
    ''' This function runs train and validate test set data'''

    # make and fit OLS model
    lm = LinearRegression()

    OLSmodel = lm.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_ols'] = lm.predict(X_train)

    #evaluate RMSE
    rmse_train_ols = mean_squared_error(y_train.value, y_train.value_pred_ols) ** .5

    # predict validate
    y_validate['value_pred_ols'] = lm.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_ols = mean_squared_error(y_validate.value, y_validate.value_pred_ols) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'ols',
        'RMSE_train': rmse_train_ols,
        'RMSE_validate': rmse_validate_ols,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_ols)    
    }, ignore_index=True)

    print(f"""RMSE for OLS using LinearRegression
        Training/In-Sample:  {rmse_train_ols:.2f} 
        Validation/Out-of-Sample: {rmse_validate_ols:.2f}\n""")


    
    # make and fit OLS model
    lars = LassoLars(alpha=0.03)

    Larsmodel = lars.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_lars'] = lars.predict(X_train)

    #evaluate RMSE
    rmse_train_lars = mean_squared_error(y_train.value, y_train.value_pred_lars) ** .5

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_lars = mean_squared_error(y_validate.value, y_validate.value_pred_lars) ** .5

    #append metric
    metric_df = metric_df.append({
        'model': 'lasso_alpha0.03',
        'RMSE_train': rmse_train_lars,
        'RMSE_validate': rmse_validate_lars,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_lars)    
    }, ignore_index=True)

    print(f"""RMSE for LassoLars
        Training/In-Sample:  {rmse_train_lars:.2f} 
        Validation/Out-of-Sample: {rmse_validate_lars:.2f}\n""")


    # make and fit OLS model
    tr = TweedieRegressor(power=1, alpha=1.0)

    Tweediemodel = tr.fit(X_train, y_train.value)

    # make a prediction and save it to the y_train
    y_train['value_pred_tweedie'] = tr.predict(X_train)

    #evaluate RMSE
    rmse_train_tweedie = mean_squared_error(y_train.value, y_train.value_pred_tweedie) ** .5

    # predict validate
    y_validate['value_pred_tweedie'] = tr.predict(X_validate)

    # evaluate RMSE for validate
    rmse_validate_tweedie = mean_squared_error(y_validate.value, y_validate.value_pred_tweedie) ** .5

    # append metric
    metric_df = metric_df.append({
        'model': 'tweedie_power1_alpha1.0',
        'RMSE_train': rmse_train_tweedie,
        'RMSE_validate': rmse_validate_tweedie,
        'R2_validate': explained_variance_score(y_validate.value, y_validate.value_pred_tweedie)    
    }, ignore_index=True)

    print(f"""RMSE for TweedieRegressor
        Training/In-Sample:  {rmse_train_tweedie:.2f} 
        Validation/Out-of-Sample: {rmse_validate_tweedie:.2f}\n""")

    return y_train, y_validate, metric_df

def lasso_test_model(X_train, y_train, X_test, y_test):
    '''This function fits the Lasso Model on train and predicts for test data.'''

    # make and fit the model
    lars = LassoLars(alpha = 0.03)
    LarsModel = lars.fit(X_train, y_train.value)

    # predict with test data
    y_test_pred = lars.predict(X_test)

    # evaluate with RMSE
    rmse_test = mean_squared_error(y_test, y_test_pred) ** .5

    # calculate explained variance

    r2_test = explained_variance_score(y_test, y_test_pred)

    print(f"""RMSE for LassoLars:
    _____________________________________________      
    Test Performance: {rmse_test:.2f}
    Test Explained Variance: {r2_test:.3f}
    Baseline: {y_train.value.mean():.2f}""")

