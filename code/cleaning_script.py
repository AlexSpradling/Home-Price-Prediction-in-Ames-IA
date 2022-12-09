import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


NOMINAL = ['ms_subclass', 'ms_zoning', 'street', 'alley', 'land_contour', 'lot_config', 'neighborhood', 'condition_1', 'condition_2', 'bldg_type',
'house_style', 'roof_style', 'roof_matl', 'exterior_1st', 'exterior_2nd', 'mas_vnr_type', 'foundation', 'heating', 'central_air', 'garage_type', 'misc_feature',
'sale_type']

ORDINAL = ['lot_shape', 'utilities', 'land_slope', 'house_style', 'overall_qual', 'overall_cond', 'exter_qual', 'exter_cond', 'bsmt_qual',
'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1', 'bsmtfin_type_2', 'heating_qc', 'electrical', 'kitchen_qual', 'functional', 'fireplace_qu', 'garage_finish',
'garage_qual', 'garage_cond', 'paved_drive', 'pool_qc', 'fence']

DISCRETE = ['year_built', 'year_remod/add', 'bsmt_full_bath', 'bsmt_half_bath', 'full_bath', 'half_bath', 'bedroom_abvgr','kitchen_abvgr',
'totrms_abvgrd', 'fireplaces', 'garage_yr_blt', 'garage_cars', 'mo_sold', 'yr_sold']

CONTINUOUS = ['lot_frontage', 'lot_area', 'mas_vnr_area', 'bsmtfin_sf_1', 'bsmtfin_sf_2','bsmt_unf_sf',
'total_bsmt_sf', '1st_flr_sf', '2nd_flr_sf', 'gr_liv_area', 'garage_area', 'wood_deck_sf', 'open_porch_sf',
'enclosed_porch', '3ssn_porch', 'screen_porch', 'pool_area','misc_val']

# functions to rank Ordinal quality ratings
def ordinal_to_numeric_expo(val):
    """
    Takes in a string value and returns a numeric value based on the ordinal scale
    """
    if val == 'Ex':
        return 4
    elif val == 'Gd':
        return 3
    elif val == 'TA':
        return 2
    elif val == 'Fa':
        return 1
    elif val == 'Po':
        return 0
    else:
        pass

def ordinal_to_numeric_glqna(val):
    """
    Takes in a string value and returns a numeric value based on the ordinal scale
    """
    if val == 'GLQ':
        return 6
    elif val == 'ALQ':
        return 5
    elif val == 'BLQ':
        return 4
    elif val == 'Rec':
        return 3
    elif val == 'LwQ':
        return 2
    elif val == 'Unf':
        return 1
    elif val == 'NA':
        return 0
    else:
        pass

def ordinal_to_numeric_functional(val):
    """
    Takes in a string value and returns a numeric value based on the ordinal scale
    """
    if val == 'Typ':
        return 3
    elif val == 'Min1':
        return 2
    elif val == 'Min2':
        return 1
    elif val == 'Mod':
        return 0
    elif val == 'Maj1':
        return -1
    elif val == 'Maj2':
        return -2
    elif val == 'Sev':
        return -3
    elif val == 'Sal':
        return -4
    else:
        pass


def plot_corr_heatmap(df, CATEGORY, title):
    """
    Takes in a dataframe, a list of columns to plot, and a title for the plot, 
    and returns a heatmap of the correlations
    """
    COLUMNS = [x for x in CATEGORY]
    COLUMNS.append('saleprice')

    fig, ax = plt.subplots(figsize=(20,12))
    corr = df[COLUMNS].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.title(f'{title} Correlations')
    
    return sns.heatmap(corr, mask=mask, annot=True, fmt='.1f'), corr['saleprice'], corr



def outlier_index(df, col, thresh):
    """
    Takes in a dataframe, a column name, and a threshold for z-score
    Returns a tuple of the column name and the indices of the outliers
    """
    z = np.abs(stats.zscore(df[col].astype(float)))
    return col, np.where(z > thresh)

# given an np array of indices, return the values at those indices
def get_outliers(df, col, indices):
    """
    Takes in a dataframe, a column name, and a list of indices
    Returns a list of the values at those indices
    """
    return df[col].iloc[indices]

def unique_by_col(category, df):
    """
    Takes in a dataframe and a category name
    Returns a dictionary of the unique values in each column of the category
    """
    items = {}
    for col in df[category].columns:
        items.update({col:df[category][col].unique()}) 
    return items

def bucketize_missing(df):
    """
    Takes in a dataframe and returns a tuple of three dictionaries
    The first dictionary contains columns with more than 20% of their data missing
    The second dictionary contains columns with less than 20% of their data missing
    The third list contains columns with no missing data
    """
    # missing more than 20 % of their data
    missing_the_most = {}
    # missing anything
    missing_some = {}
    # not missing anything
    full = []

    # iterate through the columns, find any columns with missing data, append to appropriate dict or list
    for idx, item in df.isnull().sum().items():
        missing_amount = round(item/len(df),4)
        if missing_amount > .20:
            missing_the_most.update({idx:missing_amount})
        elif item >= 1:
            missing_some.update({idx:missing_amount})
        else:
            full.append(idx)
    return missing_the_most, missing_some, full

def get_regression(df, regression_type, features):
    X = df[features]
    y = df['saleprice']

def feature_plot(df, features, dic=False):
    """
    Takes in a dataframe, a list of features, and a boolean value
    If the boolean value is True, the function will assume that the features are in a dictionary
    If the boolean value is False, the function will assume that the features are in a list
    Returns a plot of the distribution, qq plot, and regression plot for each feature
    """
    if dic == True:
        features = features.keys()
    else:
        for col in df[features]:

            fig, axs = plt.subplots(1, 3, figsize=(15,7))

            # plot distribution
            sns.histplot(df[col], kde=True, ax=axs[0])
            axs[0].set_title(f"{col} Distribution")
            axs[0].set_xlabel(col)

            # qq plot
            sm.qqplot(df[col], ax=axs[1],line = 'q')
            axs[1].set_title(f"{col} QQ Plot")

            # regplot with red line
            sns.regplot(df, x = col, y = 'saleprice', line_kws={'color': 'red'})
            axs[2].set_title(f"{col} vs Sale Price")
            plt.tight_layout()

def transformation_plot(df, features, func):
    """
    Takes in a dataframe, a list of features, and a function to apply to the features
    Returns a plot of the distribution of the features before and after the transformation
    """
    for col in df[features]:
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        # hist 1
        sns.histplot(df[col], kde=True, ax=axs[0])
        axs[0].set_title(f"{col} Before Transformation")
        axs[0].set_xlabel(col)
        # hist 2
        sns.histplot(df[col].map(func),kde=True, ax = axs[1])
        axs[1].set_title(f'{col} After Transformation')
        # qqplot
        sm.qqplot(df[col].map(func), ax=axs[2], dist = stats.norm, line = 'q')
        axs[2].set_title(f'{col} QQ plot After Transformation')

def return_plot(preds, y_train, resids):
    """
    Takes in a list of predictions, a list of actual values, and a list of residuals
    Returns a plot of the predicted vs actual values, a plot of the residuals vs fits, 
    and a plot of the residuals vs predicted values
    """
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].scatter(y_train, preds, edgecolors=(0, 0, 0))
    axs[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'r')

    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[0].set_title('Predicted vs. Actual')

    sns.residplot(x = preds,y = resids,lowess = True,line_kws = {'color' : 'red'}, ax = axs[1]).set(title = 'Residuals vs. Fits plot',
        xlabel = 'Predicted value',
        ylabel = 'Residual')

    sm.qqplot(resids,dist = stats.norm,line = 'q', ax = axs[2])
    axs[2].set_title('Normal Q-Q plot');

    fig.tight_layout()

def plot_important_features(model, X, n_features):
    """
    Takes in a model, a dataframe, and an integer
    Returns a plot of the most important features in the model
    """
    
    # plot most important features
    coef = pd.Series(model.coef_, index = X.columns)
    important_coef = pd.concat([coef.sort_values().head(n_features),
                            coef.sort_values().tail(n_features)])   
    important_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    # label the axes
    plt.xlabel("Coefficients")
    plt.ylabel("Features")


def transformation_plot_2(df, features):
    """
    Plots a histogram of a log transformed variable, 
    a square root transformed variable, 
    and inverse hyperbolic sine transformed variable
    """
    for col in df[features]:
        fig, axs = plt.subplots(1, 4, figsize=(15,5))

        # hist 1
        sns.histplot(df[col],kde=True, ax = axs[0], color='red')
        axs[0].set_title(f"{col} Before Transformation")
        axs[0].set_xlabel(col)

        # hist 2
        sns.histplot((df[col]+1).map(np.log), kde=True, ax=axs[1], color='green')
        axs[1].set_title(f'{col} After Log Transformation')
        
        # hist 3
        sns.histplot(df[col].map(np.sqrt),kde=True, ax = axs[2], color='orange')
        axs[2].set_title(f'{col} After Sqrt Transformation')
        # hist 4
        sns.histplot(df[col].map(np.arcsinh),kde=True, ax = axs[3], color='blue')
        axs[3].set_title(f'{col} After arcsinh Transformation')
        plt.tight_layout()

standardize = False
log_flag = False 

def get_scores(model, X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                                preds, standardize=standardize, log_flag=log_flag):
    """
    Takes in a model, a dataframe, a list of features, a list of target values, and a list of predictions
    Returns the training and testing scores of the model
    """
    
    if standardize == True:
        X_train_ = Xs_train
        X_val_ = Xs_val
    else:
        X_train_ = X_train
        X_val_ = X_val

    if log_flag == True:
        var_preds = np.exp(preds)
        var_y_train = np.exp(y_train)
    else:
        var_preds = preds
        var_y_train = y_train

    scores_1 = (cross_val_score(model, X_train_, y_train, cv = KFold(n_splits=5,
                                                                 shuffle=True,
                                                                 random_state=42)))
    print('Training R^2 Score: ', model.score(X_train_, y_train))
    print('Training RMSE: ', mean_squared_error(var_preds, var_y_train, squared = False))
    print('Cross Validation R^2 Score: ', scores_1.mean())
    print('Validation R^2 Score', model.score(X_val_, y_val))




def run_regression(X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                                 standardize=standardize, log_flag=log_flag):
    """
    Takes in a dataframe, a list of features, a list of target values, and a list of predictions
    Returns the training and testing scores of 4 models

    """

    if standardize == True:
        X_train_ = Xs_train
    else:
        X_train_ = X_train

    # OLS
    ols = LinearRegression()
    ols.fit(X_train_, y_train)
    preds = ols.predict(X_train_)
    resid = y_train - preds
    print('OLS')
    print('')
    get_scores(ols, X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                preds, standardize=standardize, log_flag=log_flag)
    print('')
    return_plot(preds,y_train,resid)

    # Lassos

    mlr_lasso_ = LassoCV(alphas = np.arange(0.001, 10, 1))
    mlr_lasso_.fit(X_train_, y_train)

    mlr_lasso = Lasso(alpha = mlr_lasso_.alpha_)
    mlr_lasso.fit(X_train_, y_train)
    preds = mlr_lasso.predict(X_train_)
    resid = y_train - preds
    print('Lasso')
    print('')
    get_scores(mlr_lasso, X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                preds, standardize=standardize, log_flag=log_flag)
    print('')
    return_plot(preds,y_train,resid)


    # Ridge
    mlr_ridge = Ridge(alpha=1)
    mlr_ridge.fit(X_train_, y_train)
    preds = mlr_ridge.predict(X_train_)
    resid = y_train - preds
    print('Ridge')
    print('')
    get_scores(mlr_ridge, X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                preds, standardize=standardize, log_flag=log_flag)
    print('')
    return_plot(preds,y_train,resid)

    # Elastic Net
    mlr_elastic = ElasticNetCV(alphas = np.arange(0.001, 10, 1))
    mlr_elastic.fit(X_train_, y_train)
    preds = mlr_elastic.predict(X_train_)
    resid = y_train - preds
    print('Elastic Net')
    print('')
    get_scores(mlr_elastic, X_train, X_val, Xs_train, Xs_val, y_train, y_val,
                preds, standardize=standardize, log_flag=log_flag)
    print('')
    return_plot(preds,y_train,resid)
        
        
    