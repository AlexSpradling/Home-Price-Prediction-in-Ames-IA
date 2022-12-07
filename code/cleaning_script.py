import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats


# the provided .txt file breaks the categories up into NOMINAL, ORDINAL, CONTINUOUS, and DISCRETE variables. I'll do that here
# so I can reference them thoughout multiple notebooks

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

# collect the indixes of outlier values in a column based on z-score

def outlier_index(df, col, thresh):
    z = np.abs(stats.zscore(df[col].astype(float)))
    return col, np.where(z > thresh)

# given an np array of indices, return the values at those indices
def get_outliers(df, col, indices):
    return df[col].iloc[indices]

def unique_by_col(category, df):
    items = {}
    for col in df[category].columns:
        items.update({col:df[category][col].unique()}) 
    return items

def get_regression(df, regression_type, features):
    X = df[features]
    y = df['saleprice']

def feature_plot(df, features, dic=False):
    if dic == True:
        features = features.keys()
    else:
        for col in df[features]:
            fig, axs = plt.subplots(1, 3, figsize=(15,5))
            # plot distribution
            sns.histplot(df[col], kde=True, ax=axs[0])
            axs[0].set_title(f"{col} Distribution")
            axs[0].set_xlabel(col)
            sm.qqplot(df[col], ax=axs[1],line = 'q')
            sns.regplot(df, x = col, y = 'saleprice')

def transformation_plot(df, features, func):
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
    fix, axs = plt.subplots(1,3, figsize=(15,5))
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

def plot_important_features(model, X, n_features=10):
    
    df = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})
    df = df.sort_values('importance', ascending=False)
    df = df.iloc[:n_features]
    plt.figure(figsize=(10, 6))
    plt.bar(df.feature, df.importance)
    plt.xticks(rotation=45)
    plt.title('Most Important Features in Model')
    