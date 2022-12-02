import re
import numpy as np
import pandas as pd


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

# An empty dictionary I'll apend to as I explore the data

CORRELATIVE_FACTORS = {}

def unique_by_col(category, df):
    items = {}
    for col in df[category].columns:
        items.update({col:df[category][col].unique()}) 
    return items

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return abs(modified_z_score) > thresh

