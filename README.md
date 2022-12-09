# Predicting House Prices in Ames, IA Via Multiple Linear Regression 
---

### Problem Statement

Our client, Wolliz, a burgeoning tech real-estate startup, desires to break into the real-estate marketplace. In an effort to differentiate their product, they hope to provide high-precision home price prediction to their users from a minimum number of data points. Using the myriad features available in the Ames, IA dataset as a laboratory, we look to identify correlative factors to home sales price and build a robust Hedonic price regression model that will generalize to unseen real estate data. 

### Summary

We began our exploration of the data by subdividing the 80 + features into four primary categories, `ORDINAL`, `NOMINAL`, `CONTINUOUS` and `DISCRETE`. The dataset required a large amount of cleaning and processing, and it was necessary to  impute missing values for several of the features. 

Within these categories, we conducted a correlation analysis, looking for features within the Ames, IA dataset that had a strong Pearson correlation value relative to home sale price. Pearson correlation, R, is a measure of the linear correlation between two variables, and ranges from -1 to +1. A value of 1 indicates a perfect positive linear correlation, a value of -1 indicates a perfect negative linear correlation and a value of 0 indicates no linear correlation. Since the `NOMINAL` and `ORDINAL` feature groups did not contain numeric values, we imputed numeric values based on the inherent ranking in the case of the ordinal values, or created Boolean *is* or *is not* features from the nominal values. We then filtered the list for values that had an absolute Pearson value greater than .1 and conducted thorough EDA on the datasets.

Via distribution visualization, we identified skew in a number of the features, including the dependent variable: home sales price. In an effort to build a more robust regression model, we explored log, square root and  inverse hyperbolic sine transformations on the skewed features. We found that a log transformation of gross living area, lot area, and lot frontage, as well as square root transformation of home age, garage age and total basement square footage allowed the model to perform better on the training data and generalize better to unseen data. 

The final, improved and transformed dataset was run on 4 different linear regression models, Lasso, Ridge, Elastic Net and Ordinary Least Squares. All of the models performed similarly, with Lasso performing slightly better on Cross Validation data. In an effort to account for non-linear behavior in the gross living area and overall quality features, a polynomial transformation was performed, again raising the predictive power of the linear regression models. 

The final production linear regression model was able to achieve training, validation and cross validation R^2 scores above .90 with Root Mean Squared Errors of approx 20,000 on training data and $20,700 on unseen data, indicating the model generalizes quite well.

### VI. Conclusions and Recommendations

The final production model achieved powerful predictive ability and the final deliverable was to the client's specifications.

The Ames dataset has a dizzying array of features, the final production model uses 68 total features, 21 of which are engineered. We found that choosing to proceed with the features most correlated to `saleprice` and exploring various transformations of the data gave our model robust predictive power, however this came with much multicollinearity and at the expense of inference.


1. Gross square footage, overall home quality, the age of the home and the neighborhood are the most important predictors to use in a hedonic price regression.

2. Choosing hedonic factors based on correlation to `saleprice` leads to a strong home price predictive ability, but the model will have multicollinearity isues and lose inferential power.

3. Some of the most important features, `lot_area`, `lot_frontage`, `gr_liv_area`, as well as the target, `saleprice`, will likely require a log transformation in order for the linear model to have robust predictive power.

4. `house_age` will likely require a square root transformation in order for the linear model to have robust predictive power.

5. Non-linearity within features must be corrected through polynomial regression. The model gained enhanced predictive power by squaring `gr_liv_area` and `overall_qual_cond`.

