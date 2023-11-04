# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:42:09 2023

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

## Updating the Directory and just running the Entire Script should work 
os.chdir("Users/marc/Downloads/")



# Load the dataset
df = pd.read_excel('dataton_2023_version2.xlsx')
#df.info()
#rows_with_nan_df = df[df.isna().any(axis=1)]
# Obtain rows with NaN values in the specified columns
rows_with_nan_df = df[df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].isna().all(axis=1)]


# Get the row indexes from rows_with_nan_df
indexes_to_drop = rows_with_nan_df.index

# Drop the rows with missing values from the original DataFrame df
df = df.drop(indexes_to_drop)

# # Group by 'age_range', 'Generation', and 'Market', and fill missing values with mode
# columns_to_impute = ['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']

# for col in columns_to_impute:
#     df[col] = df.groupby(['age_range', 'Generation', 'Market'])[col].transform(lambda x: x.fillna(x.mode().iloc[0]))


# Assuming df is your dataframe
df['new_customer'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).ge(202301).astype(int)
df['new_customer'] = df['new_customer'].astype('category')



# Extract unique prefixes
prefixes = set(col.rsplit('_', 1)[0] for col in df.columns if col.endswith(tuple(['_' + str(i) for i in range(202303, 202308)])))

# For each unique prefix, compute median and create new column
for prefix in prefixes:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + '_median'] = df[cols_of_interest].median(axis=1)
    df[prefix + '_std_deviation'] = df[cols_of_interest].std(axis=1)
#drop wealth product median and online enrolled median    
    
# Drop the specified columns
columns_to_drop = ['Online_Enrolled_median', 'Online_Enrolled_std_deviation', 'wealth_product_median', 'wealth_product_std_deviation']
df = df.drop(columns=columns_to_drop, errors='ignore')  # ensures no error is raised if a column doesn't exist



# Extract unique prefixes based on given examples
prefixes1 = set(['wealth_product_', 'Online_Enrolled_'])

# For each unique prefix, compute mode and create new column
for prefix in prefixes1:
    cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    df[prefix + 'mode'] = df[cols_of_interest].mode(axis=1)[0]   
    


# Impute NaN values in specified columns with 'N'
columns_to_impute = ['Online_Enrolled_mode', 'wealth_product_mode']
for column in columns_to_impute:
    df[column] = df[column].fillna('N')




# # For each unique prefix, compute month-over-month percentage difference and create a new column for median percentage difference
# for prefix in prefixes:
#     cols_of_interest = [col for col in df.columns if col.startswith(prefix)]
    
#     # Sort the columns to ensure they are in chronological order
#     cols_of_interest.sort()
    
#     # Select only the numeric columns for percentage calculations
#     numeric_cols = df[cols_of_interest].select_dtypes(include=['number'])
    
#     # Calculate month-over-month percentage difference for the numeric columns
#     percentage_difference_df = numeric_cols.pct_change(axis=1) * 100  # Multiply by 100 to get percentages
    
#     # Calculate the median percentage difference for each row
#     median_percentage_difference = percentage_difference_df.median(axis=1)
    
#     # Create a new column for the median percentage difference
#     df[prefix + '_median_pct_difference'] = median_percentage_difference



# import pandas as pd


# # Extract unique prefixes for the 'OnlineWallet_Dollars' prefix
# prefixes = set(col.rsplit('_', 1)[0] for col in df.columns if col.startswith('OnlineWallet_Dollars'))

# # For each unique prefix, compute month-over-month percentage difference and create a new column for median percentage difference
# for prefix in prefixes:
#     # Select only columns that end with months
#     cols_of_interest = [col for col in df.columns if col.startswith(prefix) and col.endswith(tuple(['_' + str(i) for i in range(202303, 202309)]))]
    
#     # Sort the columns to ensure they are in chronological order
#     cols_of_interest.sort()
    
#     # Select only the numeric columns for percentage calculations
#     numeric_cols = df[cols_of_interest].select_dtypes(include=['number'])
    
#     # Calculate month-over-month percentage difference for the numeric columns
#     percentage_difference_df = numeric_cols.pct_change(axis=1) * 100  # Multiply by 100 to get percentages
    
#     # Calculate the median percentage difference for each row
#     median_percentage_difference = percentage_difference_df.median(axis=1)
    
#     # Create a new column for the median percentage difference
#     df[prefix + '_median_pct_difference'] = median_percentage_difference

#     # Print cols_of_interest to verify that only month-ending columns are included
#     print(cols_of_interest)




# # Extract unique prefixes
# prefixes = set(col.rsplit('_', 1)[0] for col in df.columns if col.endswith(tuple(['_' + str(i) for i in range(202303, 202308)])))








# Numerator: Sum of 'Avg_Mortgage_Bal_median', 'Avg_CreditCard_Bal_median', 'Avg_Loan_Bal_median', and 'DebitCard_Spend_median'
numerator0 = df['Avg_Mortgage_Bal_median'] + df['Avg_Loan_Bal_median'] 

# Denominator: Sum of 'Avg_Checking_Bal_median', 'Avg_Savings_Bal_median', and 'Avg_CD_Bal_median'
denominator = df['Avg_Checking_Bal_median'] + df['Avg_Savings_Bal_median'] + df['Avg_CD_Bal_median']

# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)

# New feature: average_owed_median
df['average_large_owed_median'] = numerator0 / denominator


numerator1 = df['Avg_CreditCard_Bal_median'] + df['DebitCard_Spend_median']

# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)


# New feature: average_owed_median
df['average_small_owed_median'] = numerator1 / denominator




# Numerator: Sum of 'Avg_Mortgage_Bal_median', 'Avg_CreditCard_Bal_median', 'Avg_Loan_Bal_median', and 'DebitCard_Spend_median'
numerator2 = df['Remote_Dep_Amt_median']


# Adjust the denominator: if it's 0, use 1 instead
denominator = denominator.where(denominator != 0, 1)

# New feature: average_owed_median
df['average_remote_gained_median'] = numerator2 / denominator




# Define a function to calculate the difference in months
def months_difference(date1, date2):
    year_diff = (date1 // 100) - (date2 // 100)
    month_diff = (date1 % 100) - (date2 % 100)
    return year_diff * 12 + month_diff

# Calculate age in system in terms of months
df['age_in_system_months'] = df[['First_Open_Dt_Deposit', 'First_Open_Dt_Loan', 'First_Open_Dt_Card', 'First_Open_Dt_Mortgage']].min(axis=1).apply(lambda x: months_difference(202308, x))


def categorize_account(row):
    # Define the accounts for easier referencing
    deposit_accounts = ['CD_Accts_median', 'Checking_Accts_median', 'Savings_Accts_median']
    loan_account = 'Loan_Accts_median'
    card_account = 'CreditCard_Accts_median'
    mortgage_account = 'Mortgage_Accts_median'
    
    has_deposit = any([row[acct] > 0 for acct in deposit_accounts])
    has_loan = row[loan_account] > 0
    has_card = row[card_account] > 0
    has_mortgage = row[mortgage_account] > 0

    # Define the conditions for each category
    conditions = [
        (has_deposit and not has_loan and not has_card and not has_mortgage, "deposit_only"),
        (not has_deposit and has_loan and not has_card and not has_mortgage, "loan_only"),
        (not has_deposit and not has_loan and has_card and not has_mortgage, "card_only"),
        (not has_deposit and not has_loan and not has_card and has_mortgage, "mortgage_only"),
        (has_deposit and has_loan and not has_card and not has_mortgage, "deposit_loan"),
        (has_deposit and not has_loan and has_card and not has_mortgage, "deposit_card"),
        (has_deposit and not has_loan and not has_card and has_mortgage, "deposit_mortgage"),
        (not has_deposit and has_loan and has_card and not has_mortgage, "loan_card"),
        (not has_deposit and has_loan and not has_card and has_mortgage, "loan_mortgage"),
        (not has_deposit and not has_loan and has_card and has_mortgage, "card_mortgage"),
        (has_deposit and has_loan and has_card and not has_mortgage, "deposit_loan_card"),
        (has_deposit and has_loan and not has_card and has_mortgage, "deposit_loan_mortgage"),
        (has_deposit and not has_loan and has_card and has_mortgage, "deposit_card_mortgage"),
        (not has_deposit and has_loan and has_card and has_mortgage, "loan_card_mortgage"),
        (has_deposit and has_loan and has_card and has_mortgage, "deposit_loan_card_mortgage"),
    ]
    
    # Iterate through conditions and return the category when a condition is met
    for condition, category in conditions:
        if condition:
            return category
    return "other"  # If no condition is met (though it should never reach here given your setup)

# Create the new feature
df['account_category'] = df.apply(categorize_account, axis=1)


df['account_category'].value_counts()

# Given value counts
value_counts = df['account_category'].value_counts()

# Find categories with counts less than or equal to 13
low_cardinality_categories = value_counts[value_counts <= 20].index

# Update the 'account_category' column in the DataFrame to replace low cardinality categories with 'other'
df['account_category'] = df['account_category'].replace(low_cardinality_categories, 'other')




# median of  month over month percatage change feature ... idea dropped
# # Define the range of months you want to consider
# months_range = [str(i) for i in range(202303, 202309)]  # Up to and including '202308'

# # For each unique prefix, compute month-over-month percentage difference and create a new column for median percentage difference
# for prefix in prefixes:
#     # Select only columns that end with months in the specified range
#     cols_of_interest = [col for col in df.columns if col.startswith(prefix) and col.endswith(tuple(['_' + month for month in months_range]))]
    
#     # Sort the columns to ensure they are in chronological order
#     cols_of_interest.sort()
    
#     # Select only the numeric columns for percentage calculations
#     numeric_cols = df[cols_of_interest].select_dtypes(include=['number'])
    
#     # Check if any column in the row has a non-zero value in the specified range
#     nonzero_row_mask = (numeric_cols != 0).any(axis=1)
    
#     # Add 0.01 to each column value if any column in the row has a non-zero value
#     numeric_cols.loc[nonzero_row_mask] += 0.01
    
#     # Calculate month-over-month percentage difference for the numeric columns
#     percentage_difference_df = numeric_cols.pct_change(axis=1) * 100  # Multiply by 100 to get percentages
    
#     # Calculate the median percentage difference for each row
#     median_percentage_difference = percentage_difference_df.median(axis=1)
    
#     # Create a new column for the median percentage difference
#     df[prefix + '_median_pct_difference'] = median_percentage_difference

#     # Print cols_of_interest to verify that only month-ending columns are included
#     print(cols_of_interest)



# import pandas as pd
# import plotly.express as px

cols = ['OnlineWallet_Dollars_median', 'DigLogins_TotalLogins_median', 'DigLogins_UniqDays_median', 
        'OnlineWallet_Tx_median', 'Alerts_Enrolled_median', 'Remote_Dep_Amt_median', 'Remote_Dep_Ct_median']

# Create a subset of the DataFrame with only the specified columns
subset_df = df[cols]

# Calculate the correlation matrix
correlation_matrix = subset_df.corr()

# Create a heatmap using Plotly
fig = px.imshow(correlation_matrix,
                x=subset_df.columns,
                y=subset_df.columns,
                color_continuous_scale='darkmint')  # Use 'viridis' colorscale

fig.update_layout(title='Correlation Heatmap')
fig.show()




# Initial Numeric Columns Sanity Check
columns_to_use = ['new_customer','account_category','Alerts_Enrolled_median',
'DebitCard_Tx_median',
'DebitCard_Tx_std_deviation',
'Checking_Accts_median',
'Checking_Accts_std_deviation',
'BranchTX_median',
'BranchTX_std_deviation',
'DebitCard_Spend_median',
'DebitCard_Spend_std_deviation',
'Avg_Savings_Bal_median',
'Avg_Savings_Bal_std_deviation',
'Avg_Mortgage_Bal_median',
'Avg_Mortgage_Bal_std_deviation',
'Avg_CreditCard_Bal_median',
'Avg_CreditCard_Bal_std_deviation',
'Loan_Accts_median',
'Loan_Accts_std_deviation',
'CD_Accts_median',
'CD_Accts_std_deviation',
'Avg_Loan_Bal_median',
'Avg_Loan_Bal_std_deviation',
'Avg_Checking_Bal_median',
'Avg_Checking_Bal_std_deviation',
'Savings_Accts_median',
'Savings_Accts_std_deviation',
'Avg_CD_Bal_median',
'Avg_CD_Bal_std_deviation',
'CreditCard_Accts_median',
'CreditCard_Accts_std_deviation',
'Mortgage_Accts_median',
'Mortgage_Accts_std_deviation',
'wealth_product_mode',
'average_large_owed_median',
'average_small_owed_median',
'average_remote_gained_median',
'age_in_system_months','age_range', 'Generation', 'Market']

# Filter out only numeric columns from the provided list
numeric_cols = df[columns_to_use].select_dtypes(include=['float64', 'int64']).columns.tolist()

# Create a subset of the DataFrame with only the numeric columns
subset_df = df[numeric_cols]

# Calculate the correlation matrix
correlation_matrix = subset_df.corr()

# Create a heatmap using Plotly
fig = px.imshow(correlation_matrix,
                x=subset_df.columns,
                y=subset_df.columns,
                color_continuous_scale='viridis')  # Use 'viridis' colorscale

fig.update_layout(title='Correlation Heatmap')
fig.show()



# Impute missing values in numeric columns with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing values in non-numeric columns with mode
non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])

# Define the outcome columns
outcome_columns = ['OnlineWallet_Dollars_median', 'DigLogins_TotalLogins_median', 'DigLogins_UniqDays_median', 
                   'OnlineWallet_Tx_median', 'Alerts_Enrolled_median', 'Remote_Dep_Amt_median', 'Remote_Dep_Ct_median']


#remove 'DebitCard_Tx_std_deviation', 'DebitCard_Spend_median' ,'BranchTX_std_deviation', 'Avg_Savings_Bal_std_deviation',\
# 'Avg_CreditCard_Bal_std_deviation','Avg_Mortgage_Bal_median','Mortgage_Accts_median',
#to address multicollinearity 

# Define the columns to use as features
columns_to_use = [
'new_customer',
'account_category',
'Alerts_Enrolled_median',
'DebitCard_Tx_median',
'Checking_Accts_median',
'Checking_Accts_std_deviation',
'BranchTX_median',
'DebitCard_Spend_std_deviation',
'Avg_Savings_Bal_median',
'Avg_Mortgage_Bal_std_deviation',
'Avg_CreditCard_Bal_median',
'Loan_Accts_median',
'Loan_Accts_std_deviation',
'CD_Accts_median',
'CD_Accts_std_deviation',
'Avg_Loan_Bal_median',
'Avg_Loan_Bal_std_deviation',
'Avg_Checking_Bal_median',
'Avg_Checking_Bal_std_deviation',
'Savings_Accts_median',
'Savings_Accts_std_deviation',
'Avg_CD_Bal_median',
'Avg_CD_Bal_std_deviation',
'CreditCard_Accts_median',
'CreditCard_Accts_std_deviation',
'wealth_product_mode',
'Online_Enrolled_mode',
'average_large_owed_median',
'average_small_owed_median',
'average_remote_gained_median',
'age_in_system_months','age_range', 'Generation', 'Market']



# #Final Sanity Check
# # Filter out only numeric columns from the provided list
# numeric_cols = df[columns_to_use].select_dtypes(include=['float64', 'int64']).columns.tolist()

# # Create a subset of the DataFrame with only the numeric columns
# subset_df = df[numeric_cols]

# # Calculate the correlation matrix
# correlation_matrix = subset_df.corr()

# # Create a heatmap using Plotly
# fig = px.imshow(correlation_matrix,
#                 x=subset_df.columns,
#                 y=subset_df.columns,
#                 color_continuous_scale='viridis')  # Use 'viridis' colorscale

# fig.update_layout(title='Correlation Heatmap')
# fig.show()



import catboost as cb
from sklearn.metrics import mean_squared_error, r2_score


# Define the features and target
X = df[columns_to_use]
y = df[outcome_columns[1]]  # Change this for each outcome column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



# Prepare data in the CatBoost Pool format
train_dataset = cb.Pool(X_train, y_train, cat_features=X.select_dtypes(include=['object', 'category']).columns.tolist()) 
test_dataset = cb.Pool(X_test, y_test, cat_features=X.select_dtypes(include=['object', 'category']).columns.tolist())


# best model already found tuned parameters used below 
#no need to run the tuning step
# # Define the CatBoost Regressor model
# model = cb.CatBoostRegressor(loss_function='RMSE', od_type='Iter',
#                             od_wait=30,
#                             verbose=True,task_type='CPU')

# learning_rate_samples = [sp_randFloat(0.001, 0.1).rvs() for _ in range(5)]
# depth_samples = [sp_randInt(2, 10).rvs() for _ in range(5)]



# # Define hyperparameters to be tuned
# grid = {
#     'iterations': [100, 150, 200,250],
#     'learning_rate': learning_rate_samples,
#     'depth': depth_samples,
#     'l2_leaf_reg': [0.2, 0.5, 1, 3, 5, 10]
# }

# # Execute grid search with cross-validation
# model.grid_search(grid, train_dataset,verbose=1)


# # Get the best score
# best_score = model.get_best_score()

# # Get the best hyperparameters
# best_params = model.get_params()

# print("Best Score:", best_score)
# print("Best Parameters:", best_params)


# Best Score: {'learn': {'RMSE': 7.893645419762804}}
# Best Parameters: {'loss_function': 'RMSE', 'od_wait': 30, 'od_type': 'Iter', 'verbose': True, 'task_type': 'CPU', 'depth': 9, 'iterations': 200, 'learning_rate': 0.08546924783980771, 'l2_leaf_reg': 0.5}

# # Evaluate the performance on the test set
# pred = model.predict(test_dataset)
# rmse = (np.sqrt(mean_squared_error(y_test, pred)))
# r2 = r2_score(y_test, pred)

# # Display the test performance
# print("Testing performance")
# print('RMSE: {:.2f}'.format(rmse))
# print('R2: {:.2f}'.format(r2))


# Testing performance
# RMSE: 12.29
# R2: 0.42


# Best hyperparameters from the previous grid search
best_params = {'loss_function': 'RMSE', 'od_wait': 30, 'od_type': 'Iter', 'verbose': True, 'task_type': 'CPU', 'depth': 9, 'iterations': 200, 'learning_rate': 0.08546924783980771, 'l2_leaf_reg': 0.5}

#  CatBoost Regressor model with the best parameters
best_model = cb.CatBoostRegressor(**best_params)

# Train the model using the best hyperparameters
best_model.fit(train_dataset)

# sanity check predictions/ model evaluations
pred_1 = best_model.predict(test_dataset)
rmse_1 = (np.sqrt(mean_squared_error(y_test, pred_1)))
r2_1 = r2_score(y_test, pred_1)

print("Test RMSE:", rmse_1)
print("Test R2:", r2_1)



# 'model' is the trained CatBoost model and 'X' is features DataFrame
sorted_feature_importance = best_model.feature_importances_.argsort()
plt.figure(figsize=(10, len(X.columns)))
plt.barh(X.columns[sorted_feature_importance], best_model.feature_importances_[sorted_feature_importance], color='turquoise')
plt.xlabel("CatBoost Feature Importance")
plt.show()


import shap

# 'model' trained CatBoost model
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)


# Convert shap_values into an Explanation object
expl = shap.Explanation(values=shap_values, data=X_test, feature_names=X_test.columns)

# Now plot the bar chart
shap.plots.bar(expl,max_display =15)


#beeswarm plot
shap.summary_plot(shap_values, X_test)

# Waterfall plot for the first observation in X_test
expl = shap.Explanation(values=shap_values, 
                        data=X_test, 
                        feature_names=X_test.columns, 
                        base_values=explainer.expected_value)

shap.plots.waterfall(expl[0])


######################################################## need to make function out of it ########################


feature_values = X_test['Market'].reset_index(drop=True)
feature_shap_values = expl[:, 'Market'].values

# Reset the indices for proper alignment
feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
feature_values_mask = feature_values.reset_index(drop=True)

unique_categories = feature_values.unique()
new_shap_values = [feature_values_series[feature_values_mask == category].values 
                   for category in unique_categories]

max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs, [np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values).transpose()

feature_expl = shap.Explanation(values=new_shap_values, 
                                data=np.array([[0]*len(unique_categories)]*max_len), 
                                feature_names=list(unique_categories),
                                base_values=np.array([0]*max_len))

shap.plots.beeswarm(feature_expl, color_bar=False, show=True,max_display=len(unique_categories))


########################################################################################

feature_values = X_test['Generation'].reset_index(drop=True)
feature_shap_values = expl[:, 'Generation'].values

# Reset the indices for proper alignment
feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
feature_values_mask = feature_values.reset_index(drop=True)

unique_categories = feature_values.unique()
new_shap_values = [feature_values_series[feature_values_mask == category].values 
                   for category in unique_categories]

max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs, [np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values).transpose()

feature_expl = shap.Explanation(values=new_shap_values, 
                                data=np.array([[0]*len(unique_categories)]*max_len), 
                                feature_names=list(unique_categories),
                                base_values=np.array([0]*max_len))

shap.plots.beeswarm(feature_expl, color_bar=False, show=True,max_display=len(unique_categories))

######################################################################################


feature_values = X_test['age_range'].reset_index(drop=True)
feature_shap_values = expl[:, 'age_range'].values

# Reset the indices for proper alignment
feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
feature_values_mask = feature_values.reset_index(drop=True)

unique_categories = feature_values.unique()
new_shap_values = [feature_values_series[feature_values_mask == category].values 
                   for category in unique_categories]

max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs, [np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values).transpose()

feature_expl = shap.Explanation(values=new_shap_values, 
                                data=np.array([[0]*len(unique_categories)]*max_len), 
                                feature_names=list(unique_categories),
                                base_values=np.array([0]*max_len))

shap.plots.beeswarm(feature_expl, color_bar=False, show=True,max_display=len(unique_categories))

#######################################################################################

feature_values = X_test['new_customer'].reset_index(drop=True)
feature_shap_values = expl[:, 'new_customer'].values

# Reset the indices for proper alignment
feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
feature_values_mask = feature_values.reset_index(drop=True)

unique_categories = feature_values.unique()
new_shap_values = [feature_values_series[feature_values_mask == category].values 
                   for category in unique_categories]

max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs, [np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values).transpose()

feature_expl = shap.Explanation(values=new_shap_values, 
                                data=np.array([[0]*len(unique_categories)]*max_len), 
                                feature_names=list(unique_categories),
                                base_values=np.array([0]*max_len))

shap.plots.beeswarm(feature_expl, color_bar=False, show=True,max_display=len(unique_categories))

#########################################################################################


feature_values = X_test['Online_Enrolled_mode'].reset_index(drop=True)
feature_shap_values = expl[:, 'Online_Enrolled_mode'].values

# Reset the indices for proper alignment
feature_values_series = pd.Series(feature_shap_values).reset_index(drop=True)
feature_values_mask = feature_values.reset_index(drop=True)

unique_categories = feature_values.unique()
new_shap_values = [feature_values_series[feature_values_mask == category].values 
                   for category in unique_categories]

max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs, [np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values).transpose()

feature_expl = shap.Explanation(values=new_shap_values, 
                                data=np.array([[0]*len(unique_categories)]*max_len), 
                                feature_names=list(unique_categories),
                                base_values=np.array([0]*max_len))

shap.plots.beeswarm(feature_expl, color_bar=False, show=True,max_display=len(unique_categories))











########################################################################################


# Generate the HTML for the force plot
shap_html = shap.force_plot(explainer.expected_value, shap_values[:10, :], X_test.iloc[:10, :], show=False)

# Convert the plot to HTML string format
shap_html_str = shap.getjs().replace('</head>', '<script src="https://cdn.jsdelivr.net/npm/@plotly/d3@3.5.17/d3.min.js"></script></head>') + shap_html.html()

# Save the plot
with open("shap_force_plot_logins_semifinal.html", "w") as f:
    f.write(shap_html_str)


# already computed shap_values and have an explainer object

# For a single instance
shap.decision_plot(explainer.expected_value, shap_values[150], X_test.iloc[150, :])

# For multiple instances (for example the first 10)
shap.decision_plot(explainer.expected_value, shap_values[:10, :], X_test.iloc[:10, :])



# import plotly.graph_objects as go
# import numpy as np

# def shap_plotly_decision_plot(expected_value, shap_values, features):
#     # Calculate the cumulative SHAP values
#     shap_cum_values = np.cumsum(shap_values, axis=1)
    
#     # Determine the base value (starting point for each decision path)
#     base_value = np.repeat(expected_value, shap_values.shape[0])
    
#     # Initialize the Plotly figure
#     fig = go.Figure()

#     # Add a line for each sample
#     for i in range(shap_values.shape[0]):
#         fig.add_trace(go.Scatter(x=np.arange(shap_values.shape[1]),
#                                  y=shap_cum_values[i, :],
#                                  mode='lines',
#                                  name=f'Sample {i}'))

#     # Set the axis labels
#     fig.update_layout(title='SHAP Decision Plot',
#                       xaxis_title='Feature Index',
#                       yaxis_title='SHAP Value',
#                       hovermode='x unified')
    
#     return fig

# # Assuming you've already computed shap_values and have an explainer object
# # Visualize the first 10 samples
# fig = shap_plotly_decision_plot(explainer.expected_value, shap_values[:10, :], X_test.iloc[:10, :])
# fig.show()





############### benchmarking ##############################################################

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# X_train and y_train are in DataFrame and Series formats respectively
train_data = X_train.copy()
train_data['DigLogins_TotalLogins_median'] = y_train

# Compute means on training data
means = train_data.groupby(['Generation', 'Market'])['DigLogins_TotalLogins_median'].mean().reset_index()

# Merge the means with the test data to get the predictions
X_test_with_pred = pd.merge(X_test, means, on=['Generation', 'Market'], how='left')

# Handle any cases where there isn't a match 
#(i.e., combination of Generation and Market from test set wasn't in train set)
# For this, we could use the global mean, but other strategies can be used as well
X_test_with_pred['DigLogins_TotalLogins_median'].fillna(y_train.mean(), inplace=True)

# Compute RMSE on test set
rmse = np.sqrt(mean_squared_error(y_test, X_test_with_pred['DigLogins_TotalLogins_median']))


#based on grouped means
print(rmse)
#15.708934251656771

# Calculate the mean of y_train
mean_y_train = y_train.mean()

# Predict all values in the test set with the mean of y_train
y_pred = [mean_y_train] * len(y_test)

# Compute RMSE on test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#baseline mean
print(rmse)
#16.13846415876205


###############################################################

# import matplotlib.pyplot as plt
# from umap import UMAP

# # # Compute 2D embedding of raw variable values
# # X_2d = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(X_test)

# # Assuming you are interested in the first output of your model for the SHAP values
# # SHAP values will be of the shape (num_samples, num_features, num_outputs)
# # The index `[:, :, 0]` extracts values for the first output
# s_2d = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(shap_values[:, :, 0])

# import matplotlib.pyplot as plt
# from umap import UMAP

# # Compute 2D embedding of SHAP values
# s_2d = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(shap_values)

# # Plot the embeddings
# plt.figure(figsize=(10, 8))
# plt.scatter(s_2d[:, 0], s_2d[:, 1], cmap='viridis', s=10)
# plt.colorbar()
# plt.title('2D UMAP Embeddings of SHAP Values')
# plt.xlabel('UMAP 1st Dimension')
# plt.ylabel('UMAP 2nd Dimension')
# plt.show()

########################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Compute 2D embedding of SHAP values
embedding = UMAP(n_components=2, n_neighbors=200, min_dist=0).fit_transform(shap_values)

# Get the quartile indicators for the 'DigLogins_TotalLogins_median' values in y_test
quartiles = np.percentile(y_test, [25, 50, 75])
colors = np.digitize(y_test, quartiles)

# Plot the embedding colored by the quartile of 'DigLogins_TotalLogins_median'
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis')
legend1 = plt.legend(*scatter.legend_elements(), title="Quartiles")
plt.gca().add_artist(legend1)
plt.title('UMAP embedding of SHAP values colored by quartile of DigLogins_TotalLogins_median')
plt.show()






