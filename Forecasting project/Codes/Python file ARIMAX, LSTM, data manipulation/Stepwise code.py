# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:54:40 2025

@author: Bhargavi
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import invgamma
from tabulate import tabulate
import pandas as pd


#C:\Users\Bhargavi\Documents\VLK Case study\Stepwise Working Folder\1. Raw Data

#step 1 raw data



quarterly_data=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/quarterly data copy.xlsx")
#quarterly_data.info()

consumption=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/Eurozone Private final consumption.xlsx")
gdp=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/Eurozone GDP.xlsx")
unemployment=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/Eurozone unemployment.xlsx")

#gdp['TIME']=gdp['TIME PERIOD'].str.replace(r'(\d{4})(Q\d)', r'\1-\2', regex=True)
#consumption['TIME']=consumption['TIME PERIOD'].str.replace(r'(\d{4})(Q\d)', r'\1-\2', regex=True)
#quarterly_data=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/quarterly data copy.xlsx")

#US_data_set1=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/VariableSet1Q.xlsx")
monthly_data = pd.read_excel(r"C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/Monthly Data.xlsx")

#Daily_Data
daily_data=pd.read_excel(r"C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/1. Raw Data/Daily_Data.xlsx")



#%% 
#### step 2 data with aligned timeline

#quarterly data
quarterly_data['TIME'] = pd.PeriodIndex(quarterly_data['TIME'], freq='Q')


start_period = pd.Period("2000Q1", freq='Q')
end_period = pd.Period("2024Q3", freq='Q')

# Filter the dataframe within the specified quarterly range
quarterly_data_filtered = quarterly_data[(quarterly_data['TIME'] >= start_period) & (quarterly_data['TIME'] <= end_period)]

quarterly_data_Q4=quarterly_data[(quarterly_data['TIME'] > end_period)]

#monthly data
monthly_data['TIME'] = pd.to_datetime(monthly_data['INDIC (Labels)'], format='%Y-%m')
start_period = "2000-01"
end_period = "2024-11"

# Convert to datetime for filtering
start_date = pd.to_datetime(start_period)
end_date = pd.to_datetime(end_period)

monthly_data_filtered = monthly_data[(monthly_data['TIME'] >= start_date) & (monthly_data['TIME'] <= end_date)]
monthly_data_Q4=monthly_data[(monthly_data['TIME'] > end_date)]

monthly_data_filtered['TIME'] = monthly_data_filtered['TIME'].dt.strftime('%Y-%m')

del monthly_data_filtered['INDIC (Labels)']

#Target Variables

#gdp
gdp.rename(columns={'TIME PERIOD':'TIME'},inplace=True)
gdp['TIME'] = pd.PeriodIndex(gdp['TIME'], freq='Q')
gdp_filtered = gdp[(gdp['TIME'] >= start_period) & (gdp['TIME'] <= end_period)]
gdp_unused_Q4=gdp[(gdp['TIME'] < start_period)]


#consumption
consumption.rename(columns={'TIME PERIOD':'TIME'},inplace=True)
consumption['TIME'] = pd.PeriodIndex(consumption['TIME'], freq='Q')
consumption_filtered = consumption[(consumption['TIME'] >= start_period) & (consumption['TIME'] <= end_period)]
consumption_unused_Q4=consumption[(consumption['TIME'] < start_period)]

##For unemployment we have time period from 2000-01 to 2024-11 so need to alter the data
#C:\Users\Bhargavi\Documents\VLK Case study\Stepwise Working Folder\2. Data with Aligned timeline
monthly_data_filtered.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/2. Data with Aligned timeline/Monthly aligned.xlsx")
quarterly_data_filtered.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/2. Data with Aligned timeline/Quarterly aligned.xlsx")
gdp_filtered.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/2. Data with Aligned timeline/gdp aligned.xlsx")
consumption_filtered.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/2. Data with Aligned timeline/consumption aligned.xlsx")
unemployment.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/2. Data with Aligned timeline/unemployment.xlsx")



#%% 
######step 3 Extrapolation



import pmdarima as pm

####quarterly data
columns_to_impute = [
    'Factors limiting the production - financial constraints', 'House price index',
           'Labour cost index - total labour cost',
           'Labour cost index - wages and salaries',
           'Labour cost index - labour costs other than wages and salaries',
]

for col in columns_to_impute:
    print(f"Processing column: {col}")

    # Step 1: Extract relevant data
    temp_df = quarterly_data_filtered[['TIME', col]]
    temp_df_reversed = temp_df[::-1]
    temp_df_reversed.dropna(inplace=True)
    auto_arima_model = pm.auto_arima(temp_df_reversed[col], seasonal=False, stepwise=True, trace=False)
    num_missing = int(temp_df[col].isna().sum())
    
    predictions = auto_arima_model.predict(n_periods=num_missing).reset_index()
    del predictions['index']
    predictions=predictions[::-1].reset_index()
    del predictions['index']
    predictions[0]=round(predictions[0],1)

    # Fill missing values
    nan_indices = temp_df.loc[temp_df[col].isna()].index
    temp_df.loc[nan_indices, col] = predictions.values.flatten()

    # Merge back to original dataframe
    quarterly_data_filtered.drop(columns=[col], inplace=True)
    quarterly_data_filtered = pd.merge(quarterly_data_filtered, temp_df, on='TIME', how='left')




#%%
#monthly data extrapolation


columns_to_impute = [
    'HICP - Total goods', 'HICP - Industrial goods',
    'HICP - Total services', 'Balance for values/ratio for indices',
    'Imports', 'Exports',
]

for col in columns_to_impute:
    print(f"Processing column: {col}")

    # Step 1: Extract relevant data
    temp_df = monthly_data_filtered[['TIME', col]]
    temp_df_reversed = temp_df[::-1]
    temp_df_reversed.dropna(inplace=True)
    auto_arima_model = pm.auto_arima(temp_df_reversed[col], seasonal=False, stepwise=True, trace=True)
    num_missing = int(temp_df[col].isna().sum())
    
    predictions = auto_arima_model.predict(n_periods=num_missing).reset_index()
    del predictions['index']
    predictions=predictions[::-1].reset_index()
    del predictions['index']
    predictions[0]=round(predictions[0],1)

    # Fill missing values
    nan_indices = temp_df.loc[temp_df[col].isna()].index
    temp_df.loc[nan_indices, col] = predictions.values.flatten()

    # Merge back to original dataframe
    monthly_data_filtered.drop(columns=[col], inplace=True)
    monthly_data_filtered = pd.merge(monthly_data_filtered, temp_df, on='TIME', how='left')

#%%

columns_to_impute = ['yeild Curve'
]

for col in columns_to_impute:
    print(f"Processing column: {col}")

    # Step 1: Extract relevant data
    temp_df = daily_data[['TIME', col]]
    temp_df_reversed = temp_df[::-1]
    temp_df_reversed.dropna(inplace=True)
    auto_arima_model = pm.auto_arima(temp_df_reversed[col], seasonal=False, stepwise=True, trace=True)
    num_missing = int(temp_df[col].isna().sum())
    
    predictions = auto_arima_model.predict(n_periods=num_missing).reset_index()
    del predictions['index']
    predictions=predictions[::-1].reset_index()
    del predictions['index']
#    predictions[0]=round(predictions[0],1)

    # Fill missing values
    nan_indices = temp_df.loc[temp_df[col].isna()].index
    temp_df.loc[nan_indices, col] = predictions.values.flatten()

    # Merge back to original dataframe
    daily_data.drop(columns=[col], inplace=True)
    daily_data = pd.merge(daily_data, temp_df, on='TIME', how='left')





daily_data.to_excel('C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/3. Extrapolated Data/Daily extrapolated data.xlsx')

monthly_data_filtered.to_excel('C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/3. Extrapolated Data/monthly extrapolated data.xlsx')
quarterly_data_filtered.to_excel('C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/3. Extrapolated Data/Quarterly extrapolated data.xlsx')








#%%

### Step 4 Stationarity check 

from statsmodels.tsa.stattools import adfuller

def adf_test(series, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Returns:
    - ADF Statistic
    - p-value
    - Suggested Differencing Order (d)
    """
    result = adfuller(series.dropna())
    p_value = result[1]

    # If p-value > alpha, we need differencing
    d = 0
    while p_value > alpha and d < 3:  # Limit differencing to max 3
        d += 1
        diff_series = series.diff(d).dropna()
        p_value = adfuller(diff_series)[1]

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary?": "Yes" if result[1] < alpha else "No",
        "Suggested Differencing (d)": d
    }

exog_stationarity_monthly = {col: adf_test(monthly_data_filtered[col]) for col in ['Construction confidence indicator', 'Economic sentiment indicator',
       'Industrial confidence indicator', 'Retail confidence indicator',
       'Consumer confidence indicator', 'Services confidence indicator',
       'Climate indicator',
       'Employment expectation index in next three months',
       'HICP - All items (HICP=harmonized index of consumer prices)',
       'HICP - Food and non alcoholic beverages',
       'HICP - Alcoholic beverages and tobacco',
       'HICP - Clothing and footwear',
       'HICP - Housing, water, electricity,gas and other fuels',
       'HICP - Health', 'HICP - Transport', 'HICP - Education',
       'HICP - Hotels, cafes and restaurants', 'HICP - Energy',
       'Nominal effective exchange rate - 37 trading partners',
       'Nominal effective exchange rate - 42 trading partners',
       'Real effective exchange rate (deflator: consumer price indices - 37 trading partners)',
       'Real effective exchange rate (deflator: consumer price indices - 42 trading partners)',
       'Money market interest rate', 'HICP - Total goods',
       'HICP - Industrial goods', 'HICP - Total services',
       'Balance for values/ratio for indices', 'Imports', 'Exports']}

# Convert results to DataFrame for better visualization
monthly_stationarity_results = pd.DataFrame.from_dict({ **exog_stationarity_monthly}, orient='index')

#%%
#### Step 5 mean aggregation

#monthly_data_filtered



monthly_data_filtered['Date'] = pd.to_datetime(monthly_data_filtered['TIME'], format='%Y-%m')  # Ensure correct format
monthly_data_filtered.set_index('Date', inplace=True)

# Convert index to Quarterly Period
monthly_data_filtered['Quarter'] = monthly_data_filtered.index.to_period('Q')

monthly_data_filtered_1=monthly_data_filtered[['Quarter','Construction confidence indicator',
       'Economic sentiment indicator', 'Industrial confidence indicator',
       'Retail confidence indicator', 'Consumer confidence indicator',
       'Services confidence indicator', 'Climate indicator',
       'Employment expectation index in next three months',
       'HICP - All items (HICP=harmonized index of consumer prices)',
       'HICP - Food and non alcoholic beverages',
       'HICP - Alcoholic beverages and tobacco',
       'HICP - Clothing and footwear',
       'HICP - Housing, water, electricity,gas and other fuels',
       'HICP - Health', 'HICP - Transport', 'HICP - Education',
       'HICP - Hotels, cafes and restaurants', 'HICP - Energy',
       'HICP - Total goods', 'HICP - Industrial goods',
       'HICP - Total services', 'Balance for values/ratio for indices',
       'Imports', 'Exports',
       'Nominal effective exchange rate - 37 trading partners',
       'Nominal effective exchange rate - 42 trading partners',
       'Real effective exchange rate (deflator: consumer price indices - 37 trading partners)',
       'Real effective exchange rate (deflator: consumer price indices - 42 trading partners)',
       'Money market interest rate']]

# Group by Quarter and calculate the mean
quarterly_aggregated = monthly_data_filtered_1.groupby('Quarter').mean().reset_index()

#quarterly_aggregated.to_excel(r"C:/Users/Bhargavi/Documents/VLK Case study/Euro new data/mean aggregated quarterly data.xlsx")
start_period = pd.Period("2000Q1", freq='Q')
end_period = pd.Period("2024Q3", freq='Q')

# Filter the dataframe within the specified quarterly range
quarterly_aggregated_filtered = quarterly_aggregated[(quarterly_aggregated['Quarter'] >= start_period) & (quarterly_aggregated['Quarter'] <= end_period)]
quarterly_aggregated_filtered.rename(columns={'Quarter':'TIME'},inplace=True)

full_quarterly_frame=pd.merge(quarterly_data_filtered,quarterly_aggregated_filtered, on='TIME',how='left')



#%%
#mean aggregation for daily data

del daily_data['Index']

daily_data["TIME"] = pd.to_datetime(daily_data["TIME"])

# Set TIME column as index
daily_data.set_index("TIME", inplace=True)

# Resample daily data to monthly frequency and compute mean
daily_to_monthly_df = daily_data.resample("M").mean()

# Reset index to bring TIME back as a column
daily_to_monthly_df.reset_index(inplace=True)



daily_to_monthly_df["TIME"] = daily_to_monthly_df["TIME"] - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)

# Set TIME as index
daily_to_monthly_df.set_index("TIME", inplace=True)

monthly_merged_data = daily_to_monthly_df.merge(monthly_data_filtered, left_index=True, right_index=True, how="outer")





#%% 

###Elastic Net
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant

X_train=full_quarterly_frame[['Current account',
       'Current plus capital account (balance = net lending (+) / net borrowing (-))',
       'Goods and services', 'Goods', 'Services', 'Primary income',
       'Secondary income', 'Capital account',
       'Purchase or build a home within the next 12 months',
       'Home improvements over the next 12 months',
       'Assessment of current production capacity',
       'New orders in recent months',
       'Current level of capacity utilization (%)',
       'Competitive position on foreign markets inside the EU over the past three months',
       'Competitive position on foreign markets outside the EU over the past three months',
       'Factors limiting the production - none',
       'Factors limiting the production - insufficient demand',
       'Factors limiting the production - labour',
       'Factors limiting the production - equipment',
       'Factors limiting the production - other', 'Value added, gross',
       'Compensation of employees', 'Wages and salaries',
       "Employers' social contributions", 'Industry turnover index',
       'Currency and deposits', '"Transferable deposits; other deposits"',
       'Debt securities', 'Short-term debt securities',
       'Long-term debt securities', 'Loans', 'Short-term - loans',
       'Long-term - loans', 'Government consolidated gross debt', 
       'Factors limiting the production - financial constraints',
       'House price index', 'Labour cost index - total labour cost',
       'Labour cost index - wages and salaries',
       'Labour cost index - labour costs other than wages and salaries',
       'Construction confidence indicator', 'Economic sentiment indicator',
       'Industrial confidence indicator', 'Retail confidence indicator',
       'Consumer confidence indicator', 'Services confidence indicator',
       'Climate indicator',
       'Employment expectation index in next three months',
       'HICP - All items (HICP=harmonized index of consumer prices)',
       'HICP - Food and non alcoholic beverages',
       'HICP - Alcoholic beverages and tobacco',
       'HICP - Clothing and footwear',
       'HICP - Housing, water, electricity,gas and other fuels',
       'HICP - Health', 'HICP - Transport', 'HICP - Education',
       'HICP - Hotels, cafes and restaurants', 'HICP - Energy',
       'HICP - Total goods', 'HICP - Industrial goods',
       'HICP - Total services', 'Balance for values/ratio for indices',
       'Imports', 'Exports',
       'Nominal effective exchange rate - 37 trading partners',
       'Nominal effective exchange rate - 42 trading partners',
       'Real effective exchange rate (deflator: consumer price indices - 37 trading partners)',
       'Real effective exchange rate (deflator: consumer price indices - 42 trading partners)',
       'Money market interest rate']]

y_train=consumption_filtered[['consumption change']]
y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_gdp=important_features.index.tolist()
important_feature_names_consumption = important_features.index.tolist()

#%%
monthly_merged_data = monthly_merged_data.iloc[:-1] 
monthly_merged_data = monthly_merged_data.iloc[1:]
unemployment=unemployment.iloc[1:]  

X_train=monthly_merged_data[['Marginal Lending Facility Rate', 'Main Refinancing Operations Rate',
       'ECB Deposit Facility Rate', 'Euro Stoxx 50 closing price in Euros',
       'Europe crude oil price', 'Euro Exchange rate', 'yeild Curve',
       'Construction confidence indicator', 'Economic sentiment indicator',
       'Industrial confidence indicator', 'Retail confidence indicator',
       'Consumer confidence indicator', 'Services confidence indicator',
       'Climate indicator',
       'Employment expectation index in next three months',
       'HICP - All items (HICP=harmonized index of consumer prices)',
       'HICP - Food and non alcoholic beverages',
       'HICP - Alcoholic beverages and tobacco',
       'HICP - Clothing and footwear',
       'HICP - Housing, water, electricity,gas and other fuels',
       'HICP - Health', 'HICP - Transport', 'HICP - Education',
       'HICP - Hotels, cafes and restaurants', 'HICP - Energy',
       'Nominal effective exchange rate - 37 trading partners',
       'Nominal effective exchange rate - 42 trading partners',
       'Real effective exchange rate (deflator: consumer price indices - 37 trading partners)',
       'Real effective exchange rate (deflator: consumer price indices - 42 trading partners)',
       'Money market interest rate',  'HICP - Total goods',
       'HICP - Industrial goods', 'HICP - Total services',
       'Balance for values/ratio for indices', 'Imports', 'Exports']]

y_train=unemployment[['pct change unemployment']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(11),
                     coef.sort_values().tail(11)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_unemployment=important_features.index.tolist()




#%%

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


gdp_predictors=full_quarterly_frame[important_feature_names_gdp+['TIME']]

gdp_predictors=pd.merge(gdp_predictors, gdp_filtered[['TIME','pct change in gdp','gdp']],on='TIME',how='left')

gdp_predictors.set_index("TIME", inplace=True)

y = gdp_predictors["pct change in gdp"]
X = gdp_predictors[important_feature_names_gdp]


# Keep last quarter for testing
train_size = len(gdp_predictors) - 4
train_y = y.iloc[:train_size]
train_X = X.iloc[:train_size]

# Define test dataset (last quarter for out-of-sample validation)
test_y = y.iloc[train_size:]
test_X = X.iloc[train_size:]


# Example ARIMA order, can be determined via auto_arima
order = (1, 1, 1)  
model = ARIMA(train_y, exog=train_X, order=order)
arimax_result = model.fit()
#accuracy(arimax_result)


# Print summary
print(arimax_result.summary())

y_pred = arimax_result.fittedvalues  # Model's predicted values

from sklearn.metrics import mean_squared_error, mean_absolute_error

forecast_arimax = arimax_result.forecast(steps=4, exog=test_X)

# Convert to NumPy array for evaluation
predicted_values = np.array(forecast_arimax)

# Compute RMSE
rmse_arimax = np.sqrt(mean_squared_error(test_y, predicted_values))
print(f"ARIMAX RMSE: {rmse_arimax:.4f}")

# Compute MAE
mae_arimax = mean_absolute_error(test_y, predicted_values)
print(f"ARIMAX MAE: {mae_arimax:.4f}")

# # Compute MAPE
# mape_arimax = np.mean(np.abs((test_y - predicted_values) / test_y)) * 100
# print(f"ARIMAX MAPE: {mape_arimax:.2f}%")






# In-Sample Predictions (Training Data)
in_sample_pred = arimax_result.fittedvalues

# Compute In-Sample Forecast Errors
in_sample_errors = train_y - in_sample_pred

# Print In-Sample RMSE & MAE
rmse_in_sample = np.sqrt(mean_squared_error(train_y, in_sample_pred))
mae_in_sample = mean_absolute_error(train_y, in_sample_pred)

print(f"In-Sample RMSE: {rmse_in_sample:.4f}")
print(f"In-Sample MAE: {mae_in_sample:.4f}")





#%%
#consumption_predictors

consumption_predictors=full_quarterly_frame[important_feature_names_consumption+['TIME']]
consumption_predictors=pd.merge(consumption_predictors, consumption_filtered[['TIME','Private final consumption', 'consumption change']],on='TIME',how='left')


consumption_predictors.set_index("TIME", inplace=True)

y = consumption_predictors["consumption change"]
X = consumption_predictors[important_feature_names_consumption]


# Keep last quarter for testing
train_size = len(consumption_predictors) - 1 
train_y = y.iloc[:train_size]
train_X = X.iloc[:train_size]

# Define test dataset (last quarter for out-of-sample validation)
test_y = y.iloc[train_size:]
test_X = X.iloc[train_size:]


# Example ARIMA order, can be determined via auto_arima
order = (1, 1, 1)  
model = ARIMA(train_y, exog=train_X, order=order)
arimax_result = model.fit()
#accuracy(arimax_result)


# Print summary
print(arimax_result.summary())

y_pred = arimax_result.fittedvalues  # Model's predicted values

from sklearn.metrics import mean_squared_error, mean_absolute_error

forecast_arimax = arimax_result.forecast(steps=1, exog=test_X)

# Convert to NumPy array for evaluation
predicted_values = np.array(forecast_arimax)

# Compute RMSE
rmse_arimax = np.sqrt(mean_squared_error(test_y, predicted_values))
print(f"ARIMAX RMSE: {rmse_arimax:.4f}")

# Compute MAE
mae_arimax = mean_absolute_error(test_y, predicted_values)
print(f"ARIMAX MAE: {mae_arimax:.4f}")

# # Compute MAPE
# mape_arimax = np.mean(np.abs((test_y - predicted_values) / test_y)) * 100
# print(f"ARIMAX MAPE: {mape_arimax:.2f}%")










# In-Sample Predictions (Training Data)
in_sample_pred = arimax_result.fittedvalues

# Compute In-Sample Forecast Errors
in_sample_errors = train_y - in_sample_pred

# Print In-Sample RMSE & MAE
rmse_in_sample = np.sqrt(mean_squared_error(train_y, in_sample_pred))
mae_in_sample = mean_absolute_error(train_y, in_sample_pred)

print(f"In-Sample RMSE: {rmse_in_sample:.4f}")
print(f"In-Sample MAE: {mae_in_sample:.4f}")








consumption_predictors.to_Excel









#%%

unemployment["DATE"] = pd.to_datetime(unemployment["DATE"])
unemployment["DATE"] = unemployment["DATE"].dt.to_period("M").dt.to_timestamp()
unemployment.set_index("DATE", inplace=True)
##Unemployment predictors



y = unemployment['pct change unemployment']
X = monthly_merged_data[important_feature_names_unemployment]


# Keep last quarter for testing
train_size = len(unemployment) - 1 
train_y = y.iloc[:train_size]
train_X = X.iloc[:train_size]

# Define test dataset (last quarter for out-of-sample validation)
test_y = y.iloc[train_size:]
test_X = X.iloc[train_size:]


# Example ARIMA order, can be determined via auto_arima
order = (1, 1, 0)  
model = ARIMA(train_y, exog=train_X, order=order)
arimax_result = model.fit()
#accuracy(arimax_result)


# Print summary
print(arimax_result.summary())

y_pred = arimax_result.fittedvalues  # Model's predicted values

from sklearn.metrics import mean_squared_error, mean_absolute_error

forecast_arimax = arimax_result.forecast(steps=1, exog=test_X)

# Convert to NumPy array for evaluation
predicted_values = np.array(forecast_arimax)

# Compute RMSE
rmse_arimax = np.sqrt(mean_squared_error(test_y, predicted_values))
print(f"ARIMAX RMSE: {rmse_arimax:.4f}")

# Compute MAE
mae_arimax = mean_absolute_error(test_y, predicted_values)
print(f"ARIMAX MAE: {mae_arimax:.4f}")

# # Compute MAPE
# mape_arimax = np.mean(np.abs((test_y - predicted_values) / test_y)) * 100
# print(f"ARIMAX MAPE: {mape_arimax:.2f}%")






# In-Sample Predictions (Training Data)
in_sample_pred = arimax_result.fittedvalues

# Compute In-Sample Forecast Errors
in_sample_errors = train_y - in_sample_pred

# Print In-Sample RMSE & MAE
rmse_in_sample = np.sqrt(mean_squared_error(train_y, in_sample_pred))
mae_in_sample = mean_absolute_error(train_y, in_sample_pred)

print(f"In-Sample RMSE: {rmse_in_sample:.4f}")
print(f"In-Sample MAE: {mae_in_sample:.4f}")


























gdp_predictors.to_excel('C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/6. Final Data for use/gdp_predictors.xlsx')


#C:\Users\Bhargavi\Documents\VLK Case study\Stepwise Working Folder\6. Final Data for use
consumption_predictors=pd.merge(consumption_predictors, consumption_filtered[['TIME','consumption change','Private final consumption']],on='TIME',how='left')

#gdp_predictors.set_index("TIME", inplace=True)

consumption_predictors.to_excel('C:/Users/Bhargavi/Documents/VLK Case study/Stepwise Working Folder/6. Final Data for use/consumption_predictors.xlsx')



import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
#from arch.unitroot import DM

# Forecast for the next 19 observations


#%%


#Downloads
quarterly_data_us=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/USGDP Fixed.xlsx")


quarterly_data_us.set_index("Series Description", inplace=True)



y = quarterly_data_us["GDP Change"]
X = quarterly_data_us[['All sectors; U.S. wealth ',
       'All sectors; U.S. wealth (IMA)',
       'All sectors; Difference between U.S. wealth calculations',
       'All sectors; U.S. official reserve assets',
       'All sectors; monetary gold and SDRs; assets',
       'All sectors; checkable deposits and currency; asset',
       'All sectors; Treasury securities; asset',
       'All sectors; corporate and foreign bonds; asset',
       'All sectors; commercial mortgages; asset',
       'All sectors; U.S. official reserve assets; liability',
       'All sectors; corporate and foreign bonds; liability',
       'All sectors; other loans and advances; liability',
       'All sectors; taxes payable; liability',
       'All sectors; total interbank transactions; asset',
       'All sectors; short-term debt securities issued by residents; asset',
       'All sectors; long-term debt securities issued by residents; asset',
       'All sectors; total financial assets ',
       'All sectors; total interbank transactions; liability',
       'All sectors; total loans; liability', 'All sectors; total liabilities',
       'All sectors; total capital expenditures',
       'All sectors; gross investment',
       'All sectors; net saving including net capital transfers paid (sum of pieces)',
       'All sectors; disposable income, net (IMA)',
       'All sectors; social benefits received (IMA)',
       'All sectors; U.S. wealth .1',
       'Motor vehicle loans owned and securitized, not seasonally adjusted level',
       'Motor vehicle loans owned and securitized, not seasonally adjusted flow, monthly rate',
       'Finance rate on consumer installment loans at commercial banks, new autos 48 month loan; not seasonally adjusted',
       'Finance rate on personal loans at commercial banks, 24 month loan; not seasonally adjusted',
       'Commercial bank interest rate on credit card plans, all accounts; not seasonally adjusted',
       'Commercial bank interest rate on credit card plans, accounts assessed interest; not seasonally adjusted',
       'All sectors; reinvested earnings (net); received (IMA)']]


# Keep last quarter for testing
train_size = len(quarterly_data_us) -  36
train_y = y.iloc[:train_size]
train_X = X.iloc[:train_size]

# Define test dataset (last quarter for out-of-sample validation)
test_y = y.iloc[train_size:]
test_X = X.iloc[train_size:]


# Example ARIMA order, can be determined via auto_arima
order = (1, 1, 1)  
model = ARIMA(train_y, exog=train_X, order=order)
arimax_result = model.fit()
#accuracy(arimax_result)


# Print summary
print(arimax_result.summary())

y_pred = arimax_result.fittedvalues  # Model's predicted values

from sklearn.metrics import mean_squared_error, mean_absolute_error

forecast_arimax = arimax_result.forecast(steps=36, exog=test_X)

# Convert to NumPy array for evaluation
predicted_values = np.array(forecast_arimax)

# Compute RMSE
rmse_arimax = np.sqrt(mean_squared_error(test_y, predicted_values))
print(f"ARIMAX RMSE: {rmse_arimax:.4f}")

# Compute MAE
mae_arimax = mean_absolute_error(test_y, predicted_values)
print(f"ARIMAX MAE: {mae_arimax:.4f}")


# In-Sample Predictions (Training Data)
in_sample_pred = arimax_result.fittedvalues

# Compute In-Sample Forecast Errors
in_sample_errors = train_y - in_sample_pred

# Print In-Sample RMSE & MAE
rmse_in_sample = np.sqrt(mean_squared_error(train_y, in_sample_pred))
mae_in_sample = mean_absolute_error(train_y, in_sample_pred)

print(f"In-Sample RMSE: {rmse_in_sample:.4f}")
print(f"In-Sample MAE: {mae_in_sample:.4f}")


gdp_predictors=full_quarterly_frame[important_feature_names_gdp+['TIME']]

gdp_predictors=pd.merge(gdp_predictors, gdp_filtered[['TIME','pct change in gdp','gdp']],on='TIME',how='left')

gdp_predictors.set_index("TIME", inplace=True)

y = gdp_predictors["pct change in gdp"]
X = gdp_predictors[important_feature_names_gdp]


####gdp prediction
df = gdp_predictors.copy()
target_variable = 'pct change in gdp'
exog_variables = important_feature_names_gdp  # List of selected predictors

# Define the number of observations for training
train_size = 80  # First 80 observations for training
errors = []  # Store prediction errors

# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X)[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')


####Consumption
y = consumption_predictors["consumption change"]
X = consumption_predictors[important_feature_names_consumption]

df = consumption_predictors.copy()
target_variable = 'consumption change'
exog_variables = important_feature_names_consumption  # List of selected predictors

# Define the number of observations for training
train_size = 80  # First 80 observations for training
errors = []  # Store prediction errors

# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X)[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')



#####Unemployement

y = unemployment['pct change unemployment']
X = df[important_feature_names_unemployment]

unemployment_merged = pd.merge(unemployment, monthly_merged_data, left_index=True, right_index=True, how='inner')

df = unemployment_merged.copy()
target_variable = 'pct change unemployment'
exog_variables = important_feature_names_unemployment  # List of selected predictors

# Define the number of observations for training
train_size = 238  # First 80 observations for training
errors = []  # Store prediction errors

# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X)[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')


prediction = model_fit.forecast(steps=1, exog=test_X)


#########################################################################################################

##########US Quarterly
#C:\Users\Bhargavi\Documents\VLK Case study\Final Data\USA


US_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Quarterly-AllVars+Target-USA.xlsx")

US_quarterly.set_index("Series Description", inplace=True)

X_train = US_quarterly.drop(columns=[ 'GDP'])

y_train=US_quarterly[['GDP']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_gdp_us=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = US_quarterly.copy()
target_variable = 'GDP'
exog_variables = important_feature_names_gdp_us  # List of selected predictors

# Define the number of observations for training
train_size = 80  # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)
    prediction_frame.append(prediction)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')


pred_vs_act_us_gdp = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])



import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

US_gdp_residuals=model_fit.resid
US_gdp_residuals = pd.DataFrame(US_gdp_residuals, columns=['Residuals'])
US_gdp_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(US_gdp_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(US_gdp_residuals, ax=axes[0], lags=20)
plot_pacf(US_gdp_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(US_gdp_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of GDP US Residuals")
plt.show()

shapiro_test = stats.shapiro(US_gdp_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(US_gdp_fitted, US_gdp_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(US_gdp_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])


# Interpretation:
# - If p-value > 0.05 → Residuals are normally distributed ✅
# - If p-value < 0.05 → Residuals are **not** normally distributed ❌




ss_total = np.sum((pred_vs_act_us_gdp['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_us_gdp['Actual'] - pred_vs_act_us_gdp['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")





###################################################################################################


#####US Monthly

#####PCE
US_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Monthly-ARIMA-USA-AllVars+Targets.xlsx")

US_monthly.set_index("Series Description", inplace=True)

X_train = US_monthly.drop(columns=[ 'PCE','UNEMP'])

y_train=US_monthly[['PCE']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_pce_us=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = US_monthly.copy()
target_variable = 'PCE'
exog_variables = important_feature_names_pce_us  # List of selected predictors

# Define the number of observations for training
train_size = 237 # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]
    prediction_frame.append(prediction)
    # Compute Error
    error = test_y - prediction
    errors.append(error)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')


pred_vs_act_us_pce = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])


US_PCE_residuals=model_fit.resid
US_PCE_residuals = pd.DataFrame(US_PCE_residuals, columns=['Residuals'])
US_PCE_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(US_PCE_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(US_PCE_residuals, ax=axes[0], lags=20)
plot_pacf(US_PCE_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(US_PCE_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of US PCE Residuals")
plt.show()

shapiro_test = stats.shapiro(US_PCE_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(US_PCE_fitted, US_PCE_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(US_PCE_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])



ss_total = np.sum((pred_vs_act_us_pce['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_us_pce['Actual'] - pred_vs_act_us_pce['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")







###############################################################################################
#####UNEMP
US_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Monthly-ARIMA-USA-AllVars+Targets.xlsx")

US_monthly.set_index("Series Description", inplace=True)

X_train = US_monthly.drop(columns=[ 'PCE','UNEMP'])

y_train=US_monthly[['UNEMP']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_unemp_us=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = US_monthly.copy()
target_variable = 'UNEMP'
exog_variables = important_feature_names_unemp_us  # List of selected predictors

# Define the number of observations for training
train_size = 237  # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(2,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]
    prediction_frame.append(prediction)
    # Compute Error
    error = test_y - prediction
    errors.append(error)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')

pred_vs_act_us_unemp = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])


US_UNEMP_residuals=model_fit.resid
US_UNEMP_residuals = pd.DataFrame(US_UNEMP_residuals, columns=['Residuals'])
US_UNEMP_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(US_UNEMP_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(US_UNEMP_residuals, ax=axes[0], lags=20)
plot_pacf(US_UNEMP_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(US_UNEMP_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of US Unemployment Residuals")
plt.show()

shapiro_test = stats.shapiro(US_UNEMP_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(US_UNEMP_fitted, US_UNEMP_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(US_UNEMP_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])


ss_total = np.sum((pred_vs_act_us_unemp['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_us_unemp['Actual'] - pred_vs_act_us_unemp['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")



US_UNEMP_residuals.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/US UNEMP ARIMAX Residuals.xlsx")
pred_vs_act_us_unemp.to_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/Pred vs actual us unemp arimax.xlsx")



############################################################################
##Europe

#####GDP
EU_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Quarterly-ARIMA-AllVars-Europe+Targets.xlsx")

EU_quarterly.set_index("Series Description", inplace=True)

X_train = EU_quarterly.drop(columns=[ 'GDP','PCE'])

y_train=EU_quarterly[['GDP']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_gdp_eu=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = EU_quarterly.copy()
target_variable = 'GDP'
exog_variables = important_feature_names_gdp_eu  # List of selected predictors

# Define the number of observations for training
train_size = 80  # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)
    prediction_frame.append(prediction)
    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')

pred_vs_act_eu_gdp = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])


EU_GDP_residuals=model_fit.resid
EU_GDP_residuals = pd.DataFrame(EU_GDP_residuals, columns=['Residuals'])

EU_GDP_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(EU_GDP_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(EU_GDP_residuals, ax=axes[0], lags=20)
plot_pacf(EU_GDP_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(EU_GDP_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of EU GDP Residuals")
plt.show()

shapiro_test = stats.shapiro(EU_GDP_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(EU_GDP_fitted, EU_GDP_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(EU_GDP_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])


ss_total = np.sum((pred_vs_act_eu_gdp['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_eu_gdp['Actual'] - pred_vs_act_eu_gdp['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")






#######################################################################################


####PCE
EU_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Quarterly-ARIMA-AllVars-Europe+Targets.xlsx")

EU_quarterly.set_index("Series Description", inplace=True)

X_train = EU_quarterly.drop(columns=[ 'GDP','PCE'])

y_train=EU_quarterly[['PCE']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_pce_eu=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = EU_quarterly.copy()
target_variable = 'PCE'
exog_variables = important_feature_names_pce_eu  # List of selected predictors

# Define the number of observations for training
train_size = 80  # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)
    prediction_frame.append(prediction)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')

pred_vs_act_eu_pce = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])


EU_PCE_residuals=model_fit.resid
EU_PCE_residuals = pd.DataFrame(EU_PCE_residuals, columns=['Residuals'])

EU_PCE_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(EU_PCE_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(EU_PCE_residuals, ax=axes[0], lags=20)
plot_pacf(EU_PCE_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(EU_PCE_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of EU PCE Residuals")
plt.show()

shapiro_test = stats.shapiro(EU_PCE_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(EU_PCE_fitted, EU_PCE_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(EU_PCE_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])



ss_total = np.sum((pred_vs_act_eu_pce['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_eu_pce['Actual'] - pred_vs_act_eu_pce['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")



#######################################################################################
######EU monthly
#UNEMP

EU_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Monthly-ARIMA-EU-AllVars+Targets.xlsx")
EU_monthly.drop(columns=[ 'TIME'],inplace=True)

EU_monthly.set_index("Series Description", inplace=True)

X_train = EU_monthly.drop(columns=[ 'UNEMP'])

y_train=EU_monthly[['UNEMP']]
#y_train=gdp_filtered[['pct change in gdp']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

alpha = [0.0001, 0.001, 0.01, 0.1,]
max_iter = [1000, 10000]
l1_ratio = np.arange(0.0, 1.0, 0.1)
tol = [0.5]

elasticnet_gscv = GridSearchCV(estimator=ElasticNet(), 
                                param_grid={'alpha': alpha,
                                            'max_iter': max_iter,
                                            'l1_ratio': l1_ratio,
                                            'tol':tol},   
                                scoring='r2',
                                cv=5)



elasticnet_gscv.fit(X_scaled, y_train)
elasticnet_gscv.best_params_

elasticnet = ElasticNet(alpha = elasticnet_gscv.best_params_['alpha'], 
                        max_iter = elasticnet_gscv.best_params_['max_iter'],
                        l1_ratio = elasticnet_gscv.best_params_['l1_ratio'],
                        tol = elasticnet_gscv.best_params_['tol'])


elasticnet = ElasticNet(alpha = 0.001, 
                        max_iter = 400,
                        l1_ratio = 0.75,
                        tol = 0.5)

elasticnet.fit(X_scaled, y_train)

selected_features_consumption = np.where(elasticnet.coef_ != 0)[0]


coef = pd.Series(elasticnet.coef_, index = X_train.columns)
important_features = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
important_features.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")


important_feature_names_unemp_eu=important_features.index.tolist()
#important_feature_names_consumption = important_features.index.tolist()

df = EU_monthly.copy()
target_variable = 'UNEMP'
exog_variables = important_feature_names_unemp_eu  # List of selected predictors

# Define the number of observations for training
train_size = 237  # First 80 observations for training
errors = []  # Store prediction errors
prediction_frame=[]
# Initialize train and test set
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test = test.apply(pd.to_numeric, errors='coerce')

# Rolling Forecast Evaluation
for i in range(len(test)):
    # Extract training data
    train_y = train[target_variable]
    train_X = train[exog_variables]
    train_X = train_X.apply(pd.to_numeric, errors='coerce')

    # Fit ARIMAX Model
    model = ARIMA(train_y, exog=train_X, order=(1,1,1))
    model_fit = model.fit()

    # Extract test values
    test_y = test.iloc[i][target_variable]
    test_X = test.iloc[i][exog_variables].values.reshape(1, -1)  # Reshape for compatibility

    # Forecast next value
    prediction = model_fit.forecast(steps=1, exog=test_X).iloc[0]

    # Compute Error
    error = test_y - prediction
    errors.append(error)
    prediction_frame.append(prediction)

    # Append actual test observation to training set
    new_row = test.iloc[[i]]  # Get current test observation as DataFrame
    train = pd.concat([train, new_row])  # Append to training data

# Convert errors list to array for metric calculations
errors = np.array(errors)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)]))
mae = mean_absolute_error(test[target_variable][:len(errors)], errors + test[target_variable][:len(errors)])

print(f'Rolling Forecast RMSE: {rmse}')
print(f'Rolling Forecast MAE: {mae}')


pred_vs_act_eu_unemp = pd.DataFrame({
    'Actual': test[target_variable][:len(errors)],
    'Predicted': errors + test[target_variable][:len(errors)]
}, index=test.index[:len(errors)])



EU_UNEMP_residuals=model_fit.resid
EU_UNEMP_residuals = pd.DataFrame(EU_UNEMP_residuals, columns=['Residuals'])

EU_UNEMP_fitted=model_fit.fittedvalues


plt.figure(figsize=(10,5))
plt.plot(EU_UNEMP_residuals, label="Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("ARIMAX Residuals Over Time")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(EU_UNEMP_residuals, ax=axes[0], lags=20)
plot_pacf(EU_UNEMP_residuals, ax=axes[1], lags=20)
axes[0].set_title("ACF of Residuals")
axes[1].set_title("PACF of Residuals")
plt.show()


plt.figure(figsize=(6,5))
stats.probplot(EU_UNEMP_residuals['Residuals'], dist="norm", plot=plt)
plt.title("Q-Q Plot of EU Unemployment Residuals")
plt.show()

shapiro_test = stats.shapiro(EU_UNEMP_residuals['Residuals'])
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")



plt.scatter(EU_UNEMP_fitted, EU_UNEMP_residuals['Residuals'])
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()


ljung_box_test = acorr_ljungbox(EU_UNEMP_residuals['Residuals'], lags=[10], return_df=True)
print("Ljung-Box Test p-value:", ljung_box_test["lb_pvalue"].values[0])



ss_total = np.sum((pred_vs_act_eu_unemp['Actual'] - np.mean(train_y))**2)

ss_residual = np.sum((pred_vs_act_eu_unemp['Actual'] - pred_vs_act_eu_unemp['Predicted'])**2)

r2_out_sample = 1 - (ss_residual / ss_total)

print(f"Out-of-Sample R²: {r2_out_sample:.4f}")




###################################################################################



#with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
with pd.ExcelWriter('C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/Residuals.xlsx', engine='openpyxl') as writer:
    EU_UNEMP_residuals.to_excel(writer, sheet_name='EU_UNEMP', index=True)
    EU_PCE_residuals.to_excel(writer, sheet_name='EU_PCE', index=True)
    EU_GDP_residuals.to_excel(writer, sheet_name='EU_GDP', index=True)
    US_UNEMP_residuals.to_excel(writer, sheet_name='US_UNEMP', index=True)
    US_PCE_residuals.to_excel(writer, sheet_name='US_PCE', index=True)
    US_gdp_residuals.to_excel(writer, sheet_name='US_GDP', index=True)




with pd.ExcelWriter('C:/Users/Bhargavi/Documents/VLK Case study/Final Data/pred vs actual.xlsx', engine='openpyxl') as writer:
    pred_vs_act_eu_unemp.to_excel(writer, sheet_name='EU_UNEMP', index=True)
    pred_vs_act_eu_pce.to_excel(writer, sheet_name='EU_PCE', index=True)
    pred_vs_act_eu_gdp.to_excel(writer, sheet_name='EU_GDP', index=True)
    pred_vs_act_us_unemp.to_excel(writer, sheet_name='US_UNEMP', index=True)
    pred_vs_act_us_pce.to_excel(writer, sheet_name='US_PCE', index=True)
    pred_vs_act_us_gdp.to_excel(writer, sheet_name='US_GDP', index=True)





#######################################################################################

#checks
temp_us_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Monthly.xlsx")
temp_us_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Quarterly.xlsx")
temp_us_daily=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Daily.xlsx")


existing_variables_monthly = [var for var in important_feature_names_unemp_us if var in temp_us_monthly.columns]
existing_variables_daily = [var for var in important_feature_names_unemp_us if var in temp_us_daily.columns]


# Extract only those columns
df_selected = df[existing_variables]












