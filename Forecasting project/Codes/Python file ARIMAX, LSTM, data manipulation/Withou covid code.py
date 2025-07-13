# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:33:34 2025

@author: Bhargavi
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import invgamma
from tabulate import tabulate
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm



##########US Quarterly

#C:\Users\Bhargavi\Documents\VLK Case study\Final Data\Without covid\USA


US_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/USA/2000-2025-Quarterly-USA-Covid.xlsx")

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
train_size = 72  # First 80 observations for training
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
    model = ARIMA(train_y, exog=train_X, order=(1,2,1))
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
US_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/USA/2000-2025-Monthly-ARIMA-USA-Covid.xlsx")

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
train_size = 216 # First 80 observations for training
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
#US_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Monthly-ARIMA-USA-AllVars+Targets.xlsx")

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
train_size = 216  # First 80 observations for training
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







####################################################################################################

#C:\Users\Bhargavi\Documents\VLK Case study\Final Data\Without covid\Europe
##Europe

#####GDP
EU_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/Europe/2000-2025-Quarterly-ARIMA-Europe-Covid.xlsx")

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
train_size = 72 # First 80 observations for training
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
#EU_quarterly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Quarterly-ARIMA-AllVars-Europe+Targets.xlsx")

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
train_size = 72  # First 80 observations for training
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

EU_monthly=pd.read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/Europe/2000-2025-Monthly-ARIMA-EU-Covid.xlsx")
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
train_size = 216  # First 80 observations for training
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
with pd.ExcelWriter('C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/Residuals COVID ARIMAX.xlsx', engine='openpyxl') as writer:
    EU_UNEMP_residuals.to_excel(writer, sheet_name='EU_UNEMP', index=True)
    EU_PCE_residuals.to_excel(writer, sheet_name='EU_PCE', index=True)
    EU_GDP_residuals.to_excel(writer, sheet_name='EU_GDP', index=True)
    US_UNEMP_residuals.to_excel(writer, sheet_name='US_UNEMP', index=True)
    US_PCE_residuals.to_excel(writer, sheet_name='US_PCE', index=True)
    US_gdp_residuals.to_excel(writer, sheet_name='US_GDP', index=True)




with pd.ExcelWriter('C:/Users/Bhargavi/Documents/VLK Case study/Final Data/pred vs actual covid ARIMAX.xlsx', engine='openpyxl') as writer:
    pred_vs_act_eu_unemp.to_excel(writer, sheet_name='EU_UNEMP', index=True)
    pred_vs_act_eu_pce.to_excel(writer, sheet_name='EU_PCE', index=True)
    pred_vs_act_eu_gdp.to_excel(writer, sheet_name='EU_GDP', index=True)
    pred_vs_act_us_unemp.to_excel(writer, sheet_name='US_UNEMP', index=True)
    pred_vs_act_us_pce.to_excel(writer, sheet_name='US_PCE', index=True)
    pred_vs_act_us_gdp.to_excel(writer, sheet_name='US_GDP', index=True)


