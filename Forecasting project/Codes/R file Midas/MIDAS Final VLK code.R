library(readxl)
library(midasr)
library(dplyr)

library(Metrics)  # For RMSE, MAE calculations

################MIDAS PREDICTIONS WITH COVID####################

####EU####

quarterly_data_EU=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Quarterly-ARIMA-AllVars-Europe+Targets.xlsx")
monthly_data_EU=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/2000-2025-Monthly-ARIMA-EU-AllVars+Targets.xlsx")


####EU GDP ROUGH ####



x_ici=monthly_data_EU$`Industrial confidence indicator`
#x_hicp_a=monthly_data_EU$`HICP - All items (HICP=harmonized index of consumer prices)`
#x_hicp_C=monthly_data_EU$`HICP - Clothing and footwear`
x_ee=monthly_data_EU$`Employment expectation index in next three months`
x_ci=monthly_data_EU$`Consumer confidence indicator`
x_ei=monthly_data_EU$`Economic sentiment indicator`
x_reer=monthly_data_EU$`Real effective exchange rate (deflator: consumer price indices - 37 trading partners)`
x_exp=monthly_data_EU$`Money market interest rate`
x_eer=monthly_data_EU$`Euro Exchange rate`
x_gdp=quarterly_data_EU$GDP


short_names <- c(
  "Factors limiting the production - other" = "x_factors_other",
  "Competitive position on foreign markets inside the EU over the past three months" = "x_comp_inside_EU",
  "Assessment of current production capacity" = "x_prod_capacity",
  "Current account" = "x_current_acc",
  "Factors limiting the production - labour" = "x_factors_labour",
  "Short-term - loans" = "x_short_loans",
  "Short-term debt securities" = "x_short_debt_sec",
  "Secondary income" = "x_sec_income",
  "Capital account" = "x_cap_acc",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Industry turnover index" = "x_industry_turn",
  "Home improvements over the next 12 months" = "x_home_improve",
  "Factors limiting the production - insufficient demand" = "x_factors_demand",
  "New orders in recent months" = "x_new_orders"
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], quarterly_data_EU[[var]])
}

beta0 <- midas_r(x_gdp ~   mls(x_ici, 0:2, 3, nealmon)+
                   mls(x_ee, 0:2, 3, nealmon)+
                   mls(x_ci, 0:2, 3, nealmon)+
                   mls(x_ei, 0:2, 3, nealmon)+
                   mls(x_reer, 0:2, 3, nealmon)+
                   mls(x_exp, 0:2, 3, nealmon)+
                   mls(x_eer, 0:2, 3, nealmon)+
                   x_factors_other+x_comp_inside_EU+x_prod_capacity+
                   x_new_orders+x_industry_turn+x_home_improve+x_factors_demand+
                   x_cap_acc+x_short_loans+x_short_debt_sec
                 
                 ,
                 start = list(
                   # Starting values for nbeta constraint
                   x_ici = c(0.5,-0.5),  
                   # Starting values for Nealmon constraints
                   x_ee = c(0.5, -0.5),
                   x_ci = c(0.5, -0.5),
                   x_ei = c(0.5, -0.5),
                   x_reer = c(0.5, -0.5),
                   x_ei = c(0.5, -0.5),
                   x_exp = c(0.5, -0.5),
                   x_eer = c(0.5, -0.5)
                 ))
summary(beta0)

predict(beta0, newdata = list(#train_x_gdp=train_x_gdp
  x_ici=t1,
  x_ee = t2,
  x_ci = t3,
  x_reer = t4,
  x_ei = t5,
  x_exp = t6,
  x_eer = t7,
  x_factors_other = x_factors_other[train_size + 1],
  x_comp_inside_EU = x_comp_inside_EU[train_size + 1],
  x_prod_capacity = x_prod_capacity[train_size + 1],
  x_new_orders = x_new_orders[train_size + 1],
  x_industry_turn = x_industry_turn[train_size + 1],
  x_home_improve = x_home_improve[train_size + 1],
  x_factors_demand = x_factors_demand[train_size + 1],
  x_cap_acc = x_cap_acc[train_size + 1],
  x_short_loans = x_short_loans[train_size + 1],
  x_short_debt_sec = x_short_debt_sec[train_size + 1]))


library(midasr)
library(Metrics)  # For RMSE, MAE calculations

# Define the number of observations
n <- length(x_gdp)

# Split into 80% train, 20% test
train_size <- floor(0.8 * n)
train_x_gdp <- x_gdp[1:train_size]
test_x_gdp <- x_gdp[(train_size+1):n]



# Store forecasts
predictions <- numeric(length(test_x_gdp))
j=1
# Rolling Forecast Loop
for (i in 1:length(test_x_gdp)) {
  # Define training index dynamically
  train_index <- 1:(train_size + i - 1)
  train_index_2 <- 1:(237 + j -1)
  train_index_3=(237+j):(237+j+2)
  a1=x_ici[train_index_2]
  a2=x_ee[train_index_2]
  a3=x_ci[train_index_2]
  a4=x_ei[train_index_2]
  a5=x_reer[train_index_2]
  a6=x_exp[train_index_2]
  a7=x_eer[train_index_2]
  b1=x_factors_other[train_index]
  b2=x_comp_inside_EU[train_index]
  b3=x_prod_capacity[train_index]
  b4=x_new_orders[train_index]
  b5=x_industry_turn[train_index]
  b6=x_home_improve[train_index]
  b7=x_factors_demand[train_index]
  b8=x_cap_acc[train_index]
  b9=x_short_loans[train_index]
  b10=x_short_debt_sec[train_index]
  # Fit MIDAS model on rolling training data
  beta0 <- midas_r(train_x_gdp ~ 
                     #mls(train_x_gdp, 1, 1) + 
                     mls(a1, 0:2, 3, nealmon) +
                     mls(a2, 0:2, 3, nealmon) +
                     mls(a3, 0:2, 3, nealmon) +
                     mls(a4, 0:2, 3, nealmon) +
                     mls(a5, 0:2, 3, nealmon) +
                     mls(a6, 0:2, 3, nealmon) +
                     mls(a7, 0:2, 3, nealmon) +
                     b1 +
                     b2 +
                     b3+
                     b4+
                     b5+
                     b6+
                     b7+
                     b8+
                     b9+b10
                   ,
                   start = list(
                     a1 = c(0.6,-0.5),  
                     a2 = c(0.6, -0.5),
                     a3 = c(0.6, -0.5),
                     a4 = c(0.6, -0.5),
                     a5 = c(0.6, -0.5),
                     a6 = c(0.6, -0.5),
                     a7 = c(0.6, -0.5)
                   ))
  
  # Forecast next value
  t1=x_ici[train_index_3]
  t2=x_ee[train_index_3]
  t3=x_ci[train_index_3]
  t4=x_ei[train_index_3]
  t5=x_reer[train_index_3]
  t6=x_exp[train_index_3]
  t7=x_eer[train_index_3]
  forecast(beta0, newdata = list(#train_x_gdp=train_x_gdp
    a1=t1,
    a2 = t2,
    a3 = t3,
    a4 = t4,
    a5 = t5,
    a6 = t6,
    a7 = t7,
    b1 = x_factors_other[train_size + i],
    b2 = x_comp_inside_EU[train_size + i],
    b3 = x_prod_capacity[train_size + i],
    b4 = x_new_orders[train_size + i],
    b5 = x_industry_turn[train_size + i],
    b6 = x_home_improve[train_size + i],
    b7 = x_factors_demand[train_size + i],
    b8 = x_cap_acc[train_size + i],
    b9 = x_short_loans[train_size + i],
    b10 = x_short_debt_sec[train_size + i]
  ))
  
  # Store forecast
  predictions[i] <- pred$mean
  
  # Expand training set with actual value (Rolling Window)
  train_x_gdp <- c(train_x_gdp, test_x_gdp[i])
  j=j+3
}

# Evaluate Forecasting Performance
rmse_value <- rmse(test_x_gdp, predictions)
mae_value <- mae(test_x_gdp, predictions)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))

# Create a DataFrame of Actual vs. Predicted
forecast_results <- data.frame(Actual = test_x_gdp, Predicted = predictions)


predicted_eu_gdp_errors=pred$residuals
predicted_eu_gdp=pred$fitted

p=predict(beta0, newdata = list(#train_x_gdp=train_x_gdp
  x_ici=t1,
  x_ee = t2,
  x_ci = t3,
  x_reer = t4,
  x_ei = t5,
  x_exp = t6,
  x_eer = t7,
  x_factors_other = x_factors_other[train_size + 1],
  x_comp_inside_EU = x_comp_inside_EU[train_size + 1],
  x_prod_capacity = x_prod_capacity[train_size + 1],
  x_new_orders = x_new_orders[train_size + 1],
  x_industry_turn = x_industry_turn[train_size + 1],
  x_home_improve = x_home_improve[train_size + 1],
  x_factors_demand = x_factors_demand[train_size + 1],
  x_cap_acc = x_cap_acc[train_size + 1],
  x_short_loans = x_short_loans[train_size + 1],
  x_short_debt_sec = x_short_debt_sec[train_size + 1]))


residuals=p-train_x_gdp[1:98]


pred_vs_Actual=data.frame(Actual = test_x_gdp,predicted=predicted_eu_gdp[80:99])
rmse_value <- rmse(pred_vs_Actual$Actual, pred_vs_Actual$predicted)
mae_value <- mae(pred_vs_Actual$Actual, pred_vs_Actual$predicted)
residuals_frame_eu_gdp=data.frame(residuals_gdp=residuals)


###############################################################################################











##########################################################################################

set.seed(123)
######EU GDP#####


short_names <- c(
  "Factors limiting the production - other" = "x_factors_other",
  "Competitive position on foreign markets inside the EU over the past three months" = "x_comp_inside_EU",
  "Assessment of current production capacity" = "x_prod_capacity",
  "Current account" = "x_current_acc",
  "Factors limiting the production - labour" = "x_factors_labour",
  "Short-term - loans" = "x_short_loans",
  "Short-term debt securities" = "x_short_debt_sec",
  "Secondary income" = "x_sec_income",
  "Capital account" = "x_cap_acc",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Industry turnover index" = "x_industry_turn",
  "Home improvements over the next 12 months" = "x_home_improve",
  "Factors limiting the production - insufficient demand" = "x_factors_demand",
  "New orders in recent months" = "x_new_orders"
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], quarterly_data_EU[[var]])
}



x_ici=monthly_data_EU$`Industrial confidence indicator`
#x_hicp_a=monthly_data_EU$`HICP - All items (HICP=harmonized index of consumer prices)`
#x_hicp_C=monthly_data_EU$`HICP - Clothing and footwear`
x_ee=monthly_data_EU$`Employment expectation index in next three months`
x_ci=monthly_data_EU$`Consumer confidence indicator`
x_ei=monthly_data_EU$`Economic sentiment indicator`
x_reer=monthly_data_EU$`Real effective exchange rate (deflator: consumer price indices - 37 trading partners)`
x_exp=monthly_data_EU$`Money market interest rate`
x_eer=monthly_data_EU$`Euro Exchange rate`
x_gdp=quarterly_data_EU$GDP



a1=x_ici
a2=x_ee
a3=x_ci
a4=x_ei
a5=x_reer
a6=x_exp
a7=x_eer
b1=x_factors_other
b2=x_comp_inside_EU
b3=x_current_acc
b4=x_new_orders
b5=x_industry_turn
b6=x_home_improve
b7=x_factors_demand
b8=x_cap_acc
b9=x_short_loans
b10=x_short_debt_sec


beta0 <- midas_r(x_gdp ~ 
                   mls(x_gdp, 1, 1) + 
                   #mls(a1, 0:2, 3, nealmon) +
                   mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   mls(a6, 0:2, 3, nealmon) +
                   mls(a7, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10
                 ,
                 start = list(
                   #   a1 = c(0.5,-0.5),  
                   a2 = c(0.6, -0.5),
                   a3 = c(0.5, -0.5),
                   a4 = c(0.5, -0.5),
                   a5 = c(0.5, -0.5),
                   a6 = c(0.5, -0.5),
                   a7 = c(0.5, -0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(x_gdp=x_gdp,a2=a2,a3=a3,a4=a4,
                                                  a5=a5,a6=a6,a7=a7,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10), insample = 1:79, outsample = 80:99,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



gdp_eu_residuals=beta0$residuals
gdp_eu_residuals_1=data.frame(residuals=gdp_eu_residuals)

gdp_eu_fitted=beta0$fitted.values

rmse_value <- rmse(x_gdp[80:99],avgf$forecast )
mae_value <- mae(x_gdp[80:99],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))

act_vs_pred_eu_gdp=data.frame(Actual=x_gdp[80:99],Predicted=avgf$forecast)


plot(gdp_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(gdp_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(gdp_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(gdp_eu_residuals, main = "Q-Q Plot of MIDAS EU GDP Residuals")
qqline(gdp_eu_residuals, col = "red")

shapiro_test <- shapiro.test(gdp_eu_residuals)
print(shapiro_test)

plot(gdp_eu_fitted, gdp_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(gdp_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((x_gdp[80:99] - mean(x_gdp))^2)
ss_residual <- sum((act_vs_pred_eu_gdp$Actual - act_vs_pred_eu_gdp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



######################################################################################

set.seed(4444)
######EU PCE#####

short_names <- c(
  "Factors limiting the production - other" = "x_factors_other",
  "Competitive position on foreign markets inside the EU over the past three months" = "x_comp_inside_EU",
  "Assessment of current production capacity" = "x_prod_capacity",
  "Current account" = "x_current_acc",
  "Factors limiting the production - labour" = "x_factors_labour",
  "Short-term - loans" = "x_short_loans",
  "Short-term debt securities" = "x_short_debt_sec",
  "Secondary income" = "x_sec_income",
  "Capital account" = "x_cap_acc",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Industry turnover index" = "x_industry_turn",
  "Home improvements over the next 12 months" = "x_home_improve",
  "Factors limiting the production - insufficient demand" = "x_factors_demand",
  "New orders in recent months" = "x_new_orders",
  "Value added, gross"="x_gross"
  
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], quarterly_data_EU[[var]])
}



x_ici=monthly_data_EU$`Industrial confidence indicator`
#x_hicp_a=monthly_data_EU$`HICP - All items (HICP=harmonized index of consumer prices)`
#x_hicp_C=monthly_data_EU$`HICP - Clothing and footwear`
x_ee=monthly_data_EU$`Retail confidence indicator`
x_ci=monthly_data_EU$`Consumer confidence indicator`
x_ei=monthly_data_EU$`Economic sentiment indicator`
x_reer=monthly_data_EU$`Money market interest rate`
x_exp=monthly_data_EU$`yeild Curve`
x_eer=monthly_data_EU$`Real effective exchange rate (deflator: consumer price indices - 37 trading partners)`
x_pce=quarterly_data_EU$PCE



a1=x_ici
a2=x_ee
a3=x_ci
a4=x_ei
a5=x_reer
a6=x_exp
a7=x_eer
b1=x_factors_other
b2=x_prod_capacity
b3=x_current_acc
b4=x_new_orders
b5=x_industry_turn
b6=x_home_improve
b7=x_factors_demand
b8=x_cap_acc
b9=x_factors_labour
b10=x_sec_income




beta0 <- midas_r(x_pce ~ 
                   mls(x_pce, 1, 1) + 
                   #mls(a1, 0:2, 3, nealmon) +
                   mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   mls(a6, 0:2, 3, nealmon) +
                   mls(a7, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10
                 ,
                 start = list(
                   #  a1 = c(0.5,-0.5),  
                   a2 = c(0.6, -0.5),
                   a3 = c(0.5, -0.5),
                   a4 = c(0.5, -0.5),
                   a5 = c(0.5, -0.5),
                   a6 = c(0.5, -0.5),
                   a7 = c(0.5, -0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(x_pce=x_pce,a2=a2,a3=a3,a4=a4,
                                                  a5=a5,a6=a6,a7=a7,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10), insample = 1:79, outsample = 80:99,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



pce_eu_residuals=beta0$residuals
pce_eu_residuals_1=data.frame(residuals=pce_eu_residuals)

pce_eu_fitted=beta0$fitted.values

rmse_value <- rmse(x_pce[80:99],avgf$forecast )
mae_value <- mae(x_pce[80:99],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_eu_pce_2=data.frame(Actual=x_pce[80:99],Predicted=avgf$forecast)


plot(pce_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(pce_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(pce_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(pce_eu_residuals, main = "Q-Q Plot of MIDAS EU PCE Residuals")
qqline(pce_eu_residuals, col = "red")

shapiro_test <- shapiro.test(pce_eu_residuals)
print(shapiro_test)

plot(pce_eu_fitted, pce_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(pce_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)



ss_total <- sum((x_pce[80:99] - mean(x_pce))^2)
ss_residual <- sum((act_vs_pred_eu_pce_2$Actual - act_vs_pred_eu_pce_2$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample




########################################################################################



#####EU UNEMP####
daily_data=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/Daily-EU-ARIMA.xlsx")

daily_data_2 = daily_data[1:6237,]

x_unemp=monthly_data_EU$UNEMP
x_op_rate=daily_data_2$`Main Refinancing Operations Rate`


short_names <- c(
  "Factors limiting the production - other" = "x_factors_other",
  "Competitive position on foreign markets inside the EU over the past three months" = "x_comp_inside_EU",
  "Assessment of current production capacity" = "x_prod_capacity",
  "Current account" = "x_current_acc",
  "Factors limiting the production - labour" = "x_factors_labour",
  "Short-term - loans" = "x_short_loans",
  "Long-term - loans"="x_long_loans",
  "Short-term debt securities" = "x_short_debt_sec",
  "Secondary income" = "x_sec_income",
  "Capital account" = "x_cap_acc",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Industry turnover index" = "x_industry_turn",
  "Home improvements over the next 12 months" = "x_home_improve",
  "Factors limiting the production - insufficient demand" = "x_factors_demand",
  "New orders in recent months" = "x_new_orders",
  "Value added, gross"="x_gross",
  "Employment expectation index in next three months"="x_Emp_Exp_ind",
  "Retail confidence indicator"="x_ret_ci",
  "Debt securities"="x_Debt_sec",
  "House price index"="x_hs_pr_ind",
  "Consumer confidence indicator"="x_cons_ci",
  "Factors limiting the production - financial constraints"="x_factors_fin",
  "Economic sentiment indicator"="x_esi",
  "Balance for values/ratio for indices"="x_bal_rat",
  "Currency and deposits"="x_cur_dep"
  
  
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], monthly_data_EU[[var]])
}



a1=x_op_rate
b1=x_esi
b2=x_bal_rat
b3=x_hs_pr_ind
b4=x_factors_fin
b5=x_long_loans
b6=x_Debt_sec
b7=x_ret_ci
b8=x_Emp_Exp_ind
b9=x_gross
b10=x_factors_other
b11=x_cur_dep


beta0 <- midas_r(x_unemp ~ 
                   mls(x_unemp, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(x_unemp=x_unemp,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:237, outsample = 238:297,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



unemp_eu_residuals=beta0$residuals
unemp_eu_residuals_1=data.frame(residuals=unemp_eu_residuals)


rmse_value <- rmse(x_unemp[238:297],avgf$forecast )
mae_value <- mae(x_unemp[238:297],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_eu_unemp=data.frame(Actual=x_unemp[238:297],Predicted=avgf$forecast)


unemp_eu_fitted=beta0$residuals

plot(unemp_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(unemp_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(unemp_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(unemp_eu_residuals, main = "Q-Q Plot of MIDAS EU Unemployment Residuals")
qqline(unemp_eu_residuals, col = "red")

shapiro_test <- shapiro.test(unemp_eu_residuals)
print(shapiro_test)

plot(unemp_eu_fitted, unemp_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(unemp_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((x_unemp[238:297] - mean(x_unemp))^2)
ss_residual <- sum((act_vs_pred_eu_unemp$Actual - act_vs_pred_eu_unemp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



###########################################################################################


#####US#####

####US GDP####

set.seed(123)
quarterly_data_US=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Quarterly-AllVars+Target-USA.xlsx")
monthly_df_us=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Monthly.xlsx")

# Define short names mapping
short_names_us <- c(
  "All sectors; checkable deposits and currency; asset" = "us_checkable_deposits",
  "All sectors; U.S. wealth" = "us_wealth",
  "All sectors; corporate and foreign bonds; asset" = "us_corp_bonds",
  "All sectors; total interbank transactions; asset" = "us_interbank_transactions",
  "All sectors; Treasury securities; asset" = "us_treasury_sec",
  "All sectors; total loans; liability" = "us_total_loans",
  "All sectors; total capital expenditures" = "us_cap_expenditures",
  "All sectors; U.S. wealth .1" = "us_wealth_adj",
  "All sectors; long-term debt securities issued by residents; asset" = "us_long_term_debt",
  "All sectors; other loans and advances; liability" = "us_other_loans"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us)) {
  assign(short_names_us[[var]], quarterly_data_US[[var]])
}


short_names_us_monthly <- c(
  "enplane_d11" = "us_enplane",
  "Mining  (NAICS = 21); s.a. CAPUTL" = "us_mining_cap",
  "Total consumer credit owned by credit unions, not seasonally adjusted flow, monthly rate" = "us_credit_union_flow",
  "Nonindustrial supplies; s.a. IP" = "us_nonind_supplies",
  "Finished processing (capacity); s.a. CAPUTL" = "us_finished_processing",
  "Percent change of total owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_finance_receivables_pct_change",
  "Total consumer credit owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_finance_credit_flow",
  "Revolving consumer credit owned by credit unions, not seasonally adjusted flow, monthly rate" = "us_revolving_credit_union",
  "Total consumer credit owned by federal government, not seasonally adjusted flow, monthly rate" = "us_federal_credit_flow",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_consumer_credit_pct_change"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_monthly)) {
  assign(short_names_us_monthly[[var]], monthly_df_us[[var]])
}

a1=us_enplane
#a2=us_mining_cap
a3=us_credit_union_flow
a4=us_nonind_supplies
a5=us_finished_processing
a6=us_finance_receivables_pct_change
a7=us_finance_credit_flow
a8=us_revolving_credit_union
#a9=us_federal_credit_flow
#a10=us_consumer_credit_pct_change
b1=us_checkable_deposits
b2=us_wealth
b3=us_corp_bonds
b4=us_interbank_transactions
b5=us_treasury_sec
b6=us_total_loans
b7=us_cap_expenditures
b8=us_wealth_adj
b9=us_long_term_debt
us_gdp=quarterly_data_US$GDP


beta0 <- midas_r(us_gdp ~ 
                   mls(us_gdp, 1, 1) + 
                   mls(a1, 0:2, 3, nealmon) +
                   #mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   mls(a6, 0:2, 3, nealmon) +
                   mls(a7, 0:2, 3, nealmon) +
                   mls(a8, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9
                 ,
                 start = list(a1 = c(-3, -3),
                              #a2 = c(0.3, -0.3),
                              a3 = c(0.1, -0.1),
                              a4 = c(-0.4, -0.3),
                              a5 = c(0.15, -0.15),
                              a6 = c(0.3, -0.3),
                              a7 = c(-0.1, -0.1),
                              a8 = c(0.3, -0.3)
                              
                 ))


avgf <- average_forecast(list(beta0), data = list(us_gdp=us_gdp,a1=a1,a3=a3,a4=a4,
                                                  a5=a5,a6=a6,a7=a7,a8=a8,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9), insample = 1:79, outsample = 80:99,
                         type = "fixed",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))

gdp_us_residuals=beta0$residuals
gdp_us_residuals_1=data.frame(residuals=gdp_us_residuals)



rmse_value <- rmse(us_gdp[80:99],avgf$forecast )
mae_value <- mae(us_gdp[80:99],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))

act_vs_pred_us_gdp=data.frame(Actual=us_gdp[80:99],Predicted=avgf$forecast)

gdp_us_fitted=beta0$residuals

plot(gdp_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(gdp_us_residuals, main = "ACF of MIDAS Residuals")
pacf(gdp_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(gdp_us_residuals, main = "Q-Q Plot of MIDAS US GDP Residuals")
qqline(gdp_us_residuals, col = "red")

shapiro_test <- shapiro.test(gdp_us_residuals)
print(shapiro_test)

plot(gdp_us_fitted, gdp_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(gdp_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((us_gdp[80:99] - mean(us_gdp))^2)
ss_residual <- sum((act_vs_pred_us_gdp$Actual - act_vs_pred_us_gdp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample

#####US PCE####

daily_df_us=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Daily.xlsx")
monthly_data_US=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Monthly-ARIMA-USA-AllVars+Targets.xlsx")

daily_df_us_2 = daily_df_us[1:6237,]
us_asd=daily_df_us_2$`Australian Dollar`

short_names_us_monthly <- c(
  "All sectors; total interbank transactions; asset" = "us_interbank_transactions",
  "Total borrowings from the Federal Reserve; not seasonally adjusted" = "us_fed_reserve_borrowings",
  "Total securitized consumer credit, not seasonally adjusted flow, monthly rate" = "us_securitized_cons_credit",
  "tsi_passenger" = "us_tsi_passenger",
  "idx_waterborne_d11" = "us_waterborne_idx",
  "M2; Not seasonally adjusted" = "us_m2",
  "Percent change of total consumer owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_cons_fin_receivables_pct_change",
  "Total consumer credit owned and securitized, not seasonally adjusted flow, monthly rate" = "us_cons_credit_flow",
  "Consumer motor vehicle leases owned and securitized by finance companies, not seasonally adjusted flow, monthly rate" = "us_motor_vehicle_leases",
  "All sectors; commercial mortgages; asset" = "us_commercial_mortgages",
  "Materials; s.a. IP" = "us_materials_ip",
  "manuf" = "us_manufacturing",
  "Electric and gas utilities  (NAICS = 2211,2); s.a. IP" = "us_elec_gas_utilities",
  "Business equipment leases owned and securitized by finance companies, not seasonally adjusted flow, monthly rate" = "us_business_eq_leases",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_cons_credit_pct_change",
  "All sectors; U.S. wealth .1" = "us_wealth_adj",
  "Australian Dollar" = "us_aud",
  "M1; Not seasonally adjusted" = "us_m1",
  "All sectors; other loans and advances; liability" = "us_other_loans",
  "Percent change of total revolving consumer credit, seasonally adjusted at an annual rate" = "us_revolving_credit_pct_change"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_monthly)) {
  assign(short_names_us_monthly[[var]], monthly_data_US[[var]])
}



us_pce=monthly_data_US$PCE
a1=us_asd
b1=us_fed_reserve_borrowings
b2=us_revolving_credit_pct_change
b3=us_cons_fin_receivables_pct_change
b4=us_securitized_cons_credit
b5=us_materials_ip
b6=us_manufacturing
b7=us_elec_gas_utilities
b8=us_interbank_transactions
b9=us_wealth_adj
b10=us_cons_credit_flow
b11=us_tsi_passenger


beta0 <- midas_r(us_pce ~ 
                   mls(us_pce, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(us_pce=us_pce,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:237, outsample = 238:297,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



pce_us_residuals=beta0$residuals
pce_us_residuals_1=data.frame(residuals=pce_us_residuals)



rmse_value <- rmse(us_pce[238:297],avgf$forecast )
mae_value <- mae(us_pce[238:297],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_us_pce=data.frame(Actual=us_pce[238:297],Predicted=avgf$forecast)


pce_us_fitted=beta0$residuals

plot(pce_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(pce_us_residuals, main = "ACF of MIDAS Residuals")
pacf(pce_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(pce_us_residuals, main = "Q-Q Plot of MIDAS US PCE Residuals")
qqline(pce_us_residuals, col = "red")

shapiro_test <- shapiro.test(pce_us_residuals)
print(shapiro_test)

plot(pce_us_fitted, pce_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(pce_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)

ss_total <- sum((us_pce[238:297] - mean(us_pce))^2)
ss_residual <- sum((act_vs_pred_us_pce$Actual - act_vs_pred_us_pce$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



#####US UNEMP####


short_names_us_selected <- c(
  "Total index; s.a. IP" = "us_total_index_ip",
  "Consumer goods; s.a. IP" = "us_consumer_goods_ip",
  "Percent change of total revolving consumer credit, seasonally adjusted at an annual rate" = "us_revolving_credit_pct_change",
  "Mining  (NAICS = 21); s.a. IP" = "us_mining_ip",
  "enplane_d11" = "us_enplane",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_cons_credit_pct_change",
  "Small-denomination time deposits - Total; Not seasonally adjusted" = "us_small_time_deposits",
  "Percent change of total business owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_business_receivables_pct_change",
  "Total consumer credit owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_cons_credit_flow",
  "Nonindustrial supplies; s.a. IP" = "us_nonind_supplies",
  "Total consumer credit owned and securitized, seasonally adjusted flow, monthly rate" = "us_cons_credit_seasonal_flow",
  "manuf" = "us_manufacturing",
  "All sectors; Treasury securities; asset" = "us_treasury_sec",
  "idx_rail_frt_carloads" = "us_rail_freight",
  "Business equipment; s.a. IP" = "us_business_equip_ip",
  "Consumer motor vehicle loans owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_motor_vehicle_loans",
  "Percent change of total owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_total_receivables_pct_change",
  "Electric and gas utilities  (NAICS = 2211,2); s.a. IP" = "us_elec_gas_utilities",
  "Manufacturing (SIC); s.a. IP" = "us_manufacturing_sic",
  "Australian Dollar" = "us_aud"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_selected)) {
  assign(short_names_us_selected[[var]], monthly_data_US[[var]])
}

us_unemp=monthly_data_US$UNEMP
a1=us_asd
b1=us_manufacturing_sic
b2=us_total_index_ip
b3=us_cons_credit_seasonal_flow
b4=us_treasury_sec
b5=us_mining_ip
b6=us_consumer_goods_ip
b7=us_nonind_supplies
#b8=us_rail_freight
b9=us_motor_vehicle_loans
b10=us_enplane
b11=us_small_time_deposits


beta0 <- midas_r(us_unemp ~ 
                   mls(us_unemp, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(us_unemp=us_unemp,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:237, outsample = 238:297,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



unemp_us_residuals=beta0$residuals

unemp_us_residuals_1=data.frame(residuals=unemp_us_residuals)


rmse_value <- rmse(us_unemp[238:297],avgf$forecast )
mae_value <- mae(us_unemp[238:297],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_us_unemp=data.frame(Actual=us_unemp[238:297],Predicted=avgf$forecast)

unemp_us_fitted=beta0$residuals

plot(unemp_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(unemp_us_residuals, main = "ACF of MIDAS Residuals")
pacf(unemp_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(unemp_us_residuals, main = "Q-Q Plot of MIDAS US Unemployment Residuals")
qqline(unemp_us_residuals, col = "red")

shapiro_test <- shapiro.test(unemp_us_residuals)
print(shapiro_test)

plot(unemp_us_fitted, unemp_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(unemp_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)



ss_total <- sum((us_unemp[238:297] - mean(us_unemp))^2)
ss_residual <- sum((act_vs_pred_us_unemp$Actual - act_vs_pred_us_unemp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample

write_xlsx(unemp_us_residuals_1, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/US UNEMP MIDAS Residuals.xlsx")
write_xlsx(act_vs_pred_us_unemp, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Pred vs actual US UNEMP MIDAS.xlsx")


install.packages("writexl")
library(writexl)

# Create a list of dataframes
df_list <- list(
  "US_unemp" = act_vs_pred_us_unemp,
  "US_pce" = act_vs_pred_us_pce,
  "US_gdp" = act_vs_pred_us_gdp,
  "EU_unemp" = act_vs_pred_eu_unemp,
  "EU_pce" = act_vs_pred_eu_pce_2,
  "EU_gdp" = act_vs_pred_eu_gdp
)

# Write to Excel file
write_xlsx(df_list, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/MIDAS_Actual_vs_pred.xlsx")


df_list <- list(
  "US_unemp" = unemp_us_residuals_1,
  "US_pce" = pce_us_residuals_1,
  "US_gdp" = gdp_us_residuals_1,
  "EU_unemp" = unemp_eu_residuals_1,
  "EU_pce" = pce_eu_residuals_1,
  "EU_gdp" = gdp_eu_residuals_1
)

# Write to Excel file
write_xlsx(df_list, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/MIDAS_Residuals.xlsx")


#####DM TEST#####
install.packages("multDM")

library(readxl)
library(multDM)


regions <- list("EU GDP", "EU PCE", "EU UNEMP", "US GDP", "US PCE", "US UNEMP")
data_list <- lapply(regions, function(sheet) read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Predictions/Combined predictions frame.xlsx", sheet = sheet))

# Assign names to each dataframe dynamically
names(data_list) <- regions


calculate_errors <- function(df) {
  df$Error_ARIMAX <- df$Actual - df$ARIMAX
  df$Error_MIDAS <- df$Actual - df$MIDAS
  df$Error_LSTM <- df$Actual - df$LSTM
  df$Error_ARIMA_LSTM <- df$Actual - df$`ARIMAX-LSTM`
  df$Error_MIDAS_LSTM <- df$Actual - df$`MIDAS-LSTM`
  return(df)
}

# Apply function to all regional data
data_list <- lapply(data_list, calculate_errors)



library(forecast)
perform_dm_test <- function(errors1, errors2) {
  dm_test_result <- dm.test(errors1, errors2, alternative = "two.sided", h = 1, power=1 )
  return(dm_test_result$p.value)  # Extract p-value for significance interpretation
}

# Loop through each region and perform DM test for each model pair
dm_results <- list()

for (region in regions) {
  df <- data_list[[region]]
  
  # Apply DM test for various model pairs
  dm_results[[region]] <- list(
    "ARIMAX vs MIDAS" = perform_dm_test(df$Error_ARIMAX, df$Error_MIDAS),
    "ARIMAX vs LSTM" = perform_dm_test(df$Error_ARIMAX, df$Error_LSTM),
    "ARIMAX vs ARIMAX-LSTM" = perform_dm_test(df$Error_ARIMAX, df$Error_ARIMA_LSTM),
    "ARIMAX vs MIDAS-LSTM" = perform_dm_test(df$Error_ARIMAX, df$Error_MIDAS_LSTM),
    "MIDAS vs LSTM" = perform_dm_test(df$Error_MIDAS, df$Error_LSTM),
    "MIDAS vs ARIMAX-LSTM" = perform_dm_test(df$Error_MIDAS, df$Error_ARIMA_LSTM),
    "MIDAS vs MIDAS-LSTM" = perform_dm_test(df$Error_MIDAS, df$Error_MIDAS_LSTM),
    "LSTM vs ARIMAX-LSTM" = perform_dm_test(df$Error_LSTM, df$Error_ARIMA_LSTM),
    "LSTM vs MIDAS-LSTM" = perform_dm_test(df$Error_LSTM, df$Error_MIDAS_LSTM),
    "ARIMAX-LSTM vs MIDAS-LSTM" = perform_dm_test(df$Error_ARIMA_LSTM, df$Error_MIDAS_LSTM)
  )
}

# Display results
dm_results


dm_results_df <- do.call(rbind, lapply(names(dm_results), function(region) {
  data.frame(Region = region, Model_Comparison = names(dm_results[[region]]), P_Value = unlist(dm_results[[region]]))
}))


write_xlsx(dm_results_df, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Results/DM_Test_Results_power_1.xlsx")


models <- c("ARIMAX", "MIDAS", "LSTM", "ARIMAX-LSTM", "MIDAS-LSTM")

# Function to create 5x5 DM matrix for each region
create_dm_matrix <- function(df) {
  dm_matrix <- matrix(NA, nrow = length(models), ncol = length(models), dimnames = list(models, models))
  
  for (i in 1:length(models)) {
    for (j in 1:length(models)) {
      if (i != j) {  # Leave diagonal blank
        dm_matrix[i, j] <- perform_dm_test(df[[paste0("Error_", models[i])]], df[[paste0("Error_", models[j])]])
      }
    }
  }
  
  return(as.data.frame(dm_matrix))
}


dm_results_list <- list()
for (region in regions) {
  dm_results_list[[region]] <- create_dm_matrix(data_list[[region]])
}


library(Metrics)  # Load library for RMSE and MAE calculations

calculate_errors_and_metrics <- function(df) {
  df$Error_ARIMAX <- df$Actual - df$ARIMAX
  df$Error_MIDAS <- df$Actual - df$MIDAS
  df$Error_LSTM <- df$Actual - df$LSTM
  df$Error_ARIMA_LSTM <- df$Actual - df$`ARIMAX-LSTM`
  df$Error_MIDAS_LSTM <- df$Actual - df$`MIDAS-LSTM`
  
  # Calculate RMSE and MAE for LSTM predictions
  rmse_lstm <- rmse(df$Actual, df$LSTM)
  mae_lstm <- mae(df$Actual, df$LSTM)
  
  return(list(df = df, rmse_lstm = rmse_lstm, mae_lstm = mae_lstm))
}

# Apply function to all regional data
results_list <- lapply(data_list, calculate_errors_and_metrics)

# Extract RMSE and MAE results for LSTM predictions
rmse_mae_lstm <- data.frame(
  Region = names(results_list),
  RMSE_LSTM = sapply(results_list, function(x) x$rmse_lstm),
  MAE_LSTM = sapply(results_list, function(x) x$mae_lstm)
)

# Display RMSE and MAE results
print(rmse_mae_lstm)

##################################################################################

####rmse, mae####
# Load required libraries
library(readxl)
library(dplyr)
library(purrr)
library(writexl)

# Define the file path
file_path <- "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Predictions/Without Covid Results.xlsx"
#regions <- list("EU GDP", "EU PCE", "EU UNEMP", "US GDP", "US PCE", "US UNEMP")
#data_list <- lapply(regions, function(sheet) read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Predictions/Without Covid Results.xlsx", sheet = sheet))

# Get sheet names
sheet_names <- excel_sheets(file_path)

# Function to compute RMSE and MAE
compute_metrics <- function(df) {
  actual_values <- df[[2]]  # Assuming the second column contains actual values
  
  # Compute RMSE and MAE for each model (columns 3 onward)
  metrics <- map_dfr(df[3:ncol(df)], function(predicted_values) {
    rmse <- sqrt(mean((actual_values - predicted_values)^2, na.rm = TRUE))
    mae <- mean(abs(actual_values - predicted_values), na.rm = TRUE)
    data.frame(RMSE = rmse, MAE = mae)
  }, .id = "Model")
  
  return(metrics)
}

# Process each sheet and store results
results <- map(sheet_names, ~ {
  df <- read_excel(file_path, sheet = .x)
  metrics <- compute_metrics(df)
  metrics$Region <- .x  # Add region name
  metrics
})

# Combine results into a single data frame
final_results <- bind_rows(results)

# Reshape results into separate tables for RMSE and MAE
rmse_table <- final_results %>%
  select(Region, Model, RMSE) %>%
  spread(Model, RMSE)

mae_table <- final_results %>%
  select(Region, Model, MAE) %>%
  spread(Model, MAE)

# Save the results to an Excel file
write_xlsx(list(RMSE = rmse_table, MAE = mae_table), "Model_Evaluation.xlsx")

# Print results
print("RMSE Table:")
print(rmse_table)

print("MAE Table:")
print(mae_table)


data=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/2000-2025-Quarterly-AllVars+Target-USA_copy.xlsx")
library(ggplot2)
library(gridExtra)
install.packages("ggplot2")
install.packages("gridExtra")
# Load dataset
data <- read.csv("your_data.csv")  # Replace with actual file

# Automatically detect the dependent variable (assumed to be the first column)
dependent_var <- colnames(data)[1]  # Change index if needed

# Detect exogenous variables (all columns except the first one)
exogenous_vars <- colnames(data)[-1]

# Generate scatter plots
plots <- lapply(exogenous_vars, function(var) {
  ggplot(data, aes_string(x = var, y = dependent_var)) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    labs(title = paste("Scatter Plot of", dependent_var, "vs", var),
         x = var, y = dependent_var) +
    theme_minimal()
})

# Arrange plots in a grid (2 columns for better visibility)
grid.arrange(grobs = plots, ncol = 2)

#par(mfrow = c(ceiling(length(exogenous_vars) / 2), 2))  # 2 plots per row

# Generate scatter plots using base R
for (var in exogenous_vars) {
  dev.new()  # Opens a new plotting window for each plot
  plot(data[[var]], data[[dependent_var]], 
       xlab = var, ylab = dependent_var, 
       main = paste(dependent_var, "vs", var),
       col = "blue", pch = 19)
  abline(lm(data[[dependent_var]] ~ data[[var]]), col = "red")  # Regression line
}






##########################MIDAS PREDICTIONS WITHOUT COVID #################

library(readxl)
library(midasr)
library(dplyr)

library(Metrics)  # For RMSE, MAE calculations



####EU####

quarterly_data_EU=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/Europe/2000-2025-Quarterly-ARIMA-Europe-Covid.xlsx")
monthly_data_EU=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/Europe/2000-2025-Monthly-ARIMA-EU-Covid.xlsx")

set.seed(123)
######EU GDP#####


short_names <- c(
  "Assessment of current production capacity" = "x_prod_capacity",
  "Competitive position on foreign markets outside the EU over the past three months" = "x_comp_outside_EU",
  "Short-term - loans" = "x_short_loans",
  "Real effective exchange rate (deflator: consumer price indices - 37 trading partners)" = "x_reer_37",
  "Factors limiting the production - none" = "x_factors_none",
  "HICP - Education" = "x_hicp_edu",
  "HICP - Transport" = "x_hicp_transport",
  "HICP - Health" = "x_hicp_health",
  "HICP - Housing, water, electricity,gas and other fuels" = "x_hicp_housing",
  "HICP - Clothing and footwear" = "x_hicp_clothing",
  "Long-term debt securities" = "x_long_debt_sec",
  "Loans" = "x_loans",
  "Long-term - loans" = "x_long_loans",
  "Government consolidated gross debt" = "x_govt_debt",
  "Factors limiting the production - financial constraints" = "x_factors_fin",
  "House price index" = "x_house_price",
  "Factors limiting the production - equipment" = "x_factors_equip",
  "Factors limiting the production - other" = "x_factors_other",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "New orders in recent months" = "x_new_orders"
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], quarterly_data_EU[[var]])
}


x_ici=monthly_data_EU$`Construction confidence indicator`
x_ee=monthly_data_EU$`Real effective exchange rate (deflator: consumer price indices - 37 trading partners)`
x_ci=monthly_data_EU$`HICP - Housing, water, electricity,gas and other fuels`
x_ei=monthly_data_EU$`HICP - Transport`
x_reer=monthly_data_EU$`HICP - Clothing and footwear`
#x_exp=monthly_data_EU$`Money market interest rate`
#x_eer=monthly_data_EU$`Euro Exchange rate`
x_gdp=quarterly_data_EU$GDP



a1=x_ici
a2=x_ee
a3=x_ci
a4=x_ei
a5=x_reer
b1=x_prod_capacity
b2=x_house_price
b3=x_factors_equip
b4=x_long_loans
b5=x_long_debt_sec
b6=x_comp_outside_EU
b7=x_short_loans
b8=x_govt_debt



beta0 <- midas_r(x_gdp ~ 
                   mls(x_gdp, 1, 1) + 
                   #mls(a1, 0:2, 3, nealmon) +
                   mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8
                 ,
                 start = list(
                   #   a1 = c(0.5,-0.5),  
                   a2 = c(0.5, -0.5),
                   a3 = c(0.5, -0.5),
                   a4 = c(0.5, -0.5),
                   a5 = c(0.5, -0.5)
                   
                 ))






avgf <- average_forecast(list(beta0), data = list(x_gdp=x_gdp,a2=a2,a3=a3,a4=a4,
                                                  a5=a5,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8), insample = 1:72, outsample = 73:90,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



gdp_eu_residuals=beta0$residuals
gdp_eu_residuals_1=data.frame(residuals=gdp_eu_residuals)

gdp_eu_fitted=beta0$fitted.values

rmse_value <- rmse(x_gdp[73:90],avgf$forecast )
mae_value <- mae(x_gdp[73:90],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))

act_vs_pred_eu_gdp=data.frame(Actual=x_gdp[73:90],Predicted=avgf$forecast)


plot(gdp_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(gdp_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(gdp_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(gdp_eu_residuals, main = "Q-Q Plot of MIDAS EU GDP Residuals")
qqline(gdp_eu_residuals, col = "red")

shapiro_test <- shapiro.test(gdp_eu_residuals)
print(shapiro_test)

plot(gdp_eu_fitted, gdp_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(gdp_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((x_gdp[73:90] - mean(x_gdp))^2)
ss_residual <- sum((act_vs_pred_eu_gdp$Actual - act_vs_pred_eu_gdp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



######################################################################################

set.seed(4444)
######EU PCE#####

short_names <- c(
  "Assessment of current production capacity" = "x_prod_capacity",
  "Factors limiting the production - none" = "x_factors_none",
  "Competitive position on foreign markets outside the EU over the past three months" = "x_comp_outside_EU",
  "Goods" = "x_goods",
  "Short-term - loans" = "x_short_loans",
  "Balance for values/ratio for indices" = "x_balance_ratio",
  "Goods and services" = "x_goods_services",
  "Services" = "x_services",
  "HICP - Education" = "x_hicp_edu",
  "HICP - Transport" = "x_hicp_transport",
  "Long-term debt securities" = "x_long_debt_sec",
  "Short-term debt securities" = "x_short_debt_sec",
  "Debt securities" = "x_debt_sec",
  "yeild Curve" = "x_yield_curve",
  "Factors limiting the production - labour" = "x_factors_labour",
  "Factors limiting the production - other" = "x_factors_other",
  "Capital account" = "x_capital_acc",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Factors limiting the production - equipment" = "x_factors_equip",
  "New orders in recent months" = "x_new_orders"
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], quarterly_data_EU[[var]])
}


x_ici=monthly_data_EU$`Industrial confidence indicator`

x_ee=monthly_data_EU$`HICP - Transport`
x_ci=monthly_data_EU$`HICP - Education`
#x_ei=monthly_data_EU$`Balance for values/ratio for indices`
x_reer=monthly_data_EU$`yeild Curve`
#x_exp=monthly_data_EU$
#x_eer=monthly_data_EU$`Real effective exchange rate (deflator: consumer price indices - 37 trading partners)`
x_pce=quarterly_data_EU$PCE



a1=x_ici
a2=x_ee
a3=x_ci
#a4=x_ei
a5=x_reer
b1=x_comp_outside_EU
b2=x_goods_services
b3=x_factors_labour
b4=x_new_orders
b5=x_capital_acc
b6=x_debt_sec
b7=x_prod_capacity
b8=x_factors_equip





beta0 <- midas_r(x_pce ~ 
                   mls(x_pce, 1, 1) + 
                   #mls(a1, 0:2, 3, nealmon) +
                   mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   #mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8
                 ,
                 start = list(
                   #  a1 = c(0.5,-0.5),  
                   a2 = c(0.5, -0.5),
                   a3 = c(0.5, -0.5),
                   #a4 = c(0.6, -0.5),
                   a5 = c(0.5, -0.5)
                   
                 ))






avgf <- average_forecast(list(beta0), data = list(x_pce=x_pce,a2=a2,a3=a3,
                                                  a5=a5,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8), insample = 1:72, outsample = 73:90,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



pce_eu_residuals=beta0$residuals
pce_eu_residuals_1=data.frame(residuals=pce_eu_residuals)

pce_eu_fitted=beta0$fitted.values

rmse_value <- rmse(x_pce[73:90],avgf$forecast )
mae_value <- mae(x_pce[73:90],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_eu_pce_2=data.frame(Actual=x_pce[73:90],Predicted=avgf$forecast)


plot(pce_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(pce_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(pce_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(pce_eu_residuals, main = "Q-Q Plot of MIDAS EU PCE Residuals")
qqline(pce_eu_residuals, col = "red")

shapiro_test <- shapiro.test(pce_eu_residuals)
print(shapiro_test)

plot(pce_eu_fitted, pce_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(pce_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)



ss_total <- sum((x_pce[73:90] - mean(x_pce))^2)
ss_residual <- sum((act_vs_pred_eu_pce_2$Actual - act_vs_pred_eu_pce_2$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample




########################################################################################



#####EU UNEMP####
daily_data=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Europe/Daily-EU-ARIMA_covid.xlsx")

daily_data_2 = daily_data[1:5670,]

x_unemp=monthly_data_EU$UNEMP
x_op_rate=daily_data_2$`Main Refinancing Operations Rate`






short_names <- c(
  "Economic sentiment indicator" = "x_esi",
  "Construction confidence indicator" = "x_construction_ci",
  "Balance for values/ratio for indices" = "x_bal_rat",
  "New orders in recent months" = "x_new_orders",
  "Retail confidence indicator" = "x_ret_ci",
  "Factors limiting the production - other" = "x_factors_other",
  "Factors limiting the production - financial constraints" = "x_factors_fin",
  "Debt securities" = "x_debt_sec",
  "HICP - All items (HICP=harmonized index of consumer prices)" = "x_hicp_all",
  "Industrial confidence indicator" = "x_ind_ci",
  "Consumer confidence indicator" = "x_cons_ci",
  "Real effective exchange rate (deflator: consumer price indices - 42 trading partners)" = "x_reer_42",
  "Currency and deposits" = "x_cur_dep",
  "Employment expectation index in next three months" = "x_emp_exp_3m",
  "Current level of capacity utilization (%)" = "x_capacity_util",
  "Home improvements over the next 12 months" = "x_home_improve",
  "Services confidence indicator" = "x_services_ci",
  "House price index" = "x_hs_pr_ind",
  "Purchase or build a home within the next 12 months" = "x_home_purchase",
  "Money market interest rate" = "x_money_market_rate"
)

# Loop through the mapping and assign short names
for (var in names(short_names)) {
  assign(short_names[[var]], monthly_data_EU[[var]])
}


a1=x_op_rate
b1=x_esi
b2=x_ind_ci
b3=x_hs_pr_ind
b4=x_factors_fin
b5=x_debt_sec
b6=x_emp_exp_3m
b7=x_ret_ci
b8=x_money_market_rate
b9=x_bal_rat
b10=x_factors_other
b11=x_cur_dep


beta0 <- midas_r(x_unemp ~ 
                   mls(x_unemp, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(x_unemp=x_unemp,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:216, outsample = 217:270,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



unemp_eu_residuals=beta0$residuals
unemp_eu_residuals_1=data.frame(residuals=unemp_eu_residuals)


rmse_value <- rmse(x_unemp[217:270],avgf$forecast )
mae_value <- mae(x_unemp[217:270],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_eu_unemp=data.frame(Actual=x_unemp[217:270],Predicted=avgf$forecast)


unemp_eu_fitted=beta0$fitted.values

plot(unemp_eu_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(unemp_eu_residuals, main = "ACF of MIDAS Residuals")
pacf(unemp_eu_residuals, main = "PACF of MIDAS Residuals")


qqnorm(unemp_eu_residuals, main = "Q-Q Plot of MIDAS EU Unemployment Residuals")
qqline(unemp_eu_residuals, col = "red")

shapiro_test <- shapiro.test(unemp_eu_residuals)
print(shapiro_test)

plot(unemp_eu_fitted, unemp_eu_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(unemp_eu_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((x_unemp[238:297] - mean(x_unemp))^2)
ss_residual <- sum((act_vs_pred_eu_unemp$Actual - act_vs_pred_eu_unemp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



###########################################################################################


#####US#####

####US GDP####

set.seed(123)
quarterly_data_US=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/USA/2000-2025-Quarterly-USA-Covid.xlsx")
monthly_df_us=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Monthly_covid.xlsx")

# Define short names mapping
short_names_us <- c(
  "All sectors; checkable deposits and currency; asset" = "us_checkable_deposits",
  "All sectors; U.S. wealth" = "us_wealth",
  "All sectors; corporate and foreign bonds; asset" = "us_corp_bonds",
  "All sectors; total interbank transactions; asset" = "us_interbank_transactions",
  "All sectors; Treasury securities; asset" = "us_treasury_sec",
  "All sectors; total loans; liability" = "us_total_loans",
  "All sectors; total capital expenditures" = "us_cap_expenditures",
  "All sectors; U.S. wealth .1" = "us_wealth_adj",
  "All sectors; long-term debt securities issued by residents; asset" = "us_long_term_debt",
  "All sectors; other loans and advances; liability" = "us_other_loans"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us)) {
  assign(short_names_us[[var]], quarterly_data_US[[var]])
}


short_names_us_monthly <- c(
  "enplane_d11" = "us_enplane",
  "Mining  (NAICS = 21); s.a. CAPUTL" = "us_mining_cap",
  "Total consumer credit owned by credit unions, not seasonally adjusted flow, monthly rate" = "us_credit_union_flow",
  "Nonindustrial supplies; s.a. IP" = "us_nonind_supplies",
  "Finished processing (capacity); s.a. CAPUTL" = "us_finished_processing",
  "Percent change of total owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_finance_receivables_pct_change",
  "Total consumer credit owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_finance_credit_flow",
  "Revolving consumer credit owned by credit unions, not seasonally adjusted flow, monthly rate" = "us_revolving_credit_union",
  "Total consumer credit owned by federal government, not seasonally adjusted flow, monthly rate" = "us_federal_credit_flow",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_consumer_credit_pct_change"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_monthly)) {
  assign(short_names_us_monthly[[var]], monthly_df_us[[var]])
}

#a1=us_enplane
#a2=us_mining_cap
a3=us_credit_union_flow
a4=us_nonind_supplies
a5=us_finished_processing
#a6=us_finance_receivables_pct_change
a7=us_finance_credit_flow
a8=us_revolving_credit_union
#a9=us_federal_credit_flow
#a10=us_consumer_credit_pct_change
b1=us_checkable_deposits
b2=us_wealth
b3=us_corp_bonds
b4=us_interbank_transactions
b5=us_treasury_sec
b6=us_total_loans
b7=us_cap_expenditures
#b8=us_wealth_adj
b9=us_long_term_debt
us_gdp=quarterly_data_US$GDP


beta0 <- midas_r(us_gdp ~ 
                   mls(us_gdp, 1, 1) + 
                   #mls(a1, 0:2, 3, nealmon) +
                   #mls(a2, 0:2, 3, nealmon) +
                   mls(a3, 0:2, 3, nealmon) +
                   mls(a4, 0:2, 3, nealmon) +
                   mls(a5, 0:2, 3, nealmon) +
                   #mls(a6, 0:2, 3, nealmon) +
                   mls(a7, 0:2, 3, nealmon) +
                   mls(a8, 0:2, 3, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b9
                 ,
                 start = list(#a1 = c(-3, -3),
                   #a2 = c(0.3, -0.3),
                   a3 = c(0.1, -0.1),
                   a4 = c(-0.4, -0.3),
                   a5 = c(0.16, -0.15),
                   #a6 = c(0.3, -0.3),
                   a7 = c(-0.1, -0.1),
                   a8 = c(0.3, -0.3)
                   
                 ))


avgf <- average_forecast(list(beta0), data = list(us_gdp=us_gdp,a4=a4,a3=a3,
                                                  a5=a5,a7=a7,a8=a8,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b9=b9), insample = 1:72, outsample = 73:90,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))

gdp_us_residuals=beta0$residuals
gdp_us_residuals_1=data.frame(residuals=gdp_us_residuals)



rmse_value <- rmse(us_gdp[73:90],avgf$forecast )
mae_value <- mae(us_gdp[73:90],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))

act_vs_pred_us_gdp=data.frame(Actual=us_gdp[73:90],Predicted=avgf$forecast)

gdp_us_fitted=beta0$fitted.values

plot(gdp_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(gdp_us_residuals, main = "ACF of MIDAS Residuals")
pacf(gdp_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(gdp_us_residuals, main = "Q-Q Plot of MIDAS US GDP Residuals")
qqline(gdp_us_residuals, col = "red")

shapiro_test <- shapiro.test(gdp_us_residuals)
print(shapiro_test)

plot(gdp_us_fitted, gdp_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(gdp_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)


ss_total <- sum((us_gdp[80:99] - mean(us_gdp))^2)
ss_residual <- sum((act_vs_pred_us_gdp$Actual - act_vs_pred_us_gdp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample

write_xlsx(gdp_us_residuals_1, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/US GDP MIDAS Residuals WITHOUT COVID.xlsx")
write_xlsx(act_vs_pred_us_gdp, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Pred vs actual US GDP MIDAS WITHOUT COVID.xlsx")


#####US PCE####

daily_df_us=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/USA/Daily_covid.xlsx")
monthly_data_US=read_excel("C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Without covid/USA/2000-2025-Monthly-ARIMA-USA-Covid.xlsx")

daily_df_us_2 = daily_df_us[1:5670,]
us_asd=daily_df_us_2$`Australian Dollar`

short_names_us_monthly <- c(
  "All sectors; total interbank transactions; asset" = "us_interbank_transactions",
  "Total borrowings from the Federal Reserve; not seasonally adjusted" = "us_fed_reserve_borrowings",
  "Total securitized consumer credit, not seasonally adjusted flow, monthly rate" = "us_securitized_cons_credit",
  "tsi_passenger" = "us_tsi_passenger",
  "idx_waterborne_d11" = "us_waterborne_idx",
  "M2; Not seasonally adjusted" = "us_m2",
  "Percent change of total consumer owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_cons_fin_receivables_pct_change",
  "Total consumer credit owned and securitized, not seasonally adjusted flow, monthly rate" = "us_cons_credit_flow",
  "Consumer motor vehicle leases owned and securitized by finance companies, not seasonally adjusted flow, monthly rate" = "us_motor_vehicle_leases",
  "All sectors; commercial mortgages; asset" = "us_commercial_mortgages",
  "Materials; s.a. IP" = "us_materials_ip",
  "manuf" = "us_manufacturing",
  "Electric and gas utilities  (NAICS = 2211,2); s.a. IP" = "us_elec_gas_utilities",
  "Business equipment leases owned and securitized by finance companies, not seasonally adjusted flow, monthly rate" = "us_business_eq_leases",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_cons_credit_pct_change",
  "All sectors; U.S. wealth .1" = "us_wealth_adj",
  "Australian Dollar" = "us_aud",
  "M1; Not seasonally adjusted" = "us_m1",
  "Consumer goods; s.a. IP"='us_cs',
  "All sectors; other loans and advances; liability" = "us_other_loans",
  "Percent change of total revolving consumer credit, seasonally adjusted at an annual rate" = "us_revolving_credit_pct_change"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_monthly)) {
  assign(short_names_us_monthly[[var]], monthly_data_US[[var]])
}



us_pce=monthly_data_US$PCE
a1=us_asd
b1=us_fed_reserve_borrowings
b2=us_revolving_credit_pct_change
b3=us_cons_fin_receivables_pct_change
b4=us_securitized_cons_credit
b5=us_materials_ip
b6=us_manufacturing
b7=us_elec_gas_utilities
b8=us_interbank_transactions
b9=us_wealth_adj
b10=us_cons_credit_flow
b11=us_cs


beta0 <- midas_r(us_pce ~ 
                   mls(us_pce, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b8+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(us_pce=us_pce,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b8=b8,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:216, outsample = 217:270,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



pce_us_residuals=beta0$residuals
pce_us_residuals_1=data.frame(residuals=pce_us_residuals)



rmse_value <- rmse(us_pce[217:270],avgf$forecast )
mae_value <- mae(us_pce[217:270],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_us_pce=data.frame(Actual=us_pce[217:270],Predicted=avgf$forecast)


pce_us_fitted=beta0$fitted.values

plot(pce_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(pce_us_residuals, main = "ACF of MIDAS Residuals")
pacf(pce_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(pce_us_residuals, main = "Q-Q Plot of MIDAS US PCE Residuals")
qqline(pce_us_residuals, col = "red")

shapiro_test <- shapiro.test(pce_us_residuals)
print(shapiro_test)

plot(pce_us_fitted, pce_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(pce_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)

ss_total <- sum((us_pce[238:297] - mean(us_pce))^2)
ss_residual <- sum((act_vs_pred_us_pce$Actual - act_vs_pred_us_pce$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample



#####US UNEMP####

us_cp=daily_df_us_2$CP

short_names_us_selected <- c(
  "Total index; s.a. IP" = "us_total_index_ip",
  "Consumer goods; s.a. IP" = "us_consumer_goods_ip",
  "Percent change of total revolving consumer credit, seasonally adjusted at an annual rate" = "us_revolving_credit_pct_change",
  "Mining  (NAICS = 21); s.a. IP" = "us_mining_ip",
  "enplane_d11" = "us_enplane",
  "All sectors; commercial mortgages; asset"="us_com_morg_Asset",
  "Percent change of total consumer credit, seasonally adjusted at an annual rate" = "us_cons_credit_pct_change",
  "Small-denomination time deposits - Total; Not seasonally adjusted" = "us_small_time_deposits",
  "Percent change of total business owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_business_receivables_pct_change",
  "Total consumer credit owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_cons_credit_flow",
  "Nonindustrial supplies; s.a. IP" = "us_nonind_supplies",
  "Total consumer credit owned and securitized, seasonally adjusted flow, monthly rate" = "us_cons_credit_seasonal_flow",
  "manuf" = "us_manufacturing",
  "All sectors; Treasury securities; asset" = "us_treasury_sec",
  "idx_rail_frt_carloads" = "us_rail_freight",
  "Business equipment; s.a. IP" = "us_business_equip_ip",
  "Business retail motor vehicle loans securitized by finance companies, not seasonally adjusted flow, monthly rate"='us_bus_veh_loan',
  "Consumer motor vehicle loans owned by finance companies, not seasonally adjusted flow, monthly rate" = "us_motor_vehicle_loans",
  "Percent change of total owned and managed receivables outstanding held by finance companies, seasonally adjusted at an annual rate" = "us_total_receivables_pct_change",
  "Electric and gas utilities  (NAICS = 2211,2); s.a. IP" = "us_elec_gas_utilities",
  "Manufacturing (SIC); s.a. IP" = "us_manufacturing_sic",
  "Australian Dollar" = "us_aud"
)

# Loop through the mapping and assign short names
for (var in names(short_names_us_selected)) {
  assign(short_names_us_selected[[var]], monthly_data_US[[var]])
}

us_unemp=monthly_data_US$UNEMP
a1=us_cp
b1=us_manufacturing_sic
b2=us_total_index_ip
b3=us_cons_credit_seasonal_flow
b4=us_treasury_sec
b5=us_bus_veh_loan
b6=us_consumer_goods_ip
b7=us_nonind_supplies
#b8=us_rail_freight
b9=us_motor_vehicle_loans
b10=us_com_morg_Asset
b11=us_small_time_deposits


beta0 <- midas_r(us_unemp ~ 
                   mls(us_unemp, 1, 1) + 
                   mls(a1, 0:10, 21, nealmon) +
                   b1 +
                   b2 +
                   b3+
                   b4+
                   b5+
                   b6+
                   b7+
                   b9+b10+b11
                 ,
                 start = list(
                   a1 = c(0.5,-0.5)
                 ))






avgf <- average_forecast(list(beta0), data = list(us_unemp=us_unemp,a1=a1,
                                                  b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,
                                                  b6=b6,b7=b7,b9=b9,
                                                  b10=b10,b11=b11), 
                         insample = 1:216, outsample = 217:270,
                         type = "rolling",
                         
                         measures = c("MSE", "MAPE", "MASE"),
                         fweights = c("EW", "BICW", "MSFE", "DMSFE"))



unemp_us_residuals=beta0$residuals

unemp_us_residuals_1=data.frame(residuals=unemp_us_residuals)

rmse_value <- rmse(us_unemp[217:270],avgf$forecast )
mae_value <- mae(us_unemp[217:270],avgf$forecast)

# Print RMSE and MAE
print(paste("Rolling Forecast RMSE:", rmse_value))
print(paste("Rolling Forecast MAE:", mae_value))


act_vs_pred_us_unemp=data.frame(Actual=us_unemp[217:270],Predicted=avgf$forecast)

unemp_us_fitted=beta0$residuals

plot(unemp_us_residuals, type = "l", main = "MIDAS Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

acf(unemp_us_residuals, main = "ACF of MIDAS Residuals")
pacf(unemp_us_residuals, main = "PACF of MIDAS Residuals")


qqnorm(unemp_us_residuals, main = "Q-Q Plot of MIDAS US Unemployment Residuals")
qqline(unemp_us_residuals, col = "red")

shapiro_test <- shapiro.test(unemp_us_residuals)
print(shapiro_test)

plot(unemp_us_fitted, unemp_us_residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

ljung_box_test <- Box.test(unemp_us_residuals, lag = 10, type = "Ljung-Box")
print(ljung_box_test)



ss_total <- sum((us_unemp[238:297] - mean(us_unemp))^2)
ss_residual <- sum((act_vs_pred_us_unemp$Actual - act_vs_pred_us_unemp$Predicted)^2)
r2_out_sample <- 1 - (ss_residual / ss_total)
r2_out_sample


install.packages("writexl")
library(writexl)

# Create a list of dataframes
df_list <- list(
  "US_unemp" = act_vs_pred_us_unemp,
  "US_pce" = act_vs_pred_us_pce,
  "US_gdp" = act_vs_pred_us_gdp,
  "EU_unemp" = act_vs_pred_eu_unemp,
  "EU_pce" = act_vs_pred_eu_pce_2,
  "EU_gdp" = act_vs_pred_eu_gdp
)

# Write to Excel file
write_xlsx(df_list, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/MIDAS_Actual_vs_pred_Covid.xlsx")


df_list <- list(
  "US_unemp" = unemp_us_residuals_1,
  "US_pce" = pce_us_residuals_1,
  "US_gdp" = gdp_us_residuals_1,
  "EU_unemp" = unemp_eu_residuals_1,
  "EU_pce" = pce_eu_residuals_1,
  "EU_gdp" = gdp_eu_residuals_1
)

# Write to Excel file
write_xlsx(df_list, "C:/Users/Bhargavi/Documents/VLK Case study/Final Data/Residuals/MIDAS_Residuals_covid.xlsx")
