# HOLT WINTERS METHOD
 

# AIM:
To implement the Holt Winters Method Model using Python.

# ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

# PROGRAM:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

file_path = "results.csv"
data = pd.read_csv(file_path, parse_dates=["date"])

data['home_score'] = pd.to_numeric(data['home_score'], errors='coerce')
data = data.dropna(subset=['home_score'])

data.set_index('date', inplace=True)

monthly_data = data['home_score'].resample('MS').mean()
monthly_data = monthly_data.dropna()

split_index = int(0.9 * len(monthly_data))
train_data = monthly_data[:split_index]
test_data = monthly_data[split_index:]

if len(train_data) >= 24:
    seasonal_periods = 12
    seasonal_type = 'add'
    print("Using seasonal model (12-month period).")
else:
    seasonal_periods = None
    seasonal_type = None
    print("Using non-seasonal model (data too short for seasonality).")

fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=seasonal_type,
    seasonal_periods=seasonal_periods
).fit()

test_predictions = fitted_model.forecast(len(test_data))

plt.figure(figsize=(12, 8))
train_data.plot(label='Train')
test_data.plot(label='Test')
test_predictions.plot(label='Predicted', color='red', linestyle='dashed')
plt.title('Train, Test, and Predicted using Holt-Winters')
plt.xlabel('Date')
plt.ylabel('Average Home Score')
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

final_model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal=seasonal_type,
    seasonal_periods=seasonal_periods
).fit()

forecast_predictions = final_model.forecast(steps=12)

plt.figure(figsize=(12, 8))
monthly_data.plot(label='Original Data')
forecast_predictions.plot(label='Forecasted Data', color='purple', linestyle='dashed')
plt.title('Original and Forecasted Home Scores')
plt.xlabel('Date')
plt.ylabel('Average Home Score')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:

## TEST_PREDICTION

<img width="1265" height="761" alt="image" src="https://github.com/user-attachments/assets/c5e4744a-2999-4058-9836-15b934f6f8bf" />

### FINAL_PREDICTION

<img width="1285" height="767" alt="image" src="https://github.com/user-attachments/assets/c63d387a-3674-4b95-bed0-ce769d532c8f" />


# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
