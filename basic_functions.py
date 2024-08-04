# Time Series Forecasting Snippets for Finance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

# 1. Data Preparation
def load_and_prepare_data(file_path):
    """
    Load financial time series data and prepare it for forecasting.
    
    Args:
    file_path (str): Path to the CSV file containing date and price columns.
    
    Returns:
    pd.DataFrame: Prepared time series data with 'ds' (date) and 'y' (price) columns.
    """
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['price']
    return df[['ds', 'y']]

# Example usage:
# df = load_and_prepare_data('stock_prices.csv')

# 2. ARIMA Model
def fit_arima(data, order=(1,1,1)):
    """
    Fit an ARIMA model to the time series data.
    
    Args:
    data (pd.Series): Time series data.
    order (tuple): ARIMA model order (p,d,q).
    
    Returns:
    ARIMA: Fitted ARIMA model.
    """
    model = ARIMA(data, order=order)
    return model.fit()

def forecast_arima(model, steps):
    """
    Generate forecasts using a fitted ARIMA model.
    
    Args:
    model (ARIMA): Fitted ARIMA model.
    steps (int): Number of steps to forecast.
    
    Returns:
    pd.Series: Forecasted values.
    """
    return model.forecast(steps=steps)

# Example usage:
# arima_model = fit_arima(df['y'])
# arima_forecast = forecast_arima(arima_model, steps=30)

# 3. Prophet Model
def fit_prophet(data):
    """
    Fit a Prophet model to the time series data.
    
    Args:
    data (pd.DataFrame): DataFrame with 'ds' and 'y' columns.
    
    Returns:
    Prophet: Fitted Prophet model.
    """
    model = Prophet()
    return model.fit(data)

def forecast_prophet(model, periods):
    """
    Generate forecasts using a fitted Prophet model.
    
    Args:
    model (Prophet): Fitted Prophet model.
    periods (int): Number of periods to forecast.
    
    Returns:
    pd.DataFrame: Forecasted values.
    """
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)

# Example usage:
# prophet_model = fit_prophet(df)
# prophet_forecast = forecast_prophet(prophet_model, periods=30)

# 4. Auto ARIMA
def fit_auto_arima(data):
    """
    Automatically find the best ARIMA model using auto_arima.
    
    Args:
    data (pd.Series): Time series data.
    
    Returns:
    ARIMA: Best fitted ARIMA model.
    """
    return auto_arima(data, seasonal=False, stepwise=True)

# Example usage:
# auto_arima_model = fit_auto_arima(df['y'])
# auto_arima_forecast = auto_arima_model.predict(n_periods=30)

# 5. Evaluation Metrics
def calculate_metrics(actual, predicted):
    """
    Calculate common evaluation metrics for time series forecasts.
    
    Args:
    actual (array-like): Actual values.
    predicted (array-like): Predicted values.
    
    Returns:
    dict: Dictionary containing MAE, MSE, and RMSE.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# Example usage:
# metrics = calculate_metrics(df['y'][-30:], forecast[-30:])

# 6. Visualization
def plot_forecast(actual, forecast, title='Time Series Forecast'):
    """
    Plot actual vs forecasted values.
    
    Args:
    actual (pd.Series): Actual time series data.
    forecast (pd.Series): Forecasted values.
    title (str): Plot title.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast', color='red')
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage:
# plot_forecast(df['y'], arima_forecast)

# 7. Rolling Window Forecast
def rolling_window_forecast(data, window_size, forecast_horizon, model_func):
    """
    Perform rolling window forecast.
    
    Args:
    data (pd.Series): Time series data.
    window_size (int): Size of the rolling window.
    forecast_horizon (int): Number of steps to forecast in each window.
    model_func (function): Function to fit and forecast the model.
    
    Returns:
    pd.DataFrame: DataFrame with actual and forecasted values.
    """
    results = []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        train = data.iloc[i:i+window_size]
        test = data.iloc[i+window_size:i+window_size+forecast_horizon]
        forecast = model_func(train, forecast_horizon)
        results.append(pd.DataFrame({
            'actual': test,
            'forecast': forecast
        }))
    return pd.concat(results)

# Example usage:
# def arima_forecast_func(train, horizon):
#     model = fit_arima(train)
#     return forecast_arima(model, steps=horizon)
# rolling_forecast = rolling_window_forecast(df['y'], window_size=252, forecast_horizon=30, model_func=arima_forecast_func)

# 8. Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_time_series(data, period):
    """
    Perform seasonal decomposition of time series data.
    
    Args:
    data (pd.Series): Time series data.
    period (int): Period of seasonality.
    
    Returns:
    DecomposeResult: Decomposition results containing trend, seasonal, and residual components.
    """
    return seasonal_decompose(data, period=period)

# Example usage:
# decomposition = decompose_time_series(df['y'], period=252)  # Assuming daily data with yearly seasonality
# decomposition.plot()
# plt.show()
