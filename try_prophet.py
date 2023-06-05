import pandas as pd
from fbprophet import Prophet

# Generate some example time series data
df = pd.DataFrame({'ds': pd.date_range(start='2022-01-01', periods=100, freq='D'), 'y': range(100)})

# Create the Prophet model
m = Prophet(seasonality_mode='additive')

# Fit the model to the data
m.fit(df)

# Generate backcasted predictions
future = m.make_future_dataframe(periods=10, freq='D', include_history=False)
forecast = m.predict(future)

# Print the backcasted predictions
backcast = forecast.iloc[::-1]
print(backcast)
