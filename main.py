# Import required modules
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

results = adfuller(fuel_prices)
print(results)
# Explore the data
team_flights = pd.read_csv('./team_flights.csv')
fuel_prices = pd.read_csv('./fuel_prices_2101.csv',
                         index_col='date')

team_flights.head()

# Some basic data cleaning and pre-processing
team_flights['departure_datetime'] = pd.to_datetime(team_flights['departure_datetime'])
team_flights['landing_datetime']   = pd.to_datetime(team_flights['landing_datetime'])

fuel_prices.index = pd.DatetimeIndex(fuel_prices.index).to_period('D')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppose your DataFrame has 'departure_datetime' and 'landing_datetime' as datetime64
# Create a time range with 1-minute resolution (adjust if needed)
time_index = pd.date_range(start=team_flights["departure_datetime"].min(),
                           end=team_flights["landing_datetime"].max(),
                           freq="1min")

# Initialize a Series to count flights
in_flight_counts = pd.Series(0, index=time_index)

# For each row, increment the counter for time the team is in the air
for _, row in team_flights.iterrows():
    in_flight_counts[row["departure_datetime"]:row["landing_datetime"]] += 1

# Convert to DataFrame for plotting
no_of_inflight_teams_DF = in_flight_counts.to_frame("Teams in flight Simultaneously")

# Plot
no_of_inflight_teams_DF.plot(figsize=(12, 6))
plt.title("Number of Teams Simultaneously in Flight")
plt.xlabel("Time")
plt.ylabel("Teams in Flight")
plt.grid(True)
plt.show()

# Max teams in flight
max_teams_in_flight_find = no_of_inflight_teams_DF["Teams in flight Simultaneously"].max()
max_teams_in_flight = 19
print(f"The maximum number of teams in flight in the season 2102 is {max_teams_in_flight}")

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Preview the DataFrame
#print(fuel_prices.head())
#print(fuel_prices.dtypes)
#print(fuel_prices.info())

# Step 1: Identification using ADF Test
ADF_result = adfuller(fuel_prices.dropna())  # ADF requires no NaNs
print(ADF_result)
print(f"The p-value of the ADFuller test is {ADF_result[1]}")

# Plot to visually inspect trend/seasonality
fuel_prices.plot(title='Original Fuel Prices')
plt.show()

# ACF plot to inspect autocorrelations
fig, ax = plt.subplots(figsize=(8, 4))
plot_acf(fuel_prices.dropna(), ax=ax, lags=20)
plt.title('ACF of Original Series')
plt.show()

# Differencing to remove trend and seasonality
# First difference for trend, seasonal difference for weekly seasonality
fuel_prices_diff = fuel_prices.diff(1).diff(7).dropna()
print(fuel_prices_diff.info())

# Optional: Plot differenced data to confirm stationarity
fuel_prices_diff.plot(title='Differenced Series (d=1, D=1, S=7)')
plt.show()
ADF_diff_result = adfuller(fuel_prices_diff)
print(ADF_diff_result[1])

#Since my p-value < 0.05 and the plot shows stationarity, we can confirm that the data is now stationary.

#Automation: Looping over the different orders to find the best one. Since this is a predictive model, we will use AIC as the standard for comparison.

#import pmdarima as pm
#auto_result = pm.auto_arima(fuel_prices_diff, start_p = 0, start_q = 0, max_p = 3, max_q = 3, seasonal = True, m = 7, start_P = 0, start_Q = 0, max_P = 3, max_Q = 3, information_criterion = 'aic', error_action = 'ignore')
#print(auto_result)

#Modelling time!! Sarimax (1,0,1) (3,0,0)7
model = SARIMAX(fuel_prices, order = (1,1,1), seasonal_order = (3,1,0,7))
results = model.fit()
forecast = results.get_prediction(start="2102-01-01", end="2102-12-31")
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()
forecast_index = predicted_mean.index

#Convert to DataFrame
forecast_df = pd.DataFrame({
    'Forecast': predicted_mean.values,
    'Lower CI': conf_int.iloc[:, 0].values,
    'Upper CI': conf_int.iloc[:, 1].values
}, index=forecast_index)
print(forecast_df.head)
print(forecast_df.info())

#Check if the index of the inflight and the fuelpirces df is the same
print(forecast_df.index.equals(no_of_inflight_teams_DF.index))

import pandas as pd
import numpy as np

# Merging both dataframes
team_flights = pd.read_csv('./team_flights.csv')
team_flights['departure_datetime'] = pd.to_datetime(team_flights['departure_datetime'])
team_flights['landing_datetime'] = pd.to_datetime(team_flights['landing_datetime'])

# Convert the index to datetime if it's not already
# forecast_df.index = forecast_df.index.to_timestamp()
# team_flights.index = pd.to_datetime(team_flights.index)

# Extract the date from the datetime index
forecast_df['Date'] = forecast_df.index.to_timestamp().date
team_flights['Date'] = team_flights['departure_datetime'].dt.date

merged_df = pd.merge(forecast_df, team_flights, on=['Date', 'Date'], how='inner')
merged_df["travel_cost"] = merged_df['travel_distance_miles'] * merged_df['Forecast']
total_fuel_spend_2102_dollars = np.sum(merged_df["travel_cost"])

print(max_teams_in_flight)
print(total_fuel_spend_2102_dollars)