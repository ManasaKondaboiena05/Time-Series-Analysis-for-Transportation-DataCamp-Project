# Time-Series-Analysis-for-Transportation-DataCamp-Project

**What does the project do?**

This project analyzes team travel patterns for the NBA 2102 season. It focuses on evaluating total travel distances, estimating fuel costs, and determining how many teams are airborne at any given time during the season.

**Dataset**

Thsi project used two datasets: fuel prices in 2101 and team flights 2102. This first dataset included one column, namely, fuel prices. The second dataset included the following columns: departure time, landing time, team name, flight duration. 

**Aim of the project**

The aim of the project was twofold:

1. To identify the maximum number of teams that were in the air simultaneously during the 2102 season.
2. To estimate the total fuel cost incurred by these flights over the course of the season.

**Maximum Teams in Air**

The first objective was achieved through a time-indexed loop that iterated between the earliest departure and the latest landing time, incrementing a counter whenever a flight was airborne during a given minute. The result, stored in the variable max_teams_in_flight, reflects the maximum number of teams in the air at the same time during the 2102 season.

**Fuel Cost Estimation**

To address the second objective, I employed the Box-Jenkins methodology to model the daily jet fuel prices and forecast them throughout 2102. Here's a breakdown of the steps followed:

**- Identification:**
  I conducted the Augmented Dickey-Fuller (ADF) test on the raw fuel price data, which revealed non-stationarity. Suspecting seasonality (a common trait in fuel prices), I applied both first-order and seasonal      differencing. The differenced series passed the ADF test, indicating stationarity.

**- Model Selection:**
  Using ACF and PACF plots, I selected suitable orders for both seasonal and non-seasonal components and fit a SARIMAX model to the original series.

**- Diagnostics:**
  I used plot_diagnostics() to inspect model residuals. While the residuals showed no autocorrelation and passed the Ljung-Box test, there were violations of normality observed in both the Q-Q plot and the          histogram of residuals. The Jarque-Bera statistic’s p-value was 0, suggesting non-normality.

**- Forecasting & Cost Calculation:**
  Despite the normality issues, the model’s forecasts were used to estimate daily fuel prices, which were then merged with the flight schedule data to compute the total fuel cost for the 2102 season, stored in      total_fuel_spend_2102_dollars.


**Final Thoughts**

While the model diagnostics revealed certain limitations, particularly regarding the normality of residuals, the Box-Jenkins process provided a structured and rigorous framework for modeling and forecasting. These issues highlight potential areas for further refinement, such as applying transformations (e.g., Box-Cox) or experimenting with alternative model structures.

