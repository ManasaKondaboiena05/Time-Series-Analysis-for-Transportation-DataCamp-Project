# Time-Series-Analysis-for-Transportation-DataCamp-Project

**What does the project do?**

This project analyzes team travel patterns for the NBA 2102 season. It focuses on evaluating total travel distances, estimating fuel costs, and determining how many teams are airborne at any given time during the season.

**Dataset**

Thsi project used two datasets: fuel prices in 2101 and team flights 2102. This first dataset included one column, namely, fuel prices. The second dataset included the following columns: departure time, landing time, team name, flight duration. 

**How does it work?**

The aim of the project was to find the maximum number of teams that were in the air simulateneously and to find the total fuel cost for the sason 2102. The former was accomplished through a simple loop iterated through the minimum departure time and maximum landing time and added to the counter whenever it encountered a flight that was in the air at that moment. The maximum numer of teams that were in the air simultaneously is stored in a variable called 'max_teams_in_flight'.

The latter was accomplished through a thorough process. I followed the Box-Jenking method to analyse and model the fuel prices dataset. The first step is Identification. In this step, I used the Augmented Dicky-Fuller test to determine whether the data was stationary. After finding a p-value higher than 0.05, I checked for seasonality. I had initially guessed that the data was seasonal as fuel prices tend to to be cyclical. I used normal and seasonal differencing and found the data to be stationary after performing the ADF test again. I used the 'plot_acf' and 'plot_pacf' functions to plot the graphs for the differenced dataset and dound the following values for the model: p (normal AR order), q (normal MA order), P (seasonal AR order), Q (seasonal MA order). The second step is Modelling. This was done using the SARIMAX object from statsmodels. 
