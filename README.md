# COVID_predict
Encoder-Decoder LSTM Model With Multivariate Input For 
Predicting U.S. COVID-19 deaths. 

This repository contains a deep learning, sequence analysis 
model designed to predict the number of deaths in the 
United States from COVID-19 some number of days (typically 7) 
in the future based on the county-specific data on deaths and 
diagnosed ("confirmed") cases available in the Johns Hopkins CSSEGISandDATA github
repository: https://github.com/CSSEGISandData/COVID-19 (referred
to below as JHU or JHU data).
The model also uses county specific data on population density.

The main code is COVID_predict_{version}.(py,ipynb), described now.

The architecture of the model loosely follows that presented in 
Jason Brownlee's Machine Learning Mastery webpage:
https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/.
(scroll down to Encoder-Decoder LSTM model with multivariate input).
There are two LSTM layers followed by a split to two dense layers
for output. 

The input consists of a dataset which is split into training and
testing data (3/4 to 1/4). The datafile is covid_data_4col_06092020.csv
or similar designation. 

Each record in the dataset consists of the following columns:
1. County (FIPS) designation 
2. Day 1 - the first day that a death was reported in that county.
Days are numbered from January 22, 2020.
3. ilast - a simple binary variable which is unity only for the
last record associated with a given county.
4,8,12,16,...,84 the number of deaths recorded on day1, day1+1,...,day1+20
5,9,13,17,...,85 the number of cases recorded on day1,day1+1,...,day1+20
6,10,...,86 the population density for the county (note that this is the
same in all columns - it does not change with day).
7,11,...,87 the day, measured from January 22, 2020, of the given deaths
and cases.

Thus, there are data from 21 consecutive days (deaths,cases,population,day).
Under the current configuration, COVID_predict uses the first 14 days as
input (train_x and test_x) and attempts to predict the deaths in the last 
seven days (for each record). Thus, ground truth (train_y, test_y) is the 
set of deaths on days 15-21. The first three columns of the dataset are
not employed in the deep learning model.

Data file production: Coronavirus_v5.ipynb.

Datasets such as covid_data_4col_06092020.csv are created from the JHU
data using an auxilliary program called Coronavirus_{version}.ipynb.
this code reads three datasets to produce covid_data_4col_06092020.csv:
1. time_series-covid19_deaths_US.csv renamed to
Hopkins_US_deaths_county_06092020.csv
2. time_series_covid19_confirmed_US.csv renamed to
Hopkins_US_confirmed_county_06092020.csv
3. C:/Users/MStopa/County_data/Density_vs_fips.csv which
is extracted from PopDensity_v2_county.csv from U.S.
Census Bureau. 

Model parameters

Typical values of model parameters used in training:
epochs = 40-320
neurons=200-400
batch_size=80
days_in=14
days_out=7



