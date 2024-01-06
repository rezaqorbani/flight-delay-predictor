# US Flight delay predictor

## Table of Contents

- [US Flight delay predictor](#us-flight-delay-predictor)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
    - [Flight data](#flight-data)
    - [Weather data](#weather-data)
  - [Features](#features)
  - [Model](#model)
  - [Results](#results)
  - [Inference](#inference)
    - [Batch inference](#batch-inference)
  - [Montioring](#montioring)
  - [Appendix](#appendix)
    - [Airports](#airports)

## Introduction

This is a serveless service for predicting the delay (in minutes) of any arriving or departing flight in any of top 18 busiest airports (listed in [Airports](#airports) section) in US. Our model is trained on historical flight and weather data from 2021 and 2022 but uses real-time weather data during inference. Using GitHub actions we schedule a run of data pipeline to add fresh data to our training dataset as well as monitor the performance of our model on the latest data. The model can be trained on-demand. There are seperate serverless UI for inference and monitoring. [Hopsworks](https://www.hopsworks.ai/) is used to store extracted features as well the trained model.

## Data

### Flight data

The raw historical flight data used in this project is Airline Reporting Carrier On-Time Performance Dataset which can be downloaded from [Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr). This dataset contains detailed and fine-grained information about all the flights in US for a large range of years. We focus on a subset of these features as described in [Features](#features) section. We store this raw file locally, before preprocessing.

We add new data to our training dataset from [Aviationstack](https://aviationstack.com/). Aviationstack provides real-time flight data for all the flights in numerous locations around the world. We specifically fetch flight data for airports described in [Airports](#airports) section.

### Weather data

The source of raw historical weather data in this project is the U.S. Local Climatological Data (LCD) from [National Centers for Environmental Information](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00684/html). This dataset contains daily weather summaries for all the weather stations in US. From search interface, we select the stations that are situated in the airports described in [Airports](#airports) section. We use the weather data from 2021 and 2022 for training our model. After selection, all the data is downloaded as one csv file and is stored locally. Similar to the flight data, this dataset contains many features but we focus on a subset of these features as described in [Features](#features) section.

For the new weather data we use two different APIs. The first one is [CheckWX](https://www.checkwxapi.com/) which provides real-time weather data for all the weather stations around the world. We use this API only to fetch pressure data (more on this in [Features](#features) section) for the airports used because there were some inconsistencies between the features provided by the API and our historical weather data. The second API is [AVWX Aviation Weather REST API](https://info.avwx.rest/) which provides real-time weather data for all the airports around the world, similar to CheckWX. We use this API to fetch all the remaining weather features for the airports described in [Airports](#airports) section.

## Features

The raw flight and weather data are preprocessed in [preprocessing_flight_data.ipynb](./feature_engineering/preprocessing_flight_data.ipynb) and [weather_data_preprocessing.ipynb](./feature_engineering/weather_data_preprocessing.ipynb) respectively. The preprocessed data is then used to extract features in [feature_engineering.ipynb](./feature_engineering/feature_engineering.ipynb). The extracted features are stored in Hopsworks and are used for training the model. For flight data we extract basic features about flight time/date and delay. For weather data we extract a diverse set of features about temperature, wind, visibility, pressure, etc. We had to perform a lot of manual feature engineering and data wrangling to extract these features as the raw data was not in a very clean format. The extracted features and the preprocessing steps are described in detail in the notebooks mentioned above.

In order to create our final dataset for training the model, we join the extracted flight and weather features on the flight date and airport code. We also add a few more features to this dataset. The final dataset contains approximately 1.7 million rows and 19 features.

## Model

## Results

## Inference

### Batch inference

The [monitoring](./monitoring/monitoring.py) job performs batch inference where a batch of 15 most recent data points are fetched from the database and the model is used to predict the delay for each data point. We take 15 most recent data because on average our new data pipeline adds 15 new data points to the database every time it is run.

## Montioring

## Appendix

### Airports

| IATA Code | Airport Name                                          | City               |
|-----------|-------------------------------------------------------|--------------------|
| ATL       | Hartsfield–Jackson Atlanta International Airport     | Atlanta            |
| CLT       | Charlotte Douglas International Airport              | Charlotte          |
| DEN       | Denver International Airport                          | Denver             |
| DTW       | Detroit Metropolitan Wayne County Airport             | Detroit            |
| EWR       | Newark Liberty International Airport                  | Newark             |
| FLL       | Fort Lauderdale–Hollywood International Airport      | Fort Lauderdale    |
| IAD       | Washington Dulles International Airport               | Dulles             |
| IAH       | George Bush Intercontinental Airport                  | Houston            |
| JFK       | John F. Kennedy International Airport                 | New York           |
| LAS       | McCarran International Airport                        | Las Vegas          |
| LAX       | Los Angeles International Airport                     | Los Angeles        |
| MCO       | Orlando International Airport                         | Orlando            |
| MIA       | Miami International Airport                           | Miami              |
| ORD       | O'Hare International Airport                          | Chicago            |
| PHL       | Philadelphia International Airport                    | Philadelphia       |
| SEA       | Seattle-Tacoma International Airport                  | Seattle            |
| SFO       | San Francisco International Airport                   | San Francisco      |
