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

This is a serveless service for predicting the delay (in minutes) of any departing flight in any of top 17 busiest airports (listed in [Airports](#airports) section) in US. Our model is trained on historical flight and weather data from 2021 and 2022 but uses real-time weather data during inference. Using GitHub actions we schedule a daily run of data pipeline to add fresh data to our training dataset as well as monitor the performance of our model on the latest data. The model can be trained on-demand. There are seperate serverless UI for inference and monitoring. [Hopsworks](https://www.hopsworks.ai/) is used to store extracted features as well as the trained model.

[MONITORING UI](https://huggingface.co/spaces/rezaqorbani/flight_delay_monitor)

[Flight Delay Prediction app](https://huggingface.co/spaces/Yulle/flight_delay_prediction)

## Data

### Flight data

The raw historical flight data used in this project is Airline Reporting Carrier On-Time Performance Dataset which can be downloaded from [Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr). This dataset contains detailed and fine-grained information about all the flights in US for a large range of years. We focus on a subset of these features as described in [Features](#features) section. We store this raw file locally, before preprocessing. 

The Flight data contains the target variable which is the departure delay. The departure delay is a numerical value that represents how many minutes delayed the flight departed compared to the scheduled departure time.

We add new data to our training dataset from [Aviationstack](https://aviationstack.com/). Aviationstack provides real-time flight data for all the flights in numerous locations around the world. We specifically fetch flight data for airports described in [Airports](#airports) section.

### Weather data

The source of raw historical weather data in this project is the U.S. Local Climatological Data (LCD) from [National Centers for Environmental Information](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00684/html). This dataset contains daily weather summaries for all the weather stations in US. From search interface, we select the stations that are situated in the airports described in [Airports](#airports) section. We use the weather data from 2021 and 2022 for training our model. After selection, all the data is downloaded as one csv file and is stored locally. Similar to the flight data, this dataset contains many features but we focus on a subset of these features as described in [Features](#features) section.

For the new weather data we use two different APIs. The first one is [CheckWX](https://www.checkwxapi.com/) which provides real-time weather data for all the weather stations around the world. We use this API only to fetch pressure data (more on this in [Features](#features) section) for the airports used because there were some inconsistencies between the features provided by the API and our historical weather data. The second API is [AVWX Aviation Weather REST API](https://info.avwx.rest/) which provides real-time weather data for all the airports around the world, similar to CheckWX. We use this API to fetch all the remaining weather features for the airports described in [Airports](#airports) section.

## Features

The raw flight and weather data are preprocessed in [preprocessing_flight_data.ipynb](./feature_engineering/preprocessing_flight_data.ipynb) and [weather_data_preprocessing.ipynb](./feature_engineering/weather_data_preprocessing.ipynb) respectively. The preprocessed data is then used to extract features in [feature_engineering.ipynb](./feature_engineering/feature_engineering.ipynb). The extracted features are stored in Hopsworks and are used for training the model. For flight data we extract basic features about scheduled flight time/date and delay. For weather data we extract a diverse set of features about temperature, wind, visibility, pressure, etc. We had to perform a lot of manual feature engineering and data wrangling to extract these features as the raw data was not in a very clean format. The extracted features and the preprocessing steps are described in detail in the notebooks mentioned above.

In order to create our final dataset for training the model, we join the extracted flight and weather features on the flight date and airport code. We also add a few more features to this dataset. The final dataset contains approximately 1.7 million rows and 19 features.
The dataset is stored in Hopsworks where we can easily access it for training the model or performing inference.

## Model

3 Different models were trained using the data that we created. A Linear Regression, a Random Forest Regressr model and a Neural Network model were each created and trained on the flight and weather data. The Neural Network model was implemented using Pytorch and the Linear Regression as well as the Random Forest Regressor was implemented using Scikit-Learn. The notebook for the training of the models can be found at: [training_pipeline](./training_pipeline.ipynb). A 80/20 train/test split was used for training the models. The best performing model, which was the Neural Network model, was uploaded to Hopsworks model registry. The Neural network model used ReLu as activation function, it was trained using MSE loss and used Adam as the optimizer. Dropout was also implemented for regularization.

## Results

| Model                  | MSE                      |             
| ---------------------- | ------------------------ |
| Linear Regression      |               2342       |
| Random Forest Regressor|               2291       |
| Neural Network         |               2280       |

## Inference

### Prediction app

A Gradio app was created to allow users to input scheduled arrival/departure time as well as origin and destination airports to predict the departure delay using our trained Neural Network model. The app fetches live weather data for the chosen airports. Then, it also fetches data about the current date for the prediction. All the data is preprocessed and then used for inference using the trained Neural Network model. The model is retrieved by calling the hopsworks API to fetch it from the model registry. The predicted departure delay from the model is then shown to the user. Thee delay is shown in minutes. Since live weather data is used, the user has to input scheduled departure times close to the time when they are making the inference.
### Batch inference

The [monitoring](./monitoring/monitoring.py) job performs batch inference where a batch of 15 most recent data points are fetched from the database and the model is used to predict the delay for each data point. The data is retrieved from Hopsworks. We take 15 most recent data because on average our new data pipeline adds 15 new data points to the database every time it is run.

## Montioring

For monitoring we simply show previous prediction of our model and calculated MSE. For each new prediction, we calculate the MSE of the 15 last predictions and show it. This way we get a better picture of how MSE is evolving as new data comes in. The [monitoring](./monitoring/monitoring.py) job is scheduled to run every day at 21:00 UTC. The monitoring data is also stored in Hopsworks.

## Appendix

### Airports

| IATA Code | Airport Name                                     | City            |
| --------- | ------------------------------------------------ | --------------- |
| ATL       | Hartsfield–Jackson Atlanta International Airport | Atlanta         |
| CLT       | Charlotte Douglas International Airport          | Charlotte       |
| DEN       | Denver International Airport                     | Denver          |
| DTW       | Detroit Metropolitan Wayne County Airport        | Detroit         |
| EWR       | Newark Liberty International Airport             | Newark          |
| FLL       | Fort Lauderdale–Hollywood International Airport  | Fort Lauderdale |
| IAD       | Washington Dulles International Airport          | Dulles          |
| IAH       | George Bush Intercontinental Airport             | Houston         |
| JFK       | John F. Kennedy International Airport            | New York        |
| LAS       | McCarran International Airport                   | Las Vegas       |
| LAX       | Los Angeles International Airport                | Los Angeles     |
| MCO       | Orlando International Airport                    | Orlando         |
| MIA       | Miami International Airport                      | Miami           |
| ORD       | O'Hare International Airport                     | Chicago         |
| PHL       | Philadelphia International Airport               | Philadelphia    |
| SEA       | Seattle-Tacoma International Airport             | Seattle         |
| SFO       | San Francisco International Airport              | San Francisco   |
