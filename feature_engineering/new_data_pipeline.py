# %% [markdown]
# ## Imports

# %%
import os
import httpx
import pandas as pd
import requests
import datetime
import json
from urllib.request import Request, urlopen
import random
import datetime
from sklearn.preprocessing import LabelEncoder
hopsworks_api_key=os.getenv("HOPSWORKS_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
pressure_api_key = os.getenv("PRESSURE_API_KEY")
flight_api_key = os.getenv("FLIGHT_API_KEY")

# %% [markdown]
# ## Variables and Maps

# %%
selected_airports_iata = [
    "DTW",
    "LAS",
    "PHL",
    "DEN",
    "CLT",
    "SEA",
    "MCO",
    "FLL",
    "IAD",
    "IAH",
    "SFO",
    "EWR",
    "MIA",
    "JFK",
    "LAX",
    "ORD",
    "ATL",
]
icao_to_iata_map = {
    "KDTW": "DTW",
    "KLAS": "LAS",
    "KPHL": "PHL",
    "KDEN": "DEN",
    "KCLT": "CLT",
    "KSEA": "SEA",
    "KMCO": "MCO",
    "KFLL": "FLL",
    "KIAD": "IAD",
    "KIAH": "IAH",
    "KSFO": "SFO",
    "KEWR": "EWR",
    "KMIA": "MIA",
    "KJFK": "JFK",
    "KLAX": "LAX",
    "KORD": "ORD",
    "KATL": "ATL",
}
iata_to_icao_map = {v: k for k, v in icao_to_iata_map.items()}
wac_map = {
    "BOS": 13,
    "CLT": 36,
    "DEN": 82,
    "DTW": 43,
    "EWR": 21,
    "FLL": 33,
    "IAD": 38,
    "IAH": 74,
    "JFK": 22,
    "LAS": 85,
    "LAX": 91,
    "MCO": 33,
    "MIA": 33,
    "ORD": 41,
    "PHL": 23,
    "SEA": 93,
    "SFO": 91,
    "ATL": 34,
}
weather_features = [
    ("dewpoint", "value"),
    "relative_humidity",
    ("remarks_info", "precip_hourly", "value"),
    ("remarks_info", "temperature_decimal", "value"),
    ("visibility", "value"),
    ("wind_direction", "value"),
    ("wind_gust", "value"),
    ("wind_speed", "value"),
]
pressure_features = [("pressure", "hpa")]
flight_features = [
    "flight_date",
    ("departure", "iata"),
    ("departure", "delay"),
    ("departure", "scheduled"),
    ("arrival", "iata"),
    ("arrival", "delay"),
    ("arrival", "scheduled"),
]
previous_mapping = {
    "ATL": 0,
    "CLT": 1,
    "DEN": 2,
    "DTW": 3,
    "EWR": 4,
    "FLL": 5,
    "IAD": 6,
    "IAH": 7,
    "JFK": 8,
    "LAS": 9,
    "LAX": 10,
    "MCO": 11,
    "MIA": 12,
    "ORD": 13,
    "PHL": 14,
    "SEA": 15,
    "SFO": 16,
}

# Create a label encoder
label_encoder = LabelEncoder()

# Fit the label encoder with the previous mappings
label_encoder.fit(list(previous_mapping.keys()))
airport_encoding_map = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print(airport_encoding_map)
# %% [markdown]
# ## Weather data (no pressure data)

# %%
responses = {}
for airport in selected_airports_iata:
    print(f"Getting weather for {airport}")
    request = Request(
        f"https://avwx.rest/api/metar/{iata_to_icao_map[airport]}",
        headers={"Authorization": weather_api_key},
    )
    response_body = urlopen(request).read()
    response_json = json.loads(response_body)
    responses[airport] = response_json

weather_data = []

for airport in selected_airports_iata:
    response_json = responses[airport]
    data = {"airport": airport}
    data["HourlyDewPointTemperature"] = response_json["remarks_info"][
        "dewpoint_decimal"
    ]["value"]
    data["HourlyRelativeHumidity"] = response_json["relative_humidity"]
    if response_json["remarks_info"]["precip_hourly"] is not None:
        data["HourlyPrecipitation"] = response_json["remarks_info"]["precip_hourly"][
            "value"
        ]
    else:
        data["HourlyPrecipitation"] = 0
    data["HourlyDryBulbTemperature"] = response_json["remarks_info"][
        "temperature_decimal"
    ]["value"]
    data["HourlyVisibility"] = response_json["visibility"]["value"]
    data["HourlyWindDirection"] = response_json["wind_direction"]["value"]
    if response_json["wind_gust"] is not None:
        data["HourlyWindGustSpeed"] = response_json["wind_gust"]["value"]
    else:
        data["HourlyWindGustSpeed"] = 0
    data["HourlyWindSpeed"] = response_json["wind_speed"]["value"]
    weather_data.append(data)

weather_data = pd.DataFrame(weather_data)
print(weather_data)
# %% [markdown]
# ## Barometric pressure data

# %%
url = "https://api.checkwx.com/metar/KJFK/decoded"

response = requests.request("GET", url, headers={"X-API-Key": pressure_api_key})

for airport in selected_airports_iata:
    print(f"Getting pressure for {airport}")
    request = Request(
        f"https://api.checkwx.com/metar/{iata_to_icao_map[airport]}/decoded",
        headers={"X-API-Key": pressure_api_key},
    )
    response_body = urlopen(request).read()
    response_json = json.loads(response_body)
    responses[airport] = response_json

pressure_data = []

for airport in selected_airports_iata:
    response_json = responses[airport]
    data = {"airport": airport}
    data["HourlyStationPressure"] = response_json["data"][0]["barometer"]["hpa"]
    pressure_data.append(data)

pressure_data = pd.DataFrame(pressure_data)
print(pressure_data)

# %% [markdown]
# ## Flight data

# %%
endpoint = "http://api.aviationstack.com/v1/flights"
flight_data = []


for i in range(len(selected_airports_iata)):
    print(f"Getting flights for {selected_airports_iata[i]}")
    # Parameters for the API request
    rand_int = random.randint(0, len(selected_airports_iata) - 1)
    params = {
        "access_key": flight_api_key,
        "dep_iata": selected_airports_iata[
            i
        ],  # Replace 'SFO' with your desired airport code
        "arr_iata": selected_airports_iata[rand_int],
        "flight_status": ["active"],
    }

    # Making the GET request
    response = requests.get(endpoint, params=params)
    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        response = response.json()  # Parsing the JSON response
        # Handle and process 'data' (flight information)
        data = {"airport": selected_airports_iata[i]}
        if len(response["data"]) == 0:
            continue

        rand_int = random.randint(0, len(response["data"]) - 1)

        response = response["data"][rand_int]

        data["FL_DATE"] = response["flight_date"]
        data["ORIGIN"] = response["departure"]["iata"]
        if response["departure"]["delay"] is not None:
            data["DEP_DELAY_NEW"] = response["departure"]["delay"]
        else:
            data["DEP_DELAY_NEW"] = 0.0
        data["DEPARTURE_DATETIME"] = response["departure"]["scheduled"]
        data["DEST"] = response["arrival"]["iata"]
        data["CRS_ARR_TIME"] = response["arrival"]["scheduled"]
        flight_data.append(data)
    else:
        print(response.json())
        print("Request failed:", response.status_code)

flight_data = pd.DataFrame(flight_data)

# %% [markdown]
# ## Merging the data

# %%
weather_delay_data = pd.merge(weather_data, flight_data, on="airport")
weather_delay_data = pd.merge(weather_delay_data, pressure_data, on="airport")

# %% [markdown]
# ## Transformation

# %%
# drop rows with missing values
weather_delay_data = weather_delay_data.dropna()

# drop airport column
weather_delay_data = weather_delay_data.drop(columns=["airport"])

# Transform DEST and ORIGIN to string
weather_delay_data["DEST"] = weather_delay_data["DEST"].astype(str)
weather_delay_data["ORIGIN"] = weather_delay_data["ORIGIN"].astype(str)

# add wac code to weather_delay_data
weather_delay_data["DEST_WAC"] = weather_delay_data["DEST"].map(wac_map)
weather_delay_data["ORIGIN_WAC"] = weather_delay_data["ORIGIN"].map(wac_map)

# Transform date columns to datetime
weather_delay_data["FL_DATE"] = pd.to_datetime(weather_delay_data["FL_DATE"])
weather_delay_data["DEPARTURE_DATETIME"] = pd.to_datetime(
    weather_delay_data["DEPARTURE_DATETIME"]
)

# Transform hourly wind direction to int
weather_delay_data["HourlyWindDirection"] = weather_delay_data[
    "HourlyWindDirection"
].astype(int)

weather_delay_data["CRS_DEP_TIME"] = pd.to_datetime(
    weather_delay_data["DEPARTURE_DATETIME"]
).dt.time

# change  CRS_DEP_TIME to the format HHMM
weather_delay_data["CRS_DEP_TIME"] = weather_delay_data["CRS_DEP_TIME"].apply(
    lambda x: x.strftime("%H%M")
)

weather_delay_data["CRS_ARR_TIME"] = pd.to_datetime(
    weather_delay_data["CRS_ARR_TIME"]
).dt.time

# change  CRS_ARR_TIME to the format HHMM
weather_delay_data["CRS_ARR_TIME"] = weather_delay_data["CRS_ARR_TIME"].apply(
    lambda x: x.strftime("%H%M")
)

# add year, quarter, month, day_of_month, day_of_week
# weather_delay_data["YEAR"] = weather_delay_data["FL_DATE"].dt.year
# weather_delay_data["QUARTER"] = weather_delay_data["FL_DATE"].dt.quarter
weather_delay_data["MONTH"] = weather_delay_data["FL_DATE"].dt.month
weather_delay_data["DAY_OF_MONTH"] = weather_delay_data["FL_DATE"].dt.day
weather_delay_data["DAY_OF_WEEK"] = weather_delay_data["FL_DATE"].dt.dayofweek

# %%
airport_id_map = {
    "CLT": 11057,
    "DEN": 11292,
    "DTW": 11433,
    "EWR": 11618,
    "FLL": 11697,
    "IAD": 12264,
    "IAH": 12266,
    "JFK": 12478,
    "LAS": 12889,
    "LAX": 12892,
    "MCO": 13204,
    "MIA": 13303,
    "ORD": 13930,
    "PHL": 14100,
    "SEA": 14747,
    "SFO": 14771,
    "ATL": 10397,
}

# Mapping IATA codes to airport IDs for 'dest' and 'origin' columns
weather_delay_data["dest_airport_id"] = weather_delay_data["DEST"].map(
    airport_encoding_map
)
weather_delay_data["origin_airport_id"] = weather_delay_data["ORIGIN"].map(
    airport_encoding_map
)


weather_delay_data.drop(
    columns=["DEST", "ORIGIN", "FL_DATE", "DEPARTURE_DATETIME"], inplace=True
)


columns_to_int64 = [
    "CRS_ARR_TIME",
    "CRS_DEP_TIME",
    # "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
]
columns_to_float64 = [
    "HourlyPrecipitation",
    "HourlyVisibility",
    "HourlyWindGustSpeed",
    "HourlyWindSpeed",
    "DEP_DELAY_NEW",
]

for column in columns_to_int64:
    # Convert to int64
    weather_delay_data[column] = weather_delay_data[column].astype("int64")
for column in columns_to_float64:
    # Convert to int64
    weather_delay_data[column] = weather_delay_data[column].astype("float64")

# %% [markdown]
# ### Insert into hopswork dataset

# %%
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

flight_delay_fg = fs.get_feature_group(name="flight_data_v3", version=1)
flight_delay_fg.insert(weather_delay_data)

# %%
