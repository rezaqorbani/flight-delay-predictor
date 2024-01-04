# %%
import gradio as gr
import numpy as np
import requests
import pandas as pd
import hopsworks
import joblib
import torch
from torch import nn


import os
from dotenv import load_dotenv
import httpx
import datetime
import json
from urllib.request import Request, urlopen
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler




# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %%
#Load api keys
load_dotenv()
weather_api_key = os.getenv("weather_api_key")
pressure_api_key = os.getenv("pressure_api_key")
flight_api_key = os.getenv("flight_api_key")


# %%
#Mappings
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
airport_id_map={
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
label_tranformed_airport_id_map={'ATL': 0, 'CLT': 1, 'DEN': 2, 'DTW': 3, 'EWR': 4, 'FLL': 5, 'IAD': 6, 'IAH': 7, 'JFK': 8,
                                  'LAS': 9, 'LAX': 10, 'MCO': 11, 'MIA': 12, 'ORD': 13, 'PHL': 14, 'SEA': 15, 'SFO': 16}
# Create predefined lists for origin and destination airport codes
airports = [ "PHL - PHILADELPHIA INTERNATIONAL AIRPORT, PA US",
    "SEA - SEATTLE TACOMA AIRPORT, WA US",
    "JFK - JFK INTERNATIONAL AIRPORT, NY US",
    "DEN - DENVER INTERNATIONAL AIRPORT, CO US",
    "EWR - NEWARK LIBERTY INTERNATIONAL AIRPORT, NJ US",
    "LAS - MCCARRAN INTERNATIONAL AIRPORT, NV US",
    "MCO - ORLANDO INTERNATIONAL AIRPORT, FL US",
    "ATL - ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT, GA US",
    "FLL - FORT LAUDERDALE INTERNATIONAL AIRPORT, FL US",
    "DTW - DETROIT METRO AIRPORT, MI US",
    "IAD - WASHINGTON DULLES INTERNATIONAL AIRPORT, VA US",
    "ORD - CHICAGO OHARE INTERNATIONAL AIRPORT, IL US",
    "LAX - LOS ANGELES INTERNATIONAL AIRPORT, CA US",
    "CLT - CHARLOTTE DOUGLAS AIRPORT, NC US",
    "MIA - MIAMI INTERNATIONAL AIRPORT, FL US",
    "IAH - HOUSTON INTERCONTINENTAL AIRPORT, TX US",
    "SFO - SAN FRANCISCO INTERNATIONAL AIRPORT, CA US"]

# %%

#Class definition needed due to the way pytorch neural networks are saved and loaded by python
# A solution, if needed, would be to save the state dict of the NN and load the model via load_state_dict
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

#Load model from model registry
mr = project.get_model_registry()
model = mr.get_model("flight_delay_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/flight_delay_model.pkl")

# get the original train test splits used for training the model and use it for fitting scaler
feature_view = fs.get_feature_view(name="flight_data_v3",version=1)
X_train, X_test, y_train, y_test = feature_view.get_train_test_split(training_dataset_version=3)

#fit scaler the same way it was used for training
scaler = StandardScaler()
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_train_scaled = scaler.fit_transform(X_train_tensor)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

results = pd.DataFrame(columns=["Origin Airport", "Destination Airport", "Scheduled Departure", "Scheduled Arrival", "Predicted Departure Delay"])


# %%
def get_weather_data(selected_airports_iata):
    # Input: list of selected airports in IATA code
    # Make API call to fetch weather data for the airport
    # Process and return weather data
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
    #weather_data.info()
    return weather_data

# %%
def get_pressure_data(selected_airports_iata):
    # Input: list of selected airports in IATA code
    responses={}
    url = "https://api.checkwx.com/metar/KJFK/decoded"

    #response = requests.request("GET", url, headers={"X-API-Key": pressure_api_key})
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
    #pressure_data.info()
    return pressure_data

# %%
def get_flight_data(origin, destination,scheduled_dep_time, scheduled_arr_time):
    # Input: origin airport IATA code, destination airport IATA code, 
    # and dep and arr time in HH:MM 24 hour format
    current_datetime = datetime.now()

    # Extract different date-related information
    day_of_week = current_datetime.weekday()  
    day_of_month = current_datetime.day  
    year = current_datetime.year  
    month = current_datetime.month



    origin_wac = wac_map[origin]
    origin_airport_id = label_tranformed_airport_id_map[origin]

    # Mapping destination to dest_WAC and dest_airport_id
    dest_wac = wac_map[destination]
    dest_airport_id = label_tranformed_airport_id_map[destination]
    # Create a DataFrame for the given airport codes
    airport_df = pd.DataFrame({
        #"Year":[year],
        "month":[month],
        "Day_of_month":[day_of_month],
        "Day_of_week":[day_of_week],
        "origin": [origin],
        "origin_airport_id": [origin_airport_id],
        "origin_WAC": [origin_wac],
        "dest": [destination],
        "dest_airport_id": [dest_airport_id],
        "dest_WAC": [dest_wac],
        "CRS_DEP_TIME":[int(scheduled_dep_time.replace(":", ""))],
        "CRS_ARR_TIME":[int(scheduled_arr_time.replace(":", ""))],
        "airport":[origin]
        
    })

    #print(airport_df.info())
    #print(airport_df)
    return airport_df

# %%
# Define the function to predict flight delay based on user inputs
def predict_delay(origin, destination,scheduled_dep_time, scheduled_arr_time):
    
    #test code to try running Gradio app
    origin=origin.split()[0]
    destination=destination.split()[0]
    
    #error handling
    try:
        # check if correct hour format by trying to convert to datetime objects
        datetime.strptime(scheduled_dep_time, "%H:%M")
        datetime.strptime(scheduled_arr_time, "%H:%M")
    except ValueError:
        # else error
        return "Error: Please enter scheduled departure and arrival times in 24-hour format (HH:MM)."
    if origin == destination:
        return "Error: Origin and destination airports cannot be the same. Please select different airports."
    
    #Get data from APIs
    selected_airports_iata = [origin,destination]
    weather_data=get_weather_data(selected_airports_iata)
    pressure_data=get_pressure_data(selected_airports_iata)
    flight_data=get_flight_data(origin, destination,scheduled_dep_time, scheduled_arr_time)

    #Merge data
    weather_delay_data = pd.merge(pressure_data, weather_data, on="airport")

    # fix order of columns so that it is same as in training
    weather_delay_data=weather_delay_data.reindex(sorted(weather_delay_data.columns), axis=1)
    
    #merge columns
    flight_weather_data=pd.merge(flight_data, weather_delay_data, on="airport")
    
    #drop objects
    flight_weather_data.drop(columns=['airport', 'origin', 'dest'], inplace=True)

    #fix type
    columns_to_float64 = ['HourlyPrecipitation', 'HourlyVisibility', 'HourlyWindGustSpeed', 'HourlyWindSpeed']
    for column in columns_to_float64:
        # Convert to int64
        flight_weather_data[column] = flight_weather_data[column].astype('float64')

    #flight_weather_data.info()

    flight_weather_data=torch.tensor(flight_weather_data.values, dtype=torch.float32)
    print(flight_weather_data)
    #flight_weather_data=scaler.transform(flight_weather_data.reshape(1, -1))
    flight_weather_data=scaler.transform(flight_weather_data)

    print(flight_weather_data)
    # transform np array to torch tensor
    flight_weather_data_tensor=torch.tensor(flight_weather_data, dtype=torch.float32)
    print(flight_weather_data_tensor)

    output=model(flight_weather_data_tensor)
    """
    return_dict = {
        'Origin Airport': origin,
        'Destination Airport': destination,
        'Scheduled Departure': scheduled_dep_time,
        'Scheduled Arrival': scheduled_arr_time,
        'Predicted Departure Delay': int(output.item()) 
    }


    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame([return_dict])
    return df
    """
    global results
    new_prediction = {
        'Origin Airport': origin,
        'Destination Airport': destination,
        'Scheduled Departure': scheduled_dep_time,
        'Scheduled Arrival': scheduled_arr_time,
        'Predicted Departure Delay': int(output.item())  
    }
    # Append the new prediction to the existing DataFrame
    results = pd.concat([results, pd.DataFrame([new_prediction])])

    return results
    #return "Predicted delay for {} to {} with the scheduled departure time {} and scheduled " \
    #    "arrival time {} is {} minutes".format(origin, destination, scheduled_dep_time, scheduled_arr_time,int(output.item()))


# %%
# Create Gradio interface with dropdowns for airport selection
with gr.Blocks() as demo:
    gr.Markdown("# Flight departure delay predictor using Flight data and Weather Data")
    gr.Markdown("Input origin airport and destination airport from the dropdown boxes. Also input the scheduled departure time and scheduled arrival time")
    gr.Markdown("The scheduled departure time should be within one hour from now since live weather data for the airports will be fetched")
    
    with gr.Row():
        output = gr.Dataframe(headers=["Origin Airport", "Destination Airport", "Scheduled Departure", "Scheduled Arrival", "Predicted Departure Delay"],
                                    row_count=3,col_count=5,type="pandas",label="Predicted Departure Delay")
    with gr.Column():
        origin_dropdown = gr.Dropdown(choices=airports, label="Origin Airport")
        destination_dropdown = gr.Dropdown(choices=airports, label="Destination Airport")
        scheduled_dep_time_text = gr.Textbox(type="text", label="Enter scheduled Departure time in 24-hour format HH:MM(eg. 17:59)")
        scheduled_arr_time_text = gr.Textbox(type="text", label="Enter scheduled Arrival time in 24-hour format HH:MM (eg. 20:59)")

    with gr.Row():
        submit_button = gr.Button("Predict Departure Delay")
    
    
        
    submit_button.click(predict_delay, inputs=[origin_dropdown, destination_dropdown, scheduled_dep_time_text, scheduled_arr_time_text], outputs=output)

demo.launch()
"""
iface = gr.Interface(
    fn=predict_delay,
    inputs=[
        gr.inputs.Dropdown(choices=airports, label="Origin Airport"),
        gr.inputs.Dropdown(choices=airports, label="Destination Airport"),
        gr.inputs.Textbox(type="text", label="Enter scheduled Departure time in 24-hour format HH:MM(eg. 17:59)"),
        gr.inputs.Textbox(type="text", label="Enter scheduled Arrival time in 24-hour format HH:MM (eg. 20:59)"),
    ],
    outputs=gr.outputs.Dataframe(headers=["origin airport", "destination airport", "scheduled departure time","scheduled arrival time","predicted departure delay"],
                                type="pandas",label="Predicted Departure Delay"),
    #outputs=gr.outputs.Textbox(label="Predicted Departure Delay"),
    layout="vertical"
)

# Launch the Gradio app
iface.launch()


with gr.Row():
        origin_dropdown = gr.Dropdown(choices=airports, label="Origin Airport")
        destination_dropdown = gr.Dropdown(choices=airports, label="Destination Airport")
        scheduled_dep_time_text = gr.Textbox(type="text", label="Enter scheduled Departure time in 24-hour format HH:MM(eg. 17:59)")
        scheduled_arr_time_text = gr.Textbox(type="text", label="Enter scheduled Arrival time in 24-hour format HH:MM (eg. 20:59)")
"""


