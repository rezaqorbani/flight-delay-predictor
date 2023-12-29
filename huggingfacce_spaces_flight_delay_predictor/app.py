import gradio as gr
import numpy as np
import requests
import pandas as pd
import hopsworks
import joblib
import torch
from torch import nn

project = hopsworks.login()
fs = project.get_feature_store()

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

mr = project.get_model_registry()
model = mr.get_model("fligt_delay_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/flight_delay_model.pkl")

# Function to fetch weather data from an API
def get_weather_data(airport_code):
    # Make API call to fetch weather data for the airport
    # Process and return weather data
    pass

# Function to fetch flight data from an API
def get_flight_data(origin, destination, date):
    # Make API call to fetch flight data based on origin, destination
    # Process and return flight data
    # We can get most data from user input airport codes. Mainly need CRS arr and CRS dep time
    pass

# Define the function to predict flight delay based on user inputs
def predict_delay(origin, destination):
    #test code to try running Gradio app
    date="2023-12-29"

    if origin == destination:
        return "Error: Origin and destination airports cannot be the same. Please select different airports."
    
    #Only keep airport codes
    origin=origin.split()[0]
    destination=destination.split()[0]


    
    return "Predicted delay for {} to {} on {}: {} minutes".format(origin, destination, date, "some_prediction")

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


# Create Gradio interface with dropdowns for airport selection
iface = gr.Interface(
    fn=predict_delay,
    inputs=[
        gr.inputs.Dropdown(choices=airports, label="Origin Airport"),
        gr.inputs.Dropdown(choices=airports, label="Destination Airport"),
    ],
    outputs=gr.outputs.Textbox(label="Predicted Departure Delay")
)

# Launch the Gradio app
iface.launch()
