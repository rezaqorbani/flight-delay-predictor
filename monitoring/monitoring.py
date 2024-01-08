# %%
import pandas as pd
import hopsworks
import joblib
import datetime
from datetime import datetime
import numpy as np

import dataframe_image as dfi
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# %%
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

# %%
project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

mr = project.get_model_registry()
model = mr.get_model("flight_delay_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/flight_delay_model.pkl")

feature_group = fs.get_feature_group(name="flight_data_v3")
feature_view = fs.get_feature_view(name="flight_data_v3")

# %% [markdown]
# ## Fitting the scaler to the training data

# %%
# fit the scaler
X_train, X_test, y_train, y_test = feature_view.get_train_test_split(training_dataset_version=3)
scaler = StandardScaler()
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_train_scaled = scaler.fit_transform(X_train_tensor)

# %% [markdown]
# ## Monitoring

# %%

batch_data = feature_view.get_batch_data()
batch_data = torch.tensor(batch_data.values, dtype=torch.float32)
batch_data = scaler.transform(batch_data)
batch_data = torch.tensor(batch_data, dtype=torch.float32)

preds = model(batch_data)

window_length = 5
latest_preds = preds[-window_length:]
latest_preds = latest_preds.detach().numpy()
latest_preds = latest_preds.ravel()
latest_pred = float(latest_preds[-1])


print("Delay_predicted: " + str(latest_pred))

df = feature_group.read()

latest_labels = df[-window_length:]["dep_delay_new"]
latest_labels = latest_labels.to_numpy()
latest_label = float(latest_labels[-1])

print("Delay actual: " + str(latest_label))

loss = mean_squared_error(latest_labels, latest_preds)
print("Running MSE (n = 5): " + str(loss))

print("Latest predictions:")
print(latest_preds)
print("Latest labels:")
print(latest_labels)

# %%
monitor_fg = fs.get_or_create_feature_group(
    name="flight_delay_predictions",
    version=1,
    primary_key=[
        "datetime",
    ],
    description="Flight delay Prediction/Outcome Monitoring",
)
now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
data = {
    "datetime": [now],
    "prediction": [latest_pred],
    "label": [latest_label],
    "mse": [loss],
}

monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df, write_options={"wait_for_job": True}) # set this to True if you want to run it faster (async) but you will not be able to run the next cell

# %% [markdown]
# ## Add to history

# %%
history_df = monitor_fg.read()
history_df = pd.concat([history_df, monitor_df])

df_recent = history_df.tail(5)
dfi.export(df_recent, "./recent_delay_performance.png", table_conversion="matplotlib")
dataset_api.upload("./recent_delay_performance.png", "Resources/images", overwrite=True)

