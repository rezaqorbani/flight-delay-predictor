# %%
import pandas as pd
import hopsworks
import joblib
import datetime
from datetime import datetime

import dataframe_image as dfi
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


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

mr = project.get_model_registry()
model = mr.get_model("flight_delay_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/flight_delay_model.pkl")

batch_feature_view = fs.get_feature_view(name="flight_data_v2", version=1)

batch_feature_group = fs.get_feature_group(name="flight_data_v2", version=1)
query = batch_feature_group.select_all()
query_feature_view = fs.get_or_create_feature_view(
    name="flight_data_v2",
    version=1,
    description="Read from Flight Delay dataset",
    labels=["dep_delay_new"],
    query=query,
)

# %% [markdown]
# ## Fit the Scaler

# %%
X_train, X_test, y_train, y_test = query_feature_view.train_test_split(test_size=0.2)
scaler = StandardScaler()
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_train_scaled = scaler.fit_transform(X_train_tensor)

# %%
batch_data = batch_feature_view.get_batch_data()
batch_data = torch.tensor(batch_data.values, dtype=torch.float32)
batch_data = scaler.transform(batch_data)
batch_data = torch.tensor(batch_data, dtype=torch.float32)

y_pred = model(batch_data)

offset = 1
pred = y_pred[- offset]
pred = float(pred)

print("Delay_predicted: " + str(pred))
df = batch_feature_group.read()

label = df.iloc[-offset]["dep_delay_new"]
label = float(label)
print("Delay_actual: " + str(label))


loss = (pred - label) ** 2
print("MSE: " + str(loss))

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
    "prediction": [pred],
    "label": [label],
    "mse": [loss],
}

monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df, write_options={"wait_for_job": False}) # set this to True if you want to run it faster (async) but you will not be able to run the next cell


# %% [markdown]
# ## Read in the data

# %%
# history_df = monitor_fg.read()
# history_df = pd.concat([history_df, monitor_df])

# df_recent = history_df.tail(4)
# dfi.export(df_recent, "./df_recent.png", table_conversion="matplotlib")
# # dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

# predictions = history_df[["prediction"]]
# labels = history_df[["label"]]
# current_mse = history_df[["mse"]]


