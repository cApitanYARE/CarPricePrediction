import pandas as pd
import joblib
from encode_scaler import EncodeAndScaler
import numpy as np

from datetime import date

CarPriceModel = joblib.load("pkl/CarPriceModel.pkl")

fields = [
    ("brand", str, "Not specified"),
    ("engine", str, "Not specified"),
    ("fuel_type", str, "Not specified"),
    ("accident", str, "No"),
    ("milage", float, 0.0),
    ("model_year", int, 0),
]

input_dict = {}

#input data for variables
for name, dtype, default in fields:
    value = input(f"Enter {name.replace('_', ' ')}: ").strip()
    if value == "":
        value = default
    else:
        if dtype != str:
            try:
                value = dtype(value)
            except ValueError:
                print(f"Invalid input for {name}, using default {default}")
                value = default
    input_dict[name] = value

input_df = pd.DataFrame([input_dict])
input_df["age"] = date.today().year - input_df["model_year"]
del input_df["model_year"]

#Encode and scaler
X_encode = EncodeAndScaler(input_df, mode="predict")

model_features = list(CarPriceModel.feature_names_in_)
X_encode = X_encode.reindex(columns=model_features, fill_value=0)

predictionPrice = CarPriceModel.predict(X_encode)

price_dollars = np.expm1(predictionPrice)
print(price_dollars)

