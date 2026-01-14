import numpy as np
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from encode_scaler import EncodeAndScaler

from datetime import date

#load a csv
data = pd.read_csv("csv/currUsedCars.csv")

#fill in the blanks
data["fuel_type"] = data["fuel_type"].fillna("Electric")
data["accident"] = np.where(
    data["accident"] == "At least 1 accident or damage reported",
    "Yes",
    "No"
)
data["clean_title"] = data["clean_title"].fillna("No")

#erased the symbol $ and "mi" and prepared features for training
def clean_engine(x):
    x = str(x)
    # search num with L
    match_with_L = re.search(r'(\d+\.?\d*)L', x)
    if match_with_L:
        return match_with_L.group(1)

    match_number = re.search(r'\d+\.?\d*', x)
    if match_number:
        return match_number.group(0)
    return None  # якщо взагалі немає чисел
data["engine"] = data["engine"].apply(clean_engine)
valid_numbers = data["engine"].dropna()
most_common = Counter(valid_numbers).most_common(1)[0][0]
data["engine"] = data["engine"].fillna(most_common)

#prepared for training
data["milage"] = data["milage"].str.replace(r'[^0-9]', '', regex=True).astype(float)
data["price"] = data["price"].replace('[$,]', '', regex=True).astype(float)
data["age"] = date.today().year - data["model_year"]

#Select X and y
X = data.drop(columns=["price","model","int_col","ext_col","transmission","model_year","clean_title"])
y = np.log1p(data["price"])

#Encode and scaler
X_encode = EncodeAndScaler(X,y, mode="train")

X_encode.to_csv("csv/afterPreprocessing.csv", index=False)

#Split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X_encode,y, test_size=0.2,random_state=42)

#Select model for predict
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
#Teaching model
model.fit(X_train, y_train)

joblib.dump(model, "pkl/CarPriceModel.pkl")

y_pred = model.predict(X_test)

#regression model quality metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 score: {r2:.2f}")

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual price (log)")
plt.ylabel("Predicted price (log)")
plt.title("Actual vs Predicted Prices")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])

plt.show()

X_plot = X_encode.copy()
X_plot['price_log'] = y

corr_with_price = X_plot.corr()['price_log'].sort_values(ascending=False)
print(corr_with_price)

