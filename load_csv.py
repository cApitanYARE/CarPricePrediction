import numpy as np
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from encode_scaler import EncodeAndScaler

from datetime import date

def classify_transmission(transmission):
    if 'M/T' in transmission or 'MT' in transmission or 'MANUAL' in transmission:
        return 'M/T'
    elif 'A/T' in transmission or 'AT' in transmission or 'AUTOMATIC' in transmission:
        return 'A/T'
    else:
        return 'OTHER'

#load a csv
data = pd.read_csv("csv/currUsedCars.csv")

data["price"] = data["price"].replace('[$,]', '', regex=True).astype(float)

#fill in the blanks and create new features
data["brand"] = data["brand"].str.lower()

data["age"] = date.today().year - data["model_year"]

data["milage"] = data["milage"].str.replace(r'[^0-9]', '', regex=True).astype(int)
data["milage_per_year"] = data['milage'] / data['age']

data["fuel_type"] = data['fuel_type'].str.strip().str.lower()
data['fuel_type'] = data['fuel_type'].replace({'plug-in hybrid': 'hybrid','not supported':'other','–':'other'})
data['fuel_type'] = data['fuel_type'].fillna('other')


data['hp'] = data['engine'].astype(str).str.extract(r'(\d+(?:\.\d+)?)\s*HP', expand=False).astype(float)
data['hp'] = data.groupby('brand')['hp'].fillna(data['hp'].mean()).astype(int)


data['engine_capacity'] = data['engine'].astype(str).str.extract(r'(\d+(?:\.\d+)?)\s*L', expand=False).astype(float)
data['engine_capacity'] = data['engine_capacity'].fillna(data['engine_capacity'].mean())

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.boxplot(x=data['milage'], ax=axes[0, 0])
axes[0, 0].set_title("Boxplot of milage")

sns.countplot(x=data['fuel_type'], ax=axes[0, 1])
axes[0, 1].set_title("Countplot of fuel_type")

sns.boxplot(x=data['hp'], ax=axes[1, 0])
axes[1, 0].set_title("Boxplot of Horsepower")

sns.boxplot(x=data['engine_capacity'], ax=axes[1, 1])
axes[1, 1].set_title("Boxplot of engine_capacity")

plt.tight_layout()
plt.show()

data['v_engine'] = data['engine'].str.contains(r'V\d+', case=False, na=False)

data['turbo'] = data['engine'].str.contains('twin turbo', case=False, na=False)

data['transmission'] = data['transmission'].apply(classify_transmission)

data['accident'] = data['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

data['clean_title'] = data['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)


#Select X and y
X = data.drop(columns=["price","model","int_col","ext_col","model_year","engine"])
y = np.log1p(data["price"])

#Encode and scaler
X_encode = EncodeAndScaler(X,y, mode="train")

X_encode.to_csv("csv/afterPreprocessing.csv", index=False)

#Split data for train and test
X_train, X_test, y_train, y_test = train_test_split(X_encode,y, test_size=0.2,random_state=42)

#Select model for predict
model = RandomForestRegressor(random_state=42)
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

