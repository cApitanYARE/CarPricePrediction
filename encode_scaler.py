from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def EncodeAndScaler(X,y=None, mode="train"):
    X_encode = X.copy()

    numerical_cols = ['age', 'milage_per_year','hp',"milage","accident"]
    target_encode_cols = ["brand", "fuel_type","transmission"]

    categorical_cols = (
        X_encode.select_dtypes(include="object").columns.drop(target_encode_cols, errors="ignore")
    )

    if mode == "train":
        encoders = {}
        target_maps = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_encode[col] = le.fit_transform(X_encode[col])
            encoders[col] = le

        for col in target_encode_cols:
            te_map = y.groupby(X_encode[col]).mean()
            X_encode[col] = X_encode[col].map(te_map)
            target_maps[col] = te_map

        scaler = StandardScaler()
        X_encode[numerical_cols] = scaler.fit_transform(X_encode[numerical_cols])

        joblib.dump(encoders, "pkl/label_encoders.pkl")
        joblib.dump(target_maps, "pkl/target_maps.pkl")
        joblib.dump(scaler, "pkl/scaler.pkl")
        joblib.dump(y.mean(), "pkl/target_global_mean.pkl")


    else:  # predict
        encoders = joblib.load("pkl/label_encoders.pkl")
        target_maps = joblib.load("pkl/target_maps.pkl")
        scaler = joblib.load("pkl/scaler.pkl")
        global_mean = joblib.load("pkl/target_global_mean.pkl")

        for col, le in encoders.items():
            X_encode[col] = le.transform(X_encode[col])

        for col, te_map in target_maps.items():
            X_encode[col] = X_encode[col].map(te_map).fillna(global_mean)

        X_encode[numerical_cols] = scaler.transform(X_encode[numerical_cols])

    return X_encode
