import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import holidays

df = pd.read_parquet("../input/msdb-2024/train.parquet")
X_test = pd.read_parquet("../input/msdb-2024/final_test.parquet")

french_holidays = holidays.France()
df["bank_holiday"] = df["date"].apply(lambda x: 1 if x in french_holidays else 0)

columns_to_drop = ["counter_name", "site_id", "site_name", "longitude", "latitude"]
df = df.drop(columns=columns_to_drop)

X_train = df.drop(columns=["log_bike_count", "bike_count"], axis=1)
y_train = df["log_bike_count"]


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

oneHot_encoder = OneHotEncoder(handle_unknown="ignore")
oneHot_cols = ["counter_id", "coordinates", "counter_technical_id"]

ord_encoder = OrdinalEncoder()
ord_cols = ["counter_installation_date"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat onehot", oneHot_encoder, oneHot_cols),
        ("cat ordinal", ord_encoder, ord_cols),
    ]
)

regressor = Ridge()

pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X_train, y_train)


y_pred = pipe.predict(X_test)

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
