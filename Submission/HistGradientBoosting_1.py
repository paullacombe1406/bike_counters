from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

import Feature_engineering

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"


X_train, y_train = Feature_engineering.get_train_data(
    path="../input/msdb-2024/train.parquet"
)

X_test = pd.read_parquet("../input/msdb-2024/final_test.parquet")


columns_encoder = FunctionTransformer(Feature_engineering._encode_columns)

oneHot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
oneHot_cols = ["counter_id", "longitude", "latitude", "counter_count"]

ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=22)
ord_cols = ["counter_installation_date"]

date_encoder = FunctionTransformer(Feature_engineering._encode_dates)
date_cols = ["year", "month", "day", "weekday", "hour"]

covid_encoder = FunctionTransformer(Feature_engineering._add_covid)
covid_cols = ["is_lockdown"]

meteo_encoder = FunctionTransformer(Feature_engineering._merge_external_data)
meteo_cols = ["t", "ff", "u"]

holidays_encoder = FunctionTransformer(Feature_engineering._add_holiday)
holidays_cols = ["is_holidays", "is_bank_holiday"]

scaler = StandardScaler(with_mean=False)

preprocessor = ColumnTransformer(
    [
        (
            "date",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            date_cols,
        ),
        ("meteo", scaler, meteo_cols),
        ("cat onehot", oneHot_encoder, oneHot_cols),
        ("cat ordinal", make_pipeline(ord_encoder, scaler), ord_cols),
    ]
)

regressor = HistGradientBoostingRegressor()

pipe = make_pipeline(
    columns_encoder,
    meteo_encoder,
    covid_encoder,
    holidays_encoder,
    date_encoder,
    preprocessor,
    regressor,
)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
