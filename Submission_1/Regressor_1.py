import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

import feature_engineering_1

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"


X_train, y_train = feature_engineering_1.get_train_data(
    path="../input/msdb-2024/train.parquet"
)

X_test = pd.read_parquet("../input/msdb-2024/final_test.parquet")


columns_encoder = FunctionTransformer(feature_engineering_1._encode_columns)

date_encoder = FunctionTransformer(feature_engineering_1._encode_dates)

time_encoder = FunctionTransformer(feature_engineering_1.get_time_of_day)

season_encoder = FunctionTransformer(feature_engineering_1.get_season)

covid_encoder = FunctionTransformer(feature_engineering_1._add_covid)

meteo_encoder = FunctionTransformer(feature_engineering_1._merge_external_data)

holidays_encoder = FunctionTransformer(feature_engineering_1._add_holiday)

district_encoder = FunctionTransformer(
    feature_engineering_1._add_arrondissement_with_geopandas
)

erase_date = FunctionTransformer(feature_engineering_1.erase_date)

ordinal_cols = ["counter_installation_date"]
onehot_cols = ["counter_name"]
scale_cols = [
    "latitude",
    "longitude",
    "year",
    "month",
    "week_number",
    "day",
    "weekday",
    "hour",
    "dayofyear",
    "time_of_day",
    "season",
    "pres",
    "u",
    "tend",
    "ww",
    "rr6",
    "rr12",
    "rr24",
    "etat_sol",
    "ht_neige",
    "n",
    "t",
    "td",
    "tend24",
    "district",
]

scaler = StandardScaler()
onehot = OneHotEncoder(sparse_output=False)
ordinal = OrdinalEncoder()

preprocessor = ColumnTransformer(
    [
        ("num", scaler, scale_cols),
        ("onehot", onehot, onehot_cols),
        ("ordinal", ordinal, ordinal_cols),
    ]
)

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.03758411108052076,
    n_estimators=452,
    subsample=0.8146926142904702,
    colsample_bytree=0.838453719208161,
    min_child_weight=9,
    gamma=0.010218531432881407,
    reg_alpha=2.2994203367699492e-05,
    reg_lambda=2.0532974192471358e-05,
)

pipe = make_pipeline(
    columns_encoder,
    date_encoder,
    time_encoder,
    season_encoder,
    meteo_encoder,
    covid_encoder,
    holidays_encoder,
    district_encoder,
    erase_date,
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
