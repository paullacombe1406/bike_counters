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

import feature_engineering_2

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"


X_train, y_train = feature_engineering_2.get_train_data(
    path="../input/msdb-2024/train.parquet"
)

X_test = pd.read_parquet("../input/msdb-2024/final_test.parquet")


columns_encoder = FunctionTransformer(feature_engineering_2._encode_columns)

date_encoder = FunctionTransformer(feature_engineering_2._encode_dates)

time_encoder = FunctionTransformer(feature_engineering_2.get_time_of_day)

season_encoder = FunctionTransformer(feature_engineering_2.get_season)

covid_encoder = FunctionTransformer(feature_engineering_2._add_covid)

meteo_encoder = FunctionTransformer(feature_engineering_2._merge_external_data)

holidays_encoder = FunctionTransformer(feature_engineering_2._add_holiday)

district_encoder = FunctionTransformer(
    feature_engineering_2._add_arrondissement_with_geopandas
)

erase_date = FunctionTransformer(feature_engineering_2.erase_date)

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
    max_depth=9,
    learning_rate=0.08982350073781493,
    n_estimators=394,
    subsample=0.6091504269638729,
    colsample_bytree=0.7564351155303641,
    min_child_weigh=6,
    gamma=8.981351819175658e-05,
    reg_alpha=9.853267873712797e-05,
    reg_lambda=5.602142068055845e-05,
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
