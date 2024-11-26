import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor

df_train = pd.read_parquet("..input/msdb-2024/train.parquet")
X_test = pd.read_parquet("..input/msdb-2024/final_test.parquet")

X_train = df_train.drop(columns="log_bike_count", axis=1)
y_train = df_train["log_bike_count"]

model = DummyRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
