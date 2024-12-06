from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_id"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def _encode_columns(X):
    columns_to_drop = [
        "counter_id",
        "site_id",
        "site_name",
        "coordinates",
        "counter_technical_id", "bike_count"
    ]
    X = X.drop(columns=columns_to_drop, axis=1)
    return X


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "week_number"] = X["date"].dt.isocalendar().week
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "dayofyear"] = X["date"].dt.dayofyear 

    # Finally we can drop the original columns from the dataframe
    return X

def get_time_of_day(X):
    def get_time(hour):
        if hour > 3 and hour <= 6:
            return 1
        if hour > 6 and hour <= 9:
            return 2
        elif hour > 9 and hour <= 12:
            return 3
        elif hour > 12 and hour <= 17:
            return 4
        elif hour > 17 and hour <= 22:
            return 5
        else:
            return 6
    X["time_of_day"] = X["hour"].apply(get_time)
    return X


def get_season(X)
    def season(date):
        if (date > datetime(2020, 9, 21)) & (date < datetime(2020, 12, 21)):
            return 1
        if (date > datetime(2020, 12, 20)) & (date < datetime(2021, 3, 20)):
            return 2
        if (date > datetime(2021, 3, 19)) & (date < datetime(2021, 6, 21)):
            return 3
        if ((date > datetime(2021, 6, 20)) & (date < datetime(2021, 9, 22))) | (
            (date > datetime(2020, 6, 19)) & (date < datetime(2020, 9, 22))
        ):
            return 4
    X["season"] = X["date"].apply(season)
    return X


def _merge_external_data(X):
    df_ext = pd.read_csv(
        "external_data/external_data_cleaned.csv", parse_dates=["date"]
    )

    X = X.copy()

    X["date"] = X["date"].astype("datetime64[ns]")
    df_ext["date"] = df_ext["date"].astype("datetime64[ns]")

    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "pres", "u", "tend", "ww", "rr6", "rr12", "rr24",
                "etat_sol", "ht_neige", "n", "t", "td", "tend24"]].sort_values("date"),
        on="date",
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def _add_holiday(X):
    link_bank_holiday = (
        "https://www.data.gouv.fr/fr/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5"
    )
    link_vacance_scolaire = (
        "https://www.data.gouv.fr/fr/datasets/r/9957d723-346e-4317-8cb3-293c94e19b2d"
    )
    start_calendar = np.min(X["date"])
    end_calendar = np.max(X["date"])
    df_bank_holiday = pd.read_csv(link_bank_holiday)
    df_bank_holiday["date"] = pd.to_datetime(df_bank_holiday["date"]).dt.date

    df_holidays = pd.read_csv(link_vacance_scolaire, sep=";")
    df_holidays = df_holidays[df_holidays["Zones"].isin(["Zone C"])]
    df_holidays = df_holidays.drop_duplicates(
        subset=["Zones", "Description", "annee_scolaire"]
    )

    df_holidays["Date de début"] = pd.to_datetime(
        df_holidays["Date de début"].str[0:10]
    )
    df_holidays["Date de fin"] = pd.to_datetime(df_holidays["Date de fin"].str[0:10])

    df_calendar = pd.DataFrame(
        index=pd.date_range(start=start_calendar, end=end_calendar)
    )
    df_calendar["date"] = df_calendar.index.date
    df_calendar["is_bank_holiday"] = 0
    df_calendar.loc[
        df_calendar["date"].isin(df_bank_holiday["date"]), "is_bank_holiday"
    ] = 1

    df_calendar["is_holidays"] = 0
    for _, row in df_holidays.iterrows():
        date_start = row["Date de début"].date()
        date_end = row["Date de fin"].date()
        df_calendar.loc[
            (df_calendar["date"] >= date_start) & (df_calendar["date"] <= date_end),
            "is_holidays",
        ] = 1

    X = X.copy()
    X["date"] = X["date"].astype("datetime64[ns]")
    df_calendar["date"] = df_calendar["date"].astype("datetime64[ns]")

    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_calendar[["date", "is_holidays", "is_bank_holiday"]].sort_values("date"),
        on="date",
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def _add_covid(X):
    start_calendar = np.min(X["date"])
    end_calendar = np.max(X["date"])

    df_calendar = pd.DataFrame(
        index=pd.date_range(start=start_calendar, end=end_calendar)
    )

    df_calendar["date"] = df_calendar.index.date

    df_calendar["is_lockdown"] = 0

    date_start_1 = pd.to_datetime("2020-10-30").date()
    date_end_1 = pd.to_datetime("2020-12-15").date()

    df_calendar.loc[
        (df_calendar["date"] >= date_start_1) & (df_calendar["date"] <= date_end_1),
        "is_lockdown",
    ] = 1

    date_start_2 = pd.to_datetime("2021-04-03").date()
    date_end_2 = pd.to_datetime("2021-05-09").date()

    df_calendar.loc[
        (df_calendar["date"] >= date_start_2) & (df_calendar["date"] <= date_end_2),
        "is_lockdown",
    ] = 1

    X = X.copy()
    X["date"] = X["date"].astype("datetime64[ns]")
    df_calendar["date"] = df_calendar["date"].astype("datetime64[ns]")

    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_calendar[["date", "is_lockdown"]].sort_values("date"),
        on="date",
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def train_test_split_temporal(X, y, delta_threshold="30 days"):

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

def _add_arrondissement_with_geopandas(X, shapefile_path="external_data/arrondissements.shp"):
    arrondissements = gpd.read_file(shapefile_path)

    X = X.copy()
    X["geometry"] = X.apply(
        lambda row: Point(row["longitude"], row["latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(X, geometry="geometry", crs=arrondissements.crs)

    merged = gpd.sjoin(gdf, arrondissements, how="left", predicate="within")

    X["district"] = merged["c_ar"].fillna(21).astype(int)

    return X


def erase_date(X):
    X = X.copy()
    return X.drop("date", axis=1)
















