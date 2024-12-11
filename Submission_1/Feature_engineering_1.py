import pandas as pd
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"


def replace_log_bike_count(X, counter_names, start_date, end_date, target_months):

    """
    For a given set of counter and a specific date range (when the counter was broken), replaces the 'log_bike_count' values 
    with the hourly mean values computed with specific target months.
    
    Parameters:
        X (DataFrame): The input Dataframe.
        start_date (datetime.date): Start of the date range where replacement occurs.
        end_date (datetime.date): End of the date range where replacement occurs.
        target_months (list): List of months numbers to use for hourly mean computation.
    
    Returns:
        X (DataFrame): The modified DataFrame with updated 'log_bike_count' values.
    """
    
    if len(counter_names) < 1:
        raise ValueError("You have to give at least one counter name.")

    X["date_only"] = X["date"].dt.date
    X["month"] = X["date"].dt.month
    X["hour"] = X["date"].dt.hour

    for counter_name in counter_names:
        data_filtered = X[X["counter_name"] == counter_name]

        seasonal_data = data_filtered[
            (
                (data_filtered["date_only"] < start_date)
                | (data_filtered["date_only"] > end_date)
            )
            & (data_filtered["month"].isin(target_months))
        ]

        mean_seasonal_per_hour = seasonal_data.groupby("hour")["log_bike_count"].mean()

        X.loc[
            (X["counter_name"] == counter_name)
            & (X["date_only"] >= start_date)
            & (X["date_only"] <= end_date),
            "log_bike_count",
        ] = X.loc[
            (X["counter_name"] == counter_name)
            & (X["date_only"] >= start_date)
            & (X["date_only"] <= end_date),
            "hour",
        ].map(
            mean_seasonal_per_hour
        )

    return X.drop(["date_only", "month", "hour"], axis=1)


def replace_broken_counters(X):
    """
    Replaces 'log_bike_count' values for 3 specific counters and date ranges
    using the `replace_log_bike_count` function with predefined parameters for each counter.

    Parameters:
        X (DataFrame): The input DataFrame

    Returns:
        X (DataFrame): The modified DataFrame with updated 'log_bike_count' values for broken counters.
    """
    X = replace_log_bike_count(
        X,
        ["20 Avenue de Clichy NO-SE", "20 Avenue de Clichy SE-NO"],
        start_date=pd.Timestamp("2021-04-09").date(),
        end_date=pd.Timestamp("2021-07-21").date(),
        target_months=np.arange(1, 13),
    )
    X = replace_log_bike_count(
        X,
        ["152 boulevard du Montparnasse O-E", "152 boulevard du Montparnasse E-O"],
        start_date=pd.Timestamp("2021-01-25").date(),
        end_date=pd.Timestamp("2021-02-24").date(),
        target_months=[11, 12, 1, 2, 3],
    )
    X = replace_log_bike_count(
        X,
        ["Voie Georges Pompidou NE-SO", "Voie Georges Pompidou SO-NE"],
        start_date=pd.Timestamp("2021-02-01").date(),
        end_date=pd.Timestamp("2021-02-15").date(),
        target_months=[1, 2],
    )
    return X


def get_train_data(path="data/train.parquet"):
        """
    Loads the training data and replace broken counters values before spliting the data.

    Parameters:
        path (str): The file path to the training data in Parquet format. Default is "data/train.parquet".

    Returns:
        tuple:
            - X_df (DataFrame): The feature DataFrame.
            - y_array (numpy array): The target variable values.
    """
    data = pd.read_parquet(path)
    data = data.sort_values(["date", "counter_name"])
    data = replace_broken_counters(data)
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def _encode_columns(X):
    """
    Gets rid of redundant columns.

    Parameters:
        X (DataFrame): The input DataFrame containing various columns.

    Returns:
        X (DataFrame): The modified DataFrame with specified columns dropped.
    """
    columns_to_drop = [
        "counter_id",
        "site_id",
        "site_name",
        "coordinates",
        "counter_technical_id"
    ]
    X = X.drop(columns=columns_to_drop, axis=1)
    return X


def _encode_dates(X):
    """
    Encodes date-related features from the 'date' column in the input DataFrame.

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame with new date-related features.
    """
    X = X.copy()

    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "week_number"] = X["date"].dt.isocalendar().week
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X.loc[:, "dayofyear"] = X["date"].dt.dayofyear

    return X


def get_time_of_day(X):
    """
    Adds a 'time_of_day' column to the DataFrame, categorizing each hour into predefined time segments.
    
    The categories are:
        - 1: Early Morning (4 AM to 6 AM)
        - 2: Morning (7 AM to 9 AM)
        - 3: Late Morning (10 AM to 12 PM)
        - 4: Afternoon (1 PM to 5 PM)
        - 5: Evening (6 PM to 10 PM)
        - 6: Night (11 PM to 3 AM)

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame with an additional 'time_of_day' column.
    """
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


def get_season(X):
    """
    Adds a 'season' column to the DataFrame, categorizing each date into one of four seasons.

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame with an additional 'season' column.
    """
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
    """
    Merges external weather data with the input DataFrame. Note that this data comes from a cleaned 
    version of "external_data" called "external_data_cleaned".

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame with external data merged. Selected features from external data includes:
            - 'pres': Pressure
            - 'u': Wind speed
            - 'tend': Pressure Tendency
            - 'ww': Current weather (code WMO 4677)
            - 'rr6', 'rr12', 'rr24': Precipitation in the last 6, 12, and 24 hours
            - 'etat_sol': Soil condition
            - 'ht_neige': Snow height
            - 'n': Cloud cover
            - 't': Temperature
            - 'td': Dew point temperature
            - 'tend24': 24-hour temperature trend
    """
    df_ext = pd.read_csv(
        "external_data/external_data_cleaned.csv", parse_dates=["date"]
    )

    X = X.copy()

    X["date"] = X["date"].astype("datetime64[ns]")
    df_ext["date"] = df_ext["date"].astype("datetime64[ns]")

    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[
            [
                "date",
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
            ]
        ].sort_values("date"),
        on="date",
    )

    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def _add_holiday(X):
    """
    Adds holiday-related features to the input DataFrame, including bank holidays and school vacations.
    These new features are:
        - 'is_bank_holiday': Indicates if the date is a bank holiday (1 for holiday, 0 otherwise).
        - 'is_holidays': Indicates if the date falls within Parisian school vacation periods (1 for vacation, 0 otherwise).

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame with added holiday-related features.
    """
    link_bank_holiday = "external_data/jours_feries_metropole.csv"
    # https://www.data.gouv.fr/fr/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5

    link_vacance_scolaire = "external_data/fr-en-calendrier-scolaire.csv"
    # https://www.data.gouv.fr/fr/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5

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

    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def _add_covid(X):
    """
    Adds a 'is_lockdown' column, indicating whether a given date falls within a COVID-19 lockdown period
    (1 for lockdown period, 0 otherwise).

    Lockdown periods:
        - 1st lockdown: October 30, 2020, to December 15, 2020.
        - 2nd lockdown: April 3, 2021, to May 9, 2021.

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        X (DataFrame): The modified DataFrame.
    """
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


def _add_arrondissement_with_geopandas(X):
    """
    Adds a 'district' column, indicating the Paris arrondissement where each coordinate 
    point (latitude and longitude) is located. If a point liest within Paris, the corresponding district code
    is added, otherwise, it is assigned a default value of 21.

    Parameters:
        X (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The modified DataFrame with an additional 'district' column.
    """
    arrondissements = gpd.read_file("external_data/arrondissements.shp")
    # https://opendata.paris.fr/explore/dataset/arrondissements/export/?disjunctive.c_ar&disjunctive.c_arinsee&disjunctive.l_ar&location=13,48.85156,2.32327

    X = X.copy()
    X["geometry"] = X.apply(
        lambda row: Point(row["longitude"], row["latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(X, geometry="geometry", crs=arrondissements.crs)

    merged = gpd.sjoin(gdf, arrondissements, how="left", predicate="within")

    X["district"] = merged["c_ar"].fillna(21).astype(int)

    return X.drop("geometry", axis=1)


# Finally erase date column (date is useful for the other functions)
def erase_date(X):
    X = X.copy()
    return X.drop(["date"], axis=1)
