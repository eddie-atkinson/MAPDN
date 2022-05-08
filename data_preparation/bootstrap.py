from pathlib import Path
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
from multiprocessing.pool import ThreadPool
from scipy import io, stats
import math
from typing import Tuple, Any
from datetime import datetime, timedelta
import random

INPUT_DATA_PATH = Path(".")

# Used for creating test and training sets later
# Dates sourced from: https://www.calendardate.com/year2018.php
SPRING_START = datetime(2018, 3, 20, 0, 0, 0)
SPRING_END = datetime(2018, 6, 20, 0, 0, 0)

SUMMER_START = datetime(2018, 6, 21, 0, 0)
SUMMER_END = datetime(2018, 9, 21, 0, 0)

AUTUMN_START = datetime(2018, 9, 22, 0, 0)
AUTUMN_END = datetime(2018, 12, 20, 0, 0)

START_OF_YEAR = datetime(2018, 1, 1, 0, 0)
WINTER_START = datetime(2018, 12, 21, 0, 0)
END_OF_YEAR = datetime(2018, 12, 31, 0, 0)


FREQ = "90S"
avg_pf = 0.9918161559771888

MIN_DATE = datetime(2018, 1, 1, 0, 0, 0)
MAX_DATE = datetime(2019, 1, 1, 0, 0, 0)
AVG_SOLAR_SYSTEM_KW = 6.9


def add_missing_periods(df, freq=FREQ):
    date_range = pd.date_range(MIN_DATE, MAX_DATE, freq=freq)
    df = df.reindex(date_range, fill_value=np.nan)
    df["datetime"] = df.index
    return df


def get_na_proportion(df) -> Any:
    solar_proportion = df["solar"].isna().sum() / len(df["solar"])
    use_proportion = df["use"].isna().sum() / len(df["use"])
    return {"solar": solar_proportion, "use": use_proportion}


def remove_outliers(df, column, sigma_threshold=5):
    outliers = np.abs(stats.zscore(df[column], nan_policy="omit")) > sigma_threshold
    df.loc[outliers, column] = np.nan
    # Only fill at max 5 intervals forward
    df[column] = df[column].fillna(method="ffill", limit=5)


def get_replacement_value(df, index, column):
    min_date = np.min(df["datetime"]).to_pydatetime()
    max_date = np.max(df["datetime"]).to_pydatetime()

    row = df.loc[index]

    row_datetime = row["datetime"].to_pydatetime()

    first_relevant_datetime = min_date.replace(
        hour=row_datetime.hour,
        minute=row_datetime.minute,
        second=row_datetime.second,
    )
    last_relevant_datetime = max_date.replace(
        hour=row_datetime.hour,
        minute=row_datetime.minute,
        second=row_datetime.second,
    )

    forward_dt = row_datetime
    back_dt = row_datetime

    replacement_val = np.nan

    while forward_dt <= last_relevant_datetime or back_dt >= first_relevant_datetime:
        if back_dt in df.index and not np.isnan(df.loc[back_dt, column]):
            replacement_val = df.loc[back_dt, column]
            break

        if forward_dt in df.index and not np.isnan(df.loc[forward_dt, column]):
            replacement_val = df.loc[forward_dt, column]
            break

        forward_dt += timedelta(days=1)
        back_dt -= timedelta(days=1)

    if np.isnan(replacement_val):
        raise ValueError(f"Couldn't find replacement values for date: {index}")

    return replacement_val


def replace_missing_values(df, column):
    indices_to_replace = df.loc[df[column].isna(), column].index
    replacement_values = {}
    # This is O(N^2) in the worst case (which could be pretty bad)
    # Fortunately for us our data is pretty complete
    for index in indices_to_replace:
        replacement_val = get_replacement_value(df, index, column)
        replacement_values[index] = replacement_val
    df[column] = df[column].fillna(value=replacement_values)


def normalise_solar(df):
    # Remove all values less than 0
    # Parasitics are important, but not *that* important
    negative_vals = df["solar"] < 0
    df.loc[negative_vals, "solar"] = 0
    abs_solar = np.abs(df["solar"])
    max_val = np.max(abs_solar)
    # Need to produce outputs in MW not kW
    df["solar"] = (df["solar"] / max_val) * AVG_SOLAR_SYSTEM_KW / 1000


def add_reactive(df):
    power_factor = pd.Series([avg_pf] * len(df), index=df.index)
    random_perturbation = np.random.uniform(0.9, 1.0, size=len(df))
    power_factor *= random_perturbation
    real_power = df["use"]
    apparent_power = real_power / power_factor
    # Multiplication is faster than exponentiation here
    s_squared = apparent_power * apparent_power
    p_squared = real_power * real_power
    q_squared = s_squared - p_squared
    df["use_reactive"] = np.sqrt(q_squared)
    df.rename(columns={"use": "use_active"}, inplace=True)


def process_site_data(file_path) -> Tuple[pd.DataFrame, Any]:
    df = pd.read_csv(
        file_path,
        parse_dates=["datetime"],
        dtype={
            "use": float,
            "solar": float,
        },
    )
    df = df.set_index(df["datetime"])
    df = add_missing_periods(df)

    na_proportions = get_na_proportion(df)
    site_has_solar = na_proportions["solar"] != 1.0

    remove_outliers(df, "solar")
    remove_outliers(df, "use")

    if site_has_solar:
        replace_missing_values(df, "solar")
        normalise_solar(df)

    replace_missing_values(df, "use")
    # Convert use from kW to MW for pandapower profiles
    df["use"] = df["use"] / 1000

    add_reactive(df)
    df = df.reset_index(drop=True)
    return df, na_proportions


household_na_proportions = []
household_index_map = []


def create_agg_profiles(active: pd.DataFrame, reactive: pd.DataFrame, pv: pd.DataFrame):
    load_index = 0
    solar_index = 0
    for file_path in INPUT_DATA_PATH.iterdir():
        site_id = file_path.stem.split("-")[0]
        if not str(file_path).endswith("load-pv-reduced.csv"):
            continue

        df, na_proportions = process_site_data(file_path)

        site_has_solar = na_proportions["solar"] != 1.0

        reactive[load_index] = df["use_reactive"]
        active[load_index] = df["use_active"]

        if site_has_solar:
            pv[solar_index] = df["solar"]

        index_map = {"household_id": site_id, "load_index": load_index}

        load_index += 1
        if site_has_solar:
            index_map["solar_index"] = solar_index
            solar_index += 1

        household_index_map.append(index_map)
        household_na_proportions.append(
            {
                "household_id": site_id,
                "solar_na": na_proportions["solar"],
                "use_na": na_proportions["use"],
            }
        )


def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def get_days_data(df, date, column):
    lower_datetime = pd.to_datetime(date)
    upper_datetime = lower_datetime.replace(hour=23, minute=59, second=59)
    mask = (df.index >= lower_datetime) & (df.index < upper_datetime)
    return df.loc[mask, column]


def get_bootstrap_samples(df, n_profiles):
    df = df.set_index("datetime")
    profiles_to_sample_from = [col for col in df.columns if is_int(col)]
    samples_to_take = []
    dates_to_sample = sorted(df.index.map(pd.Timestamp.date).unique())
    for _ in range(n_profiles):
        sample_columns = random.choices(profiles_to_sample_from, k=len(dates_to_sample))
        samples_to_take.append(list(zip(dates_to_sample, sample_columns)))
    df = df.reset_index(drop=True)
    return samples_to_take


def bootstrap_samples(df, samples_to_take, first_index):
    df = df.set_index("datetime")
    idx = first_index
    for sample in samples_to_take:
        data_sample = None
        for date, sample_column in sample:
            data = get_days_data(df, date, sample_column)
            if data_sample is None:
                data_sample = data
            else:
                data_sample = pd.concat([data_sample, data])
        data_sample = data_sample.rename(idx)
        df = df.join(data_sample)
        idx += 1
    return df


def get_days_data(df, date, column):
    lower_datetime = pd.to_datetime(date)
    upper_datetime = lower_datetime.replace(hour=23, minute=59, second=59)
    mask = (df.index >= lower_datetime) & (df.index < upper_datetime)
    return df.loc[mask, column]


def get_dates_in_range(*args):
    if len(args) % 2 != 0 or not len(args):
        raise ValueError("Date ranges must be specified in pairs")
    date_range = []
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
    for start, end in pairs:
        pair_range = list(pd.date_range(start, end, freq="1D"))
        date_range += pair_range

    return date_range


def sample_n_from_each_collection(collections, n):
    samples = []
    for collection in collections:
        samples += random.sample(collection, n)

    return samples


def create_test_train_split(df, test_dates):
    train_df = df
    test_df = train_df.iloc[0]
    return test_df, train_df


# Set random seed for pertubing active power
random.seed(42)
reactive = pd.DataFrame()
active = pd.DataFrame()
pv = pd.DataFrame()


date_range = pd.date_range(MIN_DATE, MAX_DATE, freq=FREQ)
reactive["datetime"] = date_range
active["datetime"] = date_range
pv["datetime"] = date_range

# Pretty slow again, sorry
create_agg_profiles(active, reactive, pv)


# We need to keep track of which household is mapped to which index in the final data set
household_map_df = pd.DataFrame(household_index_map)
na_proportion_df = pd.DataFrame(household_na_proportions)

household_map_df.to_csv(INPUT_DATA_PATH / "household_index_map.csv", index=False)
na_proportion_df.to_csv(INPUT_DATA_PATH / "na_proportion.csv", index=False)


# Set random seed for bootstrapping samples
random.seed(42)
load_bootstrap_samples = get_bootstrap_samples(reactive, 84)
reactive_df = bootstrap_samples(reactive, load_bootstrap_samples, 25)
active_df = bootstrap_samples(active, load_bootstrap_samples, 25)

pv_bootstrap_samples = get_bootstrap_samples(pv, 91)
pv_df = bootstrap_samples(pv, pv_bootstrap_samples, 19)


spring_dates = get_dates_in_range(SPRING_START, SPRING_END)
summer_dates = get_dates_in_range(SUMMER_START, SUMMER_END)
autumn_dates = get_dates_in_range(AUTUMN_START, AUTUMN_END)
winter_dates = get_dates_in_range(
    START_OF_YEAR, SPRING_START, WINTER_START, END_OF_YEAR
)

test_dates = sample_n_from_each_collection(
    [spring_dates, summer_dates, autumn_dates, winter_dates], 14
)

test_reactive_df, train_reactive_df = create_test_train_split(reactive_df, test_dates)
test_active_df, train_active_df = create_test_train_split(active_df, test_dates)
test_pv_df, train_pv_df = create_test_train_split(pv_df, test_dates)


# Reset the indices before saving
test_reactive_df = test_reactive_df.reset_index()
train_reactive_df = train_reactive_df.reset_index()

test_active_df = test_active_df.reset_index()
train_active_df = train_active_df.reset_index()

test_pv_df = test_pv_df.reset_index()
train_pv_df = train_pv_df.reset_index()


test_reactive_df.to_csv(INPUT_DATA_PATH / "test_reactive.csv", index=False)
train_reactive_df.to_csv(INPUT_DATA_PATH / "train_reactive.csv", index=False)

test_active_df.to_csv(INPUT_DATA_PATH / "test_active.csv", index=False)
train_active_df.to_csv(INPUT_DATA_PATH / "train_active.csv", index=False)

test_pv_df.to_csv(INPUT_DATA_PATH / "test_pv.csv", index=False)
train_pv_df.to_csv(INPUT_DATA_PATH / "train_pv.csv", index=False)
