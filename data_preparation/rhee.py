from pathlib import Path
from typing import Tuple, TypedDict
import pandas as pd
from scipy import stats
from datetime import datetime
import numpy as np
import random
import time


class NaProportion(TypedDict):
    solar: float
    use: float


MIN_DATE = datetime(2018, 1, 1, 0, 0, 0)
MAX_DATE = datetime(2018, 12, 31, 23, 59, 30)
AVG_SOLAR_SYSTEM_KW = 6.9

OUTPUT_DATA_PATH = Path("./output_data")

avg_pf = 0.9918161559771888


def add_missing_periods(df, freq="30S"):
    date_range = pd.date_range(MIN_DATE, MAX_DATE, freq=freq)
    df = df.reindex(date_range, fill_value=np.nan)
    df["datetime"] = df.index
    return df


def get_na_proportion(df) -> NaProportion:
    solar_proportion = df["solar"].isna().sum() / len(df["solar"])
    use_proportion = df["use"].isna().sum() / len(df["use"])
    return {"solar": solar_proportion, "use": use_proportion}


def remove_outliers(df, column, sigma_threshold=5):
    outliers = np.abs(stats.zscore(df[column], nan_policy="omit")) > sigma_threshold
    df.loc[outliers, column] = np.nan
    # Only fill at max 5 intervals forward
    df[column] = df[column].fillna(method="ffill", limit=5)


def sort_replacement_periods(current_datetime: datetime, a: datetime):
    return abs((current_datetime - a).total_seconds())


def get_possible_replacement_indices(df, index):
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

    possible_periods = pd.date_range(row_datetime, last_relevant_datetime, freq="1D")

    # Filter out any periods that could be used for replacement, but aren't in the dataset
    possible_periods = possible_periods[possible_periods.isin(df.index)]

    return sorted(
        possible_periods, key=lambda x: sort_replacement_periods(row_datetime, x)
    )


def replace_missing_values(df, column):
    indices_to_replace = df.loc[df[column].isna(), column].index
    replacement_values = {}
    # This is O(N^2) in the worst case (which could be pretty bad)
    # Fortunately for us our data is pretty complete
    for index in indices_to_replace:
        start_time = time.time()
        possible_replacement_indices = get_possible_replacement_indices(df, index)
        replacement_vals = df.loc[possible_replacement_indices, column].dropna()
        if not len(replacement_vals):
            raise ValueError(f"Couldn't find replacement values for date: {index}")

        replacement_val = replacement_vals[0]
        replacement_values[index] = replacement_val
        print("--- %s seconds ---" % (time.time() - start_time))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    df[column] = df[column].fillna(value=replacement_values)


def normalise_solar(df):
    # Remove all values less than 0
    # Parasitics are important, but not *that* important
    negative_vals = df["solar"] < 0
    df.loc[negative_vals, "solar"] = 0
    abs_solar = np.abs(df["solar"])
    max_val = np.max(abs_solar)
    df["solar"] = (df["solar"] / max_val) * AVG_SOLAR_SYSTEM_KW


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


# Set random seed for pertubing active power
random.seed(42)


reactive = pd.DataFrame()
active = pd.DataFrame()
pv = pd.DataFrame()


date_range = pd.date_range(MIN_DATE, MAX_DATE, freq="30S")
reactive["datetime"] = date_range
active["datetime"] = date_range
pv["datetime"] = date_range

household_na_proportions = []
household_index_map = []


def process_site_data(file_path) -> Tuple[pd.DataFrame, NaProportion]:
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

    add_reactive(df)
    df = df.reset_index(drop=True)
    return df, na_proportions


def create_agg_profiles():
    load_index = 0
    solar_index = 0
    for file_path in OUTPUT_DATA_PATH.iterdir():
        site_id = file_path.stem.split("-")[0]
        if not str(file_path).endswith("load-pv-reduced.csv"):
            continue

        print(site_id)
        df, na_proportions = process_site_data(file_path)
        print(na_proportions)

        site_has_solar = na_proportions["solar"] != 1.0

        reactive[load_index] = df["use_reactive"]
        active[load_index] = df["use_active"]

        if site_has_solar:
            pv[solar_index] = df["solar"]

        index_map = {"household_id": site_id, "load_index": load_index}

        print(f"load: {site_id} {'does'if site_has_solar else 'doesnt'} possess solar")
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


create_agg_profiles()
breakpoint()
