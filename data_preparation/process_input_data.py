from pathlib import Path
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
from multiprocessing.pool import ThreadPool, Pool
from scipy import io, stats
import math
from typing import Tuple, Any
from datetime import datetime, timedelta
import random


INPUT_DATA_PATH = Path("./input_data")

# Bit of a hack, but this is the result of computing the average power factor from the Adres data set
avg_pf = 0.9918161559771888

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


def get_closest_interval(dt):
    return dt.round(freq=FREQ)


def calculate_use(grid: str, solar: str, solar2: str):
    if not grid:
        return None
    grid_val = float(grid)
    solar_val = float(solar) if solar else 0
    solar2_val = float(solar2) if solar2 else 0

    # This may seem weird but it's explained here: https://docs.google.com/document/d/1_9H9N4cgKmJho7hK8nii6flIGKPycL7tlWEtd4UhVEQ/edit#
    return grid_val + solar_val + solar2_val


def calculate_net_solar(solar: str, solar2: str):
    if not solar and not solar2:
        return None
    solar_val = float(solar) if solar else 0
    solar2_val = float(solar) if solar2 else 0

    return solar_val + solar2_val


def downsample_data(file_path: Path, out_path: Path):
    print(f"processing {file_path}")
    data_periods = {}
    with open(file_path, "r") as infile:
        csv_reader = csv.DictReader(infile)
        for line in csv_reader:
            # The last three characters are TZ info, we will lose an hour's data every time the clocks change.
            # Fortunately, we don't really care about that
            localminute = line["localminute"][:-3]
            grid = line["grid"]
            solar = line["solar"]
            solar2 = line["solar2"]
            solar_val = calculate_net_solar(solar, solar2)
            use = calculate_use(grid, solar, solar2)
            dt = pd.to_datetime(localminute, format="%Y-%m-%d %H:%M:%S")

            interval = get_closest_interval(dt)
            if not data_periods.get(interval, None):
                data_periods[interval] = dict(
                    solar_sum=0, solar_count=0, use_sum=0, use_count=0
                )

            data = data_periods[interval]
            if solar_val is not None:
                data["solar_sum"] += solar_val
                data["solar_count"] += 1
            if use is not None:
                data["use_sum"] += use
                data["use_count"] += 1

    intervals = sorted(data_periods)
    with open(out_path, "w") as outfile:
        header = "datetime,solar,use\n"
        outfile.write(header)
        for interval in intervals:
            interval_data = data_periods[interval]
            iso_date = interval.isoformat()
            solar_count = interval_data["solar_count"]
            solar_sum = interval_data["solar_sum"]

            use_count = interval_data["use_count"]
            use_sum = interval_data["use_sum"]

            solar_val = ""
            use_val = ""
            if solar_count:
                solar_val = solar_sum / solar_count
            if use_count:
                use_val = use_sum / use_count
            outfile.write(f"{iso_date},{solar_val},{use_val}\n")


files_to_process = []
for file_path in INPUT_DATA_PATH.iterdir():
    if not str(file_path).endswith("load-pv.csv"):
        continue
    outfile_name = f"{file_path.stem}-reduced.csv"
    outfile_path = file_path.parent / outfile_name
    files_to_process.append((file_path, outfile_path))


def test(path, out_path):
    with open(out_path, "w") as f:
        print("hello world", file=f)


def main():
    with Pool() as p:
        p.starmap(test, files_to_process)


if __name__ == "__main__":
    main()
