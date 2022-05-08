import pandapower as pp
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool

import pandapower.control as control
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData

N_CORES = 36
NETWORK_NAME = "J"
INPUT_PATH = Path(f"./output_network_models/{NETWORK_NAME}")
OUTPUT_PATH = Path("./")
SIMULATION_RESULTS_PATH = Path(f"./simulation_results/{NETWORK_NAME}")
LOAD_DATA_PATH = Path("./opf_data")
LOAD_SCALE = 0.7

reactive_df = pd.read_csv(
    LOAD_DATA_PATH / "train_reactive.csv", parse_dates=["datetime"]
)
active_df = pd.read_csv(LOAD_DATA_PATH / "train_active.csv", parse_dates=["datetime"])
solar_df = pd.read_csv(LOAD_DATA_PATH / "pv_p_mw.csv")
solar_reactive_df = pd.read_csv(LOAD_DATA_PATH / "pv_q_mvar.csv")
reactive_df = reactive_df.drop(columns=["datetime"]) * LOAD_SCALE
active_df = active_df.drop(columns=["datetime"]) * LOAD_SCALE

net = pp.from_pickle(INPUT_PATH / "model.p")

ds_active = DFData(active_df)
ds_reactive = DFData(reactive_df)
ds_solar = DFData(solar_df)
ds_solar_reactive = DFData(solar_reactive_df)

load_mw_a_index = net.asymmetric_load[net.asymmetric_load["name"] == "a"].index
load_mw_b_index = net.asymmetric_load[net.asymmetric_load["name"] == "b"].index
load_mw_c_index = net.asymmetric_load[net.asymmetric_load["name"] == "c"].index


sgen_mw_a_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "a"].index
sgen_mw_b_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "b"].index
sgen_mw_c_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "c"].index


p_asymmetric_load_a = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_a_index,
    variable="p_a_mw",
    data_source=ds_active,
    profile_name=[str(i) for i in load_mw_a_index],
)

p_asymmetric_load_b = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_b_index,
    variable="p_b_mw",
    data_source=ds_active,
    profile_name=[str(i) for i in load_mw_b_index],
)
p_asymmetric_load_c = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_c_index,
    variable="p_c_mw",
    data_source=ds_active,
    profile_name=[str(i) for i in load_mw_c_index],
)

q_asymmetric_load_a = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_a_index,
    variable="q_a_mvar",
    data_source=ds_reactive,
    profile_name=[str(i) for i in load_mw_a_index],
)
q_asymmetric_load_b = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_b_index,
    variable="q_b_mvar",
    data_source=ds_reactive,
    profile_name=[str(i) for i in load_mw_b_index],
)
q_asymmetric_load_c = control.ConstControl(
    net,
    element="asymmetric_load",
    element_index=load_mw_c_index,
    variable="q_c_mvar",
    data_source=ds_reactive,
    profile_name=[str(i) for i in load_mw_c_index],
)


const_sgen_a = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_a_index,
    variable="p_a_mw",
    data_source=ds_solar,
    profile_name=[str(i) for i in sgen_mw_a_index],
)

const_sgen_b = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_b_index,
    variable="p_b_mw",
    data_source=ds_solar,
    profile_name=[str(i) for i in sgen_mw_b_index],
)
const_sgen_c = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_c_index,
    variable="p_c_mw",
    data_source=ds_solar,
    profile_name=[str(i) for i in sgen_mw_c_index],
)

const_sgen_a = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_a_index,
    variable="q_a_mvar",
    data_source=ds_solar_reactive,
    profile_name=[str(i) for i in sgen_mw_a_index],
)

const_sgen_b = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_b_index,
    variable="q_b_mvar",
    data_source=ds_solar_reactive,
    profile_name=[str(i) for i in sgen_mw_b_index],
)
const_sgen_c = control.ConstControl(
    net,
    element="asymmetric_sgen",
    element_index=sgen_mw_c_index,
    variable="q_c_mvar",
    data_source=ds_solar_reactive,
    profile_name=[str(i) for i in sgen_mw_c_index],
)


N_ITER = len(solar_df)


def run_timeseries(time_range):
    start_index, end_index = time_range
    output_path = SIMULATION_RESULTS_PATH / str(end_index)

    ow = timeseries.OutputWriter(
    net,
    output_path=output_path,
    output_file_type=".csv",
    csv_separator=",",
    write_time=660,
)
    ow.log_variable("res_bus_3ph", "vm_a_pu")
    ow.log_variable("res_bus_3ph", "vm_b_pu")
    ow.log_variable("res_bus_3ph", "vm_c_pu")


    ow.log_variable("res_asymmetric_load_3ph", "p_a_mw")
    ow.log_variable("res_asymmetric_load_3ph", "p_b_mw")
    ow.log_variable("res_asymmetric_load_3ph", "p_c_mw")


    ow.log_variable("res_asymmetric_sgen_3ph", "p_a_mw")
    ow.log_variable("res_asymmetric_sgen_3ph", "p_b_mw")
    ow.log_variable("res_asymmetric_sgen_3ph", "p_c_mw")

    ow.log_variable("res_asymmetric_sgen_3ph", "q_a_mvar")
    ow.log_variable("res_asymmetric_sgen_3ph", "q_b_mvar")
    ow.log_variable("res_asymmetric_sgen_3ph", "q_c_mvar")

    ow.log_variable("res_asymmetric_load_3ph", "q_a_mvar")
    ow.log_variable("res_asymmetric_load_3ph", "q_b_mvar")
    ow.log_variable("res_asymmetric_load_3ph", "q_c_mvar")

    ow.log_variable("res_ext_grid_3ph", "p_a_mw")
    ow.log_variable("res_ext_grid_3ph", "p_b_mw")
    ow.log_variable("res_ext_grid_3ph", "p_c_mw")

    ow.remove_log_variable("res_bus", "vm_pu")
    ow.remove_log_variable("res_line", "loading_percent")

    pp.toolbox.create_continuous_elements_index(net)

    timeseries.run_timeseries(
        net,
        run=pp.runpp_3ph,
        #     Some intervals failed
        continue_on_divergence=True,
        time_steps=list(range(start_index, end_index))
    )


def get_timeseries_chunks(length, n_chunks):
    chunks = np.array_split(np.arange(0, length), n_chunks)
    start_end_chunks = []
    for chunk in chunks:
        start_end_chunks.append((chunk[0], chunk[-1] + 1))
    return start_end_chunks

chunks = get_timeseries_chunks(len(solar_df), N_CORES)

with Pool(N_CORES) as p:
    p.map(run_timeseries, chunks)
