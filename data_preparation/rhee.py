import pandapower as pp
import pandas as pd
import numpy as np
from pathlib import Path

import pandapower.control as control
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData

NETWORK_NAME = "J"
INPUT_PATH = Path(f"./output_network_models/{NETWORK_NAME}")
OUTPUT_PATH = Path("./")
SIMULATION_RESULTS_PATH = Path(f"./simulation_results/{NETWORK_NAME}")
LOAD_DATA_PATH = Path("./output_data")
N_SITES = 109



reactive_df = pd.read_csv(LOAD_DATA_PATH / "test_reactive.csv", parse_dates=["datetime"])
active_df = pd.read_csv(LOAD_DATA_PATH / "test_active.csv", parse_dates=["datetime"])
solar_df = pd.read_csv(LOAD_DATA_PATH / "test_pv.csv", parse_dates=["datetime"])


reactive_df = reactive_df.drop(columns=["datetime"])
active_df = active_df.drop(columns=["datetime"])
solar_df = solar_df.drop(columns=["datetime"])

net = pp.from_pickle(INPUT_PATH / "model.p")


ds_active = DFData(active_df)
ds_reactive = DFData(reactive_df)
ds_solar = DFData(solar_df) 


load_mw_a_index = net.asymmetric_load[net.asymmetric_load["name"] == "a"].index
load_mw_b_index = net.asymmetric_load[net.asymmetric_load["name"] == "b"].index
load_mw_c_index = net.asymmetric_load[net.asymmetric_load["name"] == "c"].index


sgen_mw_a_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "a"].index
sgen_mw_b_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "b"].index
sgen_mw_c_index = net.asymmetric_sgen[net.asymmetric_sgen["name"] == "c"].index


p_asymmetric_load_a = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_a_index,
                                  variable='p_a_mw', data_source=ds_active, profile_name=[str(i) for i in load_mw_a_index])

p_asymmetric_load_b = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_b_index,
                                  variable='p_b_mw', data_source=ds_active, profile_name=[str(i) for i in load_mw_b_index])
p_asymmetric_load_c = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_c_index,
                                  variable='p_c_mw', data_source=ds_active, profile_name=[str(i) for i in load_mw_c_index])

q_asymmetric_load_a = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_a_index,
                                  variable='q_a_mvar', data_source=ds_reactive, profile_name=[str(i) for i in load_mw_a_index])
q_asymmetric_load_b = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_b_index,
                                  variable='q_b_mvar', data_source=ds_reactive, profile_name=[str(i) for i in load_mw_b_index])
q_asymmetric_load_c = control.ConstControl(net, element='asymmetric_load', element_index=load_mw_c_index,
                                  variable='q_c_mvar', data_source=ds_reactive, profile_name=[str(i) for i in load_mw_c_index])

const_sgen_a = control.ConstControl(net, element='asymmetric_sgen', element_index=sgen_mw_a_index,
                                  variable='p_a_mw', data_source=ds_solar, profile_name=[str(i) for i in sgen_mw_a_index])

const_sgen_b = control.ConstControl(net, element='asymmetric_sgen', element_index=sgen_mw_b_index,
                                  variable='p_b_mw', data_source=ds_solar, profile_name=[str(i) for i in sgen_mw_b_index])
const_sgen_c = control.ConstControl(net, element='asymmetric_sgen', element_index=sgen_mw_c_index,
                                  variable='p_c_mw', data_source=ds_solar, profile_name=[str(i) for i in sgen_mw_c_index])

test_df = pd.DataFrame(columns=active_df.columns, index=active_df.index)

for col in active_df.columns:
    test_df[col] = active_df[col] - solar_df[col]

    
ow = timeseries.OutputWriter(net, output_path=SIMULATION_RESULTS_PATH, output_file_type=".csv", csv_separator=",")
ow.log_variable("res_bus_3ph", "vm_a_pu")
ow.log_variable("res_bus_3ph", "vm_b_pu")
ow.log_variable("res_bus_3ph", "vm_c_pu")


ow.log_variable("res_asymmetric_load_3ph", "p_a_mw")
ow.log_variable("res_asymmetric_load_3ph", "p_b_mw")
ow.log_variable("res_asymmetric_load_3ph", "p_c_mw")

ow.log_variable("res_asymmetric_sgen_3ph", "p_a_mw")
ow.log_variable("res_asymmetric_sgen_3ph", "p_b_mw")
ow.log_variable("res_asymmetric_sgen_3ph", "p_c_mw")

ow.log_variable("res_asymmetric_load_3ph", "q_a_mvar")
ow.log_variable("res_asymmetric_load_3ph", "q_b_mvar")
ow.log_variable("res_asymmetric_load_3ph", "q_c_mvar")

ow.log_variable("res_ext_grid_3ph", "p_a_mw")
ow.log_variable("res_ext_grid_3ph", "p_b_mw")
ow.log_variable("res_ext_grid_3ph", "p_c_mw")

pp.toolbox.create_continuous_elements_index(net)
timeseries.run_timeseries(
    net, 
    run=pp.runpp_3ph, 
#     Some intervals failed
    continue_on_divergence=True,
)