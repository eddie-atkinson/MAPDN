import torch
import argparse
import yaml
import pickle

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert
from utilities.tester import PGTester
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool

# N_INTERVALS = 350402
N_INTERVALS = 10
N_CORES = 2
OUTPUT_PATH = Path("./test")


def get_timeseries_chunks(length, n_chunks):
    chunks = np.array_split(np.arange(0, length), n_chunks)
    start_end_chunks = []
    for chunk in chunks:
        start_end_chunks.append((chunk[0], chunk[-1] + 1))
    return start_end_chunks


parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument(
    "--save-path",
    type=str,
    nargs="?",
    default="./",
    help="Please enter the directory of saving model.",
)
parser.add_argument(
    "--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name."
)
parser.add_argument(
    "--env",
    type=str,
    nargs="?",
    default="var_voltage_control",
    help="Please enter the env name.",
)
parser.add_argument(
    "--alias",
    type=str,
    nargs="?",
    default="",
    help="Please enter the alias for exp control.",
)
parser.add_argument(
    "--mode",
    type=str,
    nargs="?",
    default="distributed",
    help="Please enter the mode: distributed or decentralised.",
)
parser.add_argument(
    "--scenario",
    type=str,
    nargs="?",
    default="bus33_3min_final",
    help="Please input the valid name of an environment scenario.",
)
parser.add_argument(
    "--voltage-barrier-type",
    type=str,
    nargs="?",
    default="l1",
    help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.",
)
parser.add_argument(
    "--test-mode",
    type=str,
    nargs="?",
    default="single",
    help="Please input the valid test mode: single or batch.",
)
parser.add_argument(
    "--test-day",
    type=int,
    nargs="?",
    default=730,
    help="Please input the day you would test if the test mode is single.",
)
parser.add_argument(
    "--interval",
    type=int,
    nargs="?",
    default=350400,
    help="Please input the interval you would test if the test mode is single.",
)
parser.add_argument(
    "--render", action="store_true", help="Activate the rendering of the environment."
)
argv = parser.parse_args()

# load env args
with open("./args/env_args/" + argv.env + ".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
data_path[-1] = argv.scenario
env_config_dict["data_path"] = "/".join(data_path)
net_topology = argv.scenario

# set the action range
assert net_topology in [
    "case33_3min_final",
    "case141_3min_final",
    "case322_3min_final",
    "J_50percent",
], f"{net_topology} is not a valid scenario."
if argv.scenario == "case33_3min_final":
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == "case141_3min_final":
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif argv.scenario == "case322_3min_final":
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == "J_50percent":
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 1.0

assert argv.mode in [
    "distributed",
    "decentralised",
], "Please input the correct mode, e.g. distributed or decentralised."
env_config_dict["mode"] = argv.mode
env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

# for one-day test
env_config_dict["episode_limit"] = 480

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)
default_config_dict["max_steps"] = 480

# load alg args
with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

log_name = "-".join(
    [argv.env, net_topology, argv.mode, argv.alg, argv.voltage_barrier_type, argv.alias]
)
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
env = VoltageControl(env_config_dict)

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()
alg_config_dict["cuda"] = False
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] is "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path + "/"

LOAD_PATH = save_path + log_name + "/model.pt"

model = Model[argv.alg]

strategy = Strategy[argv.alg]

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)
checkpoint = (
    torch.load(LOAD_PATH, map_location="cpu")
    if not args.cuda
    else torch.load(LOAD_PATH)
)

behaviour_net.load_state_dict(checkpoint["model_state_dict"])

print(f"{args}\n")

if strategy == "pg":
    test = PGTester(args, behaviour_net, env, argv.render)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")


def create_df_from_array(array):
    return pd.DataFrame(
        index=range(len(array)),
        columns=[str(i) for i in range(len(array[0]))],
        data=array,
    )


def run_batch(chunk):
    start, stop = chunk
    test = PGTester(args, behaviour_net, env, argv.render)

    record = test.batch_run(start, stop)
    bus = record["bus"]
    pv = record["pv"]

    vm_a_pu = create_df_from_array(bus["vm_a_pu"])
    vm_b_pu = create_df_from_array(bus["vm_b_pu"])
    vm_c_pu = create_df_from_array(bus["vm_c_pu"])

    bus_a_p_mw = create_df_from_array(bus["p_a_mw"])
    bus_b_p_mw = create_df_from_array(bus["p_b_mw"])
    bus_c_p_mw = create_df_from_array(bus["p_c_mw"])

    pv_smax = create_df_from_array(pv["pv_smax"])
    pv_a_active = create_df_from_array(pv["pv_a_active"])
    pv_b_active = create_df_from_array(pv["pv_b_active"])
    pv_c_active = create_df_from_array(pv["pv_c_active"])
    pv_a_reactive = create_df_from_array(pv["pv_a_reactive"])
    pv_b_reactive = create_df_from_array(pv["pv_b_reactive"])
    pv_c_reactive = create_df_from_array(pv["pv_c_reactive"])

    dir_path = OUTPUT_PATH / f"{stop}"
    dir_path.mkdir(exist_ok=True)

    sgen_path = dir_path / "res_asymmetric_sgen_3ph"
    sgen_path.mkdir(exist_ok=True)

    bus_path = dir_path / "res_bus_3ph"
    bus_path.mkdir(exist_ok=True)

    vm_a_pu.to_csv(bus_path / "vm_a_pu.csv", index=False)
    vm_b_pu.to_csv(bus_path / "vm_b_pu.csv", index=False)
    vm_c_pu.to_csv(bus_path / "vm_c_pu.csv", index=False)

    bus_a_p_mw.to_csv(bus_path / "p_a_mw.csv", index=False)
    bus_b_p_mw.to_csv(bus_path / "p_b_mw.csv", index=False)
    bus_c_p_mw.to_csv(bus_path / "p_c_mw.csv", index=False)

    pv_a_active.to_csv(sgen_path / "p_a_mw.csv", index=False)
    pv_b_active.to_csv(sgen_path / "p_b_mw.csv", index=False)
    pv_c_active.to_csv(sgen_path / "p_c_mw.csv", index=False)

    pv_a_reactive.to_csv(sgen_path / "q_a_mvar.csv", index=False)
    pv_b_reactive.to_csv(sgen_path / "q_b_mvar.csv", index=False)
    pv_c_reactive.to_csv(sgen_path / "q_c_mvar.csv", index=False)

    pv_smax.to_csv(sgen_path / "pv_smax.csv", index=False)



if argv.test_mode == "single":
    # record = test.run(199, 23, 2) # (day, hour, 3min)
    # record = test.run(730, 23, 2) # (day, hour, 3min)
    record = test.run(argv.interval)
    with open(
        "test_record_" + log_name + f"_day{argv.test_day}" + ".pickle", "wb"
    ) as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == "batch":
    chunks = get_timeseries_chunks(N_INTERVALS, N_CORES)
    for chunk in chunks:
        run_batch(chunk)
    # with Pool(N_CORES) as p:
    #     p.map(run_batch, chunks)
