import pandapower as pp
import json
from pathlib import Path
import random

BASE_BUS_KV = 0.416
EXT_GRID_PU = 1.04
SLACK_BUS_INDEX = 21
DEFAULT_LOAD_MW = 0.01
DEFAULT_LOAD_MVAR = 0.001
DEFAULT_SGEN_MW = 0.005
DEFAULT_SGEN_MVAR = 0.001
SGEN_PROPORTION = 1.0
RANDOM_SEED = 42

BUS_PATH = Path("./buses.json")
LINECODES_PATH = Path("./linecodes.json")
LINES_PATH = Path("./lines.json")
LOADS_PATH = Path("./loads.json")


random.seed(RANDOM_SEED)


def load_json(file_path: Path) -> dict:
    with open(file_path, "r") as infile:
        return json.loads(infile.read())


buses = load_json(BUS_PATH)
loads = load_json(LOADS_PATH)
linecodes = load_json(LINECODES_PATH)
lines = load_json(LINES_PATH)


net = pp.create_empty_network()

for linecode in linecodes:
    pp.create_std_type(net, linecode, linecode["name"], element="line")


for bus in buses:
    pp.create_bus(net, vn_kv=BASE_BUS_KV, **bus)

for line in lines:
    pp.create_line(net, **line)

for load in loads:
    phase = load["phase"]
    bus = load["bus"]
    load_data = {
        "net": net,
        "bus": bus,
        "p_a_mw": 0,
        "p_b_mw": 0,
        "p_c_mw": 0,
        "q_c_mvar": 0,
        "q_a_mvar": 0,
        "q_b_mvar": 0,
        # Override the active and reactive power values set above for the relevant phase
        f"p_{phase}_mw": DEFAULT_LOAD_MW,
        f"q_{phase}_mvar": DEFAULT_LOAD_MVAR,
    }
    load = pp.create_asymmetric_load(
        **load_data,
    )

    if random.random() <= SGEN_PROPORTION:
        sgen_data = {
            "net": net,
            "bus": bus,
            "p_b_mw": 0,
            "p_c_mw": 0,
            "q_c_mvar": 0,
            "q_a_mvar": 0,
            "q_b_mvar": 0,
            # Override the active and reactive power values set above for the relevant phase
            f"p_{phase}_mw": DEFAULT_SGEN_MW,
            f"q_{phase}_mvar": DEFAULT_SGEN_MVAR,
        }
        pp.create_asymmetric_sgen(**sgen_data)

pp.create_ext_grid(net, SLACK_BUS_INDEX, EXT_GRID_PU)
