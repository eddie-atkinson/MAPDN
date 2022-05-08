from ..multiagentenv import MultiAgentEnv
import numpy as np
import pandapower as pp
from pandapower import ppException
import pandas as pd
import copy
import os
from collections import namedtuple
from .pf_res_plot import pf_res_plotly
from .voltage_barrier.voltage_barrier_backend import VoltageBarrier

SECONDS_IN_ONE_HOUR = 3600
PHASES = ["a", "b", "c"]


def convert(dictionary):
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class VoltageControl(MultiAgentEnv):
    """this class is for the environment of distributed active voltage control

    it is easy to interact with the environment, e.g.,

    state, global_state = env.reset()
    for t in range(240):
        actions = agents.get_actions(state) # a vector involving all agents' actions
        reward, done, info = env.step(actions)
        next_state = env.get_obs()
        state = next_state
    """

    def __init__(self, kwargs):
        """initialisation"""
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.data_path = args.data_path
        # set the data path
        self.is_3ph = args.is_3ph

        # load the model of power network
        self.base_powergrid = self._load_network()
        print(self.base_powergrid)

        # load data
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()

        # define episode and rewards
        self.episode_limit = args.episode_limit
        self.voltage_barrier_type = getattr(args, "voltage_barrier_type", "l1")
        self.voltage_weight = getattr(args, "voltage_weight", 1.0)
        self.q_weight = getattr(args, "q_weight", 0.1)
        self.line_weight = getattr(args, "line_weight", None)
        self.dv_dq_weight = getattr(args, "dq_dv_weight", None)

        # define constraints and uncertainty
        self.v_upper = getattr(args, "v_upper", 1.05)
        self.v_lower = getattr(args, "v_lower", 0.95)
        self.active_demand_std = self.active_demand_data.values.std(axis=0) / 100.0
        self.reactive_demand_std = self.reactive_demand_data.values.std(axis=0) / 100.0
        self.pv_std = self.pv_data.values.std(axis=0) / 100.0
        self._set_reactive_power_boundary()

        try:
            action_floor = self.args.action_floor
        except AttributeError:
            action_floor = None
        # define action space and observation space
        low = (
            action_floor
            if action_floor
            else -self.args.action_scale + self.args.action_bias
        )
        self.action_space = ActionSpace(
            low=low,
            high=self.args.action_scale + self.args.action_bias,
        )

        self.history = getattr(args, "history", 1)
        self.state_space = getattr(
            args, "state_space", ["pv", "demand", "reactive", "vm_pu", "va_degree"]
        )
        if self.args.mode == "distributed":
            self.n_actions = 2
            self.n_agents = len(self._get_base_sgen())
        elif self.args.mode == "decentralised":
            self.n_actions = len(self._get_base_sgen())
            self.n_agents = (
                len(set(self.base_powergrid.bus["zone"].to_numpy(copy=True))) - 1
            )  # exclude the main zone
        agents_obs, state = self.reset()

        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]

        # self.last_v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        # self.last_q = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)

        # initialise voltage barrier function
        self.voltage_barrier = VoltageBarrier(self.voltage_barrier_type)
        self._rendering_initialized = False

    def reset(self, reset_time=True):
        """reset the env"""
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        solvable = False
        while not solvable:
            # reset the time stamp
            if reset_time:
                self._episode_start_interval = self._select_start_interval()
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv()
            # random initialise action
            if self.args.reset_action:
                actions = self.get_action()
                self._set_sgen_q_mvar(actions)
            try:
                self.get_run_pp_fn()(self.powergrid)
                solvable = True
            except ppException:
                print(
                    "The power flow for the initialisation of demand and PV cannot be solved."
                )
                print(f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print(f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print(f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print(f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print(f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False
        return self.get_obs(), self.get_state()

    def manual_reset(self, interval):
        """manual reset the initial date"""
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)

        # reset the time stamp
        self._episode_start_interval = interval
        solvable = False
        while not solvable:
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv(add_noise=True)
            # random initialise action
            if self.args.reset_action:
                actions = self.get_action()
                self._set_sgen_q_mvar(actions)
            try:
                self.get_run_pp_fn()(self.powergrid)
                solvable = True
            except ppException:
                print(
                    "The power flow for the initialisation of demand and PV cannot be solved."
                )
                print(f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print(f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print(f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print(f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print(f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()

    def step(self, actions, add_noise=True):
        """function for the interaction between agent and the env each time step"""
        last_powergrid = copy.deepcopy(self.powergrid)
        # check whether the power balance is unsolvable
        solvable = self._take_action(actions)
        if solvable:
            # get the reward of current actions
            reward, info = self._calc_reward()
        else:
            q_loss = np.mean(np.abs(self.powergrid.sgen["q_mvar"]))
            self.powergrid = last_powergrid
            reward, info = self._calc_reward()
            reward -= 200.0
            # keep q_loss
            info["destroy"] = 1.0
            info["totally_controllable_ratio"] = 0.0
            info["q_loss"] = q_loss

        # set the pv and demand for the next time step
        self._set_demand_and_pv(add_noise=add_noise)
        # terminate if episode_limit is reached
        self.steps += 1
        self.sum_rewards += reward
        if self.steps >= self.episode_limit or not solvable:
            terminated = True
        else:
            terminated = False
        if terminated:
            print(
                f"Episode terminated at time: {self.steps} with return: {self.sum_rewards:2.4f}."
            )

        return reward, terminated, info

    def get_state(self):
        """return the global state for the power system
        the default state: voltage, active power of generators, bus state, load active power, load reactive power
        """
        if self._is_3ph():
            return self._get_state_3ph()
        else:
            return self._get_state()

    def get_obs(self):
        """return the obs for each agent in the power system
        the default obs: voltage, active power of generators, bus state, load active power, load reactive power
        each agent can only observe the state within the zone where it belongs
        """
        clusters = self._get_clusters_info()
        if self.args.mode == "distributed":
            if self._is_3ph():
                obs_zone_dict, zone_list, obs_len_list = self._get_obs_distributed_3ph(
                    clusters
                )
            else:
                obs_zone_dict, zone_list, obs_len_list = self._get_obs_distributed(
                    clusters
                )

            agents_obs = list()
            obs_max_len = max(obs_len_list)
            for zone in zone_list:
                obs_zone = obs_zone_dict[zone]
                pad_obs_zone = np.concatenate(
                    [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0
                )
                agents_obs.append(pad_obs_zone)
        # TODO: don't have decentralised mode working yet
        elif self.args.mode == "decentralised":
            obs_len_list = list()
            zone_obs_list = list()
            for i in range(self.n_agents):
                zone_buses, pv, q, sgen_buses = clusters[f"zone{i+1}"]
                obs = list()
                if "demand" in self.state_space:
                    copy_zone_buses = copy.deepcopy(zone_buses)
                    copy_zone_buses.loc[sgen_buses]["p_mw"] += pv
                    copy_zone_buses.loc[sgen_buses]["q_mvar"] += q
                    obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                    obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                if "pv" in self.state_space:
                    obs += list(pv.to_numpy(copy=True))
                if "reactive" in self.state_space:
                    obs += list(q.to_numpy(copy=True))
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    obs += list(
                        zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180
                    )
                obs = np.array(obs)
                zone_obs_list.append(obs)
                obs_len_list.append(obs.shape[0])
            agents_obs = []
            obs_max_len = max(obs_len_list)
            for obs_zone in zone_obs_list:
                pad_obs_zone = np.concatenate(
                    [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0
                )
                agents_obs.append(pad_obs_zone)
        if self.history > 1:
            agents_obs_ = []
            for i, obs in enumerate(agents_obs):
                if len(self.obs_history[i]) >= self.history - 1:
                    obs_ = np.concatenate(
                        self.obs_history[i][-self.history + 1 :] + [obs], axis=0
                    )
                else:
                    zeros = [np.zeros_like(obs)] * (
                        self.history - len(self.obs_history[i]) - 1
                    )
                    obs_ = self.obs_history[i] + [obs]
                    obs_ = zeros + obs_
                    obs_ = np.concatenate(obs_, axis=0)
                agents_obs_.append(copy.deepcopy(obs_))
                self.obs_history[i].append(copy.deepcopy(obs))
            agents_obs = agents_obs_
        return agents_obs

    def get_obs_agent(self, agent_id):
        """return observation for agent_id"""
        agents_obs = self.get_obs()
        return agents_obs[agent_id]

    def get_obs_size(self):
        """return the observation size"""
        return self.obs_size

    def get_state_size(self):
        """return the state size"""
        return self.state_size

    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)"""
        n_actions = self.get_total_actions()
        if self._is_3ph():
            size = (
                self.powergrid.asymmetric_sgen["q_a_mvar"].values.shape[0],
                n_actions,
            )
        else:
            size = (self.powergrid.sgen["q_mvar"].values.shape[0], n_actions)

        actions = np.random.uniform(
            low=self.action_space.low,
            high=self.action_space.high,
            size=size,
        )

        return actions

    def get_total_actions(self):
        """return the total number of actions an agent could ever take"""
        return self.n_actions

    def get_avail_actions(self):
        """return available actions for all agents"""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """return the available actions for agent_id"""
        if self.args.mode == "distributed":
            return [1]
        elif self.args.mode == "decentralised":
            avail_actions = np.zeros(self.n_actions)
            zone_sgens = self.base_powergrid.sgen.loc[
                self.base_powergrid.sgen["name"] == f"zone{agent_id+1}"
            ]
            avail_actions[zone_sgens.index] = 1
            return avail_actions

    def get_num_of_agents(self):
        """return the number of agents"""
        return self.n_agents

    def _get_voltage(self):
        return self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

    def _create_basenet(self, base_net):
        """initilization of power grid
        set the pandapower net to use
        """
        if base_net is None:
            raise Exception(
                "Please provide a base_net configured as pandapower format."
            )
        else:
            return base_net

    def _select_start_interval(self):
        """select start interval for an episode"""
        pv_data = self.pv_data
        available_range = (
            len(pv_data.index) - self.episode_limit - 1
        )  # subtract 1 for safety margin
        return np.random.choice(available_range)

    def _load_network(self):
        """load network"""
        network_path = os.path.join(self.data_path, "model.p")
        base_net = pp.from_pickle(network_path)
        return self._create_basenet(base_net)

    def _load_pv_data(self):
        """load pv data
        the sensor frequency is set to 3 or 15 mins as default
        """
        pv_path = os.path.join(self.data_path, "pv_active.csv")
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = "time"
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return pv

    def _load_active_demand_data(self):
        """load active demand data
        the sensor frequency is set to 3 or 15 mins as default
        """
        demand_path = os.path.join(self.data_path, "load_active.csv")
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = "time"
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def _load_reactive_demand_data(self):
        """load reactive demand data
        the sensor frequency is set to 3 min as default
        """
        demand_path = os.path.join(self.data_path, "load_reactive.csv")
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = "time"
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def _get_episode_pv_history(self):
        """return the pv history in an episode"""
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_pv_history = self.pv_data[start : start + nr_intervals].values
        return episode_pv_history

    def _get_episode_active_demand_history(self):
        """return the active power histories for all loads in an episode"""
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.active_demand_data[
            start : start + nr_intervals
        ].values
        return episode_demand_history

    def _get_episode_reactive_demand_history(self):
        """return the reactive power histories for all loads in an episode"""
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval
        nr_intervals = episode_length + history + 1  # margin of 1
        episode_demand_history = self.reactive_demand_data[
            start : start + nr_intervals
        ].values
        return episode_demand_history

    def _get_pv_history(self):
        """returns pv history"""
        t = self.steps
        history = self.history
        return self.pv_histories[t : t + history, :]

    def _get_active_demand_history(self):
        """return the history demand"""
        t = self.steps
        history = self.history
        return self.active_demand_histories[t : t + history, :]

    def _get_reactive_demand_history(self):
        """return the history demand"""
        t = self.steps
        history = self.history
        return self.reactive_demand_histories[t : t + history, :]

    def _set_demand_and_pv(self, add_noise=True):
        """optionally update the demand and pv production according to the histories with some i.i.d. gaussian noise"""
        pv = copy.copy(self._get_pv_history()[0, :])

        # add uncertainty to pv data with unit truncated gaussian (only positive accepted)
        if add_noise:
            pv += self.pv_std * np.abs(np.random.randn(*pv.shape))
        active_demand = copy.copy(self._get_active_demand_history()[0, :])

        # add uncertainty to active power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            active_demand += self.active_demand_std * np.abs(
                np.random.randn(*active_demand.shape)
            )
        reactive_demand = copy.copy(self._get_reactive_demand_history()[0, :])

        # add uncertainty to reactive power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            reactive_demand += self.reactive_demand_std * np.abs(
                np.random.randn(*reactive_demand.shape)
            )

        # update the record in the pandapower
        self._set_sgen_p_mw(pv)
        self._set_load_p_mw(active_demand)
        self._set_load_q_mvar(reactive_demand)

    def _set_reactive_power_boundary(self):
        """set the boundary of reactive power"""
        # Don't oversize the inverters
        self.factor = 1.0
        self.p_max = self.pv_data.to_numpy(copy=True).max(axis=0)
        self.s_max = self.factor * self.p_max
        print(f"This is the s_max: \n{self.s_max}")

    def _get_clusters_info(self):
        """return the clusters of info
        the clusters info is divided by predefined zone
        distributed: each zone is equipped with several PV generators and each PV generator is an agent
        decentralised: each zone is controlled by an agent and each agent may have variant number of actions
        """
        clusters = dict()
        if self.args.mode == "distributed":
            sgens = self.get_sgen()
            for i in range(len(sgens)):
                sgen = sgens.iloc[i]
                zone = sgen["name"]
                sgen_bus = sgen["bus"]
                pv = self._get_sgen_mw(i)
                q = self._get_sgen_mvar(i)
                res_buses = self._get_res_bus().sort_index()
                zone_res_buses = res_buses[
                    self.powergrid.bus["zone"].str.contains(zone).fillna(False)
                ]
                clusters[f"sgen{i}"] = (zone_res_buses, zone, pv, q, sgen_bus)
        elif self.args.mode == "decentralised":
            for i in range(self.n_agents):
                # TODO: this doesn't work for 3ph yet
                res_buses = self._get_res_bus().sort_index()
                zone_res_buses = res_buses.loc[
                    self.powergrid.bus["zone"] == f"zone{i+1}"
                ]
                sgen_res_buses = self.powergrid.sgen["bus"].loc[
                    self.powergrid.sgen["name"] == f"zone{i+1}"
                ]
                pv = self.powergrid.sgen["p_mw"].loc[
                    self.powergrid.sgen["name"] == f"zone{i+1}"
                ]
                q = self.powergrid.sgen["q_mvar"].loc[
                    self.powergrid.sgen["name"] == f"zone{i+1}"
                ]
                clusters[f"zone{i+1}"] = (zone_res_buses, pv, q, sgen_res_buses)

        return clusters

    def _take_action(self, actions):
        """take the control variables
        the control variables we consider are the exact reactive power
        of each distributed generator
        """
        self._set_sgen_q_mvar(actions)

        # solve power flow to get the latest voltage with new reactive power and old deamnd and PV active power
        try:
            self.get_run_pp_fn()(self.powergrid)
            return True
        except ppException:
            print("The power flow for the reactive power penetration cannot be solved.")
            print(f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
            print(f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
            print(f"This is the active demand: \n{self.powergrid.load['p_mw']}")
            print(f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
            print(f"This is the res_bus: \n{self.powergrid.res_bus}")
            return False

    def _clip_reactive_power(self, reactive_actions, active_power):
        """clip the reactive power tto the hard safety range"""
        reactive_power_constraint = np.sqrt(
            self.s_max[active_power.index] ** 2 - active_power**2
        )
        return reactive_power_constraint * reactive_actions

    def _calc_reward(self, info={}):
        """reward function
        consider 5 possible choices on voltage barrier functions:
            l1
            l2
            courant_beltrami
            bowl
            bump
        """
        if self._is_3ph():
            return self._calc_reward_3ph(info)
        else:
            return self._calc_reward_1ph(info)

    def _calc_reward_3ph(self, info):
        v_a = self.powergrid.res_bus_3ph["vm_a_pu"].sort_index().to_numpy(copy=True)
        v_b = self.powergrid.res_bus_3ph["vm_b_pu"].sort_index().to_numpy(copy=True)
        v_c = self.powergrid.res_bus_3ph["vm_c_pu"].sort_index().to_numpy(copy=True)

        v_a_below_lower = v_a < self.v_lower
        v_a_above_upper = v_a > self.v_upper

        v_a_out_of_control = np.sum(v_a_below_lower) + np.sum(v_a_above_upper)

        v_b_below_lower = v_b < self.v_lower
        v_b_above_upper = v_b > self.v_upper

        v_b_out_of_control = np.sum(v_b_below_lower) + np.sum(v_b_above_upper)

        v_c_below_lower = v_c < self.v_lower
        v_c_above_upper = v_c > self.v_upper
        v_c_out_of_control = np.sum(v_c_below_lower) + np.sum(v_c_above_upper)

        percent_a_out_of_control = v_a_out_of_control / v_a.shape[0]
        percent_b_out_of_control = v_b_out_of_control / v_b.shape[0]
        percent_c_out_of_control = v_c_out_of_control / v_c.shape[0]

        percent_out_of_control = (
            v_a_out_of_control + v_b_out_of_control + v_c_out_of_control
        ) / (v_a.shape[0] + v_b.shape[0] + v_c.shape[0])

        percent_a_above_upper = np.sum(v_a_above_upper) / v_a.shape[0]
        percent_b_above_upper = np.sum(v_b_above_upper) / v_b.shape[0]
        percent_c_above_upper = np.sum(v_c_above_upper) / v_c.shape[0]

        percent_a_below_lower = np.sum(v_a_below_lower) / v_a.shape[0]
        percent_b_below_lower = np.sum(v_b_below_lower) / v_b.shape[0]
        percent_c_below_lower = np.sum(v_c_below_lower) / v_c.shape[0]

        percent_above_upper = (
            np.sum(v_a_above_upper) + np.sum(v_b_above_upper) + np.sum(v_c_above_upper)
        ) / (v_a.shape[0] + v_b.shape[0] + v_c.shape[0])
        percent_below_lower = (
            np.sum(v_a_below_lower) + np.sum(v_b_below_lower) + np.sum(v_c_below_lower)
        ) / (v_a.shape[0] + v_b.shape[0] + v_c.shape[0])

        info["percentage_of_v_out_of_control"] = percent_out_of_control
        info["percentage_of_v_a_out_of_control"] = percent_a_out_of_control
        info["percentage_of_v_b_out_of_control"] = percent_b_out_of_control
        info["percentage_of_v_c_out_of_control"] = percent_c_out_of_control

        info["percentage_of_lower_than_lower_v"] = percent_below_lower
        info["percentage_of_higher_than_upper_v"] = percent_above_upper

        info["percentage_of_a_higher_than_upper_v"] = percent_a_above_upper
        info["percentage_of_b_higher_than_upper_v"] = percent_b_above_upper
        info["percentage_of_c_higher_than_upper_v"] = percent_c_above_upper

        info["percentage_of_a_lower_than_lower_v"] = percent_a_below_lower
        info["percentage_of_b_lower_than_lower_v"] = percent_b_below_lower
        info["percentage_of_c_lower_than_lower_v"] = percent_c_below_lower

        info["totally_controllable_ratio"] = (
            0.0 if percent_out_of_control > 1e-3 else 1.0
        )

        # voltage violation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        average_v_a = np.mean(v_a)
        average_v_b = np.mean(v_b)
        average_v_c = np.mean(v_c)

        average_v_deviation_a = np.mean(np.abs(v_a - v_ref))
        average_v_deviation_b = np.mean(np.abs(v_b - v_ref))
        average_v_deviation_c = np.mean(np.abs(v_c - v_ref))

        average_v_deviation = (
            np.sum(np.abs(v_a - v_ref))
            + np.sum(np.abs(v_b - v_ref))
            + np.sum(np.abs(v_c - v_ref)) / (v_a.shape[0] + v_b.shape[0] + v_c.shape[0])
        )

        average_v_a = np.mean(v_a)
        average_v_b = np.mean(v_b)
        average_v_c = np.mean(v_c)

        average_v = (
            np.sum(v_a)
            + np.sum(v_b)
            + np.sum(v_c) / (v_a.shape[0] + v_b.shape[0] + v_c.shape[0])
        )

        info["average_voltage_deviation"] = average_v_deviation
        info["average_a_voltage_deviation"] = average_v_deviation_a
        info["average_b_voltage_deviation"] = average_v_deviation_b
        info["average_c_voltage_deviation"] = average_v_deviation_c

        info["average_voltage"] = average_v
        info["average_voltage_a"] = average_v_a
        info["average_voltage_b"] = average_v_b
        info["average_voltage_c"] = average_v_c

        # active power loss
        sgen_res = self.powergrid.res_asymmetric_sgen_3ph

        pv_active_max = self.pv_histories[self.steps, :]

        a = self._get_sgen_on_phase("a")
        b = self._get_sgen_on_phase("b")
        c = self._get_sgen_on_phase("c")

        p_loss_a = pv_active_max[a.index] - sgen_res.loc[a.index, "p_a_mw"]
        p_loss_b = pv_active_max[b.index] - sgen_res.loc[b.index, "p_b_mw"]
        p_loss_c = pv_active_max[c.index] - sgen_res.loc[c.index, "p_c_mw"]

        p_loss = (np.sum(p_loss_a) + np.sum(p_loss_b) + np.sum(p_loss_c)) / (
            p_loss_a.shape[0] + p_loss_b.shape[0] + p_loss_c.shape[0]
        )

        info["p_loss"] = p_loss
        # reward function
        ## voltage barrier function
        v_loss_a = np.mean(self.voltage_barrier.step(v_a)) * self.voltage_weight
        v_loss_b = np.mean(self.voltage_barrier.step(v_b)) * self.voltage_weight
        v_loss_c = np.mean(self.voltage_barrier.step(v_c)) * self.voltage_weight
        ## add soft constraint for line or q
        if self.line_weight != None:
            raise NotImplementedError("Can't compute average line loss for three phase")

        # TODO: This is an insane hack, but the q_loss essentially encodes our appetite for losing active power
        elif self.q_weight != None:
            loss = p_loss * self.q_weight + v_loss_a + v_loss_b + v_loss_c
        else:
            raise NotImplementedError(
                "Please at least give one qweight, either q_weight or line_weight."
            )
        reward = -loss

        # record destroy
        info["destroy"] = 0.0

        return reward, info

    def _calc_reward_1ph(self, info):
        # percentage of voltage out of control
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        percent_of_v_out_of_control = (
            np.sum(v < self.v_lower) + np.sum(v > self.v_upper)
        ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control
        info["percentage_of_lower_than_lower_v"] = np.sum(v < self.v_lower) / v.shape[0]
        info["percentage_of_higher_than_upper_v"] = (
            np.sum(v > self.v_upper) / v.shape[0]
        )
        info["totally_controllable_ratio"] = (
            0.0 if percent_of_v_out_of_control > 1e-3 else 1.0
        )

        # voltage violation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean(np.abs(v - v_ref))
        info["average_voltage"] = np.mean(v)
        info["max_voltage_drop_deviation"] = np.max(
            (v < self.v_lower) * (self.v_lower - v)
        )
        info["max_voltage_rise_deviation"] = np.max(
            (v > self.v_upper) * (v - self.v_upper)
        )

        # line loss
        line_loss = np.sum(self.powergrid.res_line["pl_mw"])
        avg_line_loss = np.mean(self.powergrid.res_line["pl_mw"])
        info["total_line_loss"] = line_loss

        # reactive power (q) loss
        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss

        # reward function
        ## voltage barrier function
        v_loss = np.mean(self.voltage_barrier.step(v)) * self.voltage_weight
        ## add soft constraint for line or q
        if self.line_weight != None:
            loss = avg_line_loss * self.line_weight + v_loss
        elif self.q_weight != None:
            loss = q_loss * self.q_weight + v_loss
        else:
            raise NotImplementedError(
                "Please at least give one weight, either q_weight or line_weight."
            )
        reward = -loss

        # record destroy
        info["destroy"] = 0.0

        return reward, info

    def _get_res_bus_v(self):
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        return v

    def _get_res_bus_active(self):
        active = self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
        return active

    def _get_res_bus_reactive(self):
        reactive = self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
        return reactive

    def _get_res_line_loss(self):
        line_loss = self.powergrid.res_line["pl_mw"].sort_index().to_numpy(copy=True)
        return line_loss

    def _get_sgen_active(self):
        active = self.powergrid.sgen["p_mw"].to_numpy(copy=True)
        return active

    def _get_sgen_reactive(self):
        reactive = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)
        return reactive

    def _init_render(self):
        from .rendering_voltage_control_env import Viewer

        self.viewer = Viewer()
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))

    def res_pf_plot(self):
        if not os.path.exists("environments/var_voltage_control/plot_save"):
            os.mkdir("environments/var_voltage_control/plot_save")

        fig = pf_res_plotly(
            self.powergrid,
            aspectratio=(1.0, 1.0),
            filename="environments/var_voltage_control/plot_save/pf_res_plot.html",
            auto_open=False,
            climits_volt=(0.9, 1.1),
            line_width=5,
            bus_size=12,
            climits_load=(0, 100),
            cpos_load=1.1,
            cpos_volt=1.0,
        )
        fig.write_image("environments/var_voltage_control/plot_save/pf_res_plot.jpeg")

    def _is_3ph(self):
        return self.is_3ph

    def get_sgen(self):
        if self._is_3ph():
            return self.powergrid.asymmetric_sgen
        else:
            return self.powergrid.sgen

    def _get_base_sgen(self):
        if self._is_3ph():
            return self.base_powergrid.asymmetric_sgen
        else:
            return self.base_powergrid.sgen

    def get_load(self):
        if self._is_3ph():
            return self.powergrid.asymmetric_load
        else:
            return self.powergrid.load

    def _get_base_load(self):
        if self._is_3ph():
            return self.base_powergrid.asymmetric_load
        else:
            return self.base_powergrid.load

    def get_run_pp_fn(self):
        if self._is_3ph():
            return pp.runpp_3ph
        else:
            return pp.runpp

    def _get_sgen_on_phase(self, phase):
        """
        phase: one of ["a", "b", "c"]
        """
        sgen = self.get_sgen()
        return sgen[sgen["name"] == phase]

    def _get_load_on_phase(self, phase):
        """
        phase: one of ["a", "b", "c"]
        """
        load = self.get_load()
        return load[load["name"] == phase]

    def _set_load_q_mvar(self, q):
        load = self.get_load()
        q = q[: len(load)]
        if self._is_3ph():
            # PV contains more profiles than we may need

            a = self._get_load_on_phase("a")
            b = self._get_load_on_phase("b")
            c = self._get_load_on_phase("c")

            load.loc[a.index, "q_a_mvar"] = q[a.index]
            load.loc[b.index, "q_b_mvar"] = q[b.index]
            load.loc[c.index, "q_c_mvar"] = q[c.index]
        else:
            load["q_mvar"] = q

    def _set_load_p_mw(self, p):
        load = self.get_load()
        p = p[: len(load)]
        if self._is_3ph():
            # PV contains more profiles than we may need

            a = self._get_load_on_phase("a")
            b = self._get_load_on_phase("b")
            c = self._get_load_on_phase("c")

            load.loc[a.index, "p_a_mw"] = p[a.index]
            load.loc[b.index, "p_b_mw"] = p[b.index]
            load.loc[c.index, "p_c_mw"] = p[c.index]
        else:
            load["p_mw"] = p

    def _set_sgen_p_mw(self, pv):
        sgen = self.get_sgen()
        pv = pv[: len(sgen)]
        if self._is_3ph():
            # PV contains more profiles than we may need

            a = self._get_sgen_on_phase("a")
            b = self._get_sgen_on_phase("b")
            c = self._get_sgen_on_phase("c")

            sgen.loc[a.index, "p_a_mw"] = pv[a.index]
            sgen.loc[b.index, "p_b_mw"] = pv[b.index]
            sgen.loc[c.index, "p_c_mw"] = pv[c.index]
        else:
            sgen["p_mw"] = pv

    def _map_to_range(self, old_value, old_max, old_min, new_max, new_min):
        old_range = old_max - old_min
        new_range = new_max - new_min
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        return new_value

        return new_value

    def _apply_actions(self, phase, apparent, reactive):
        sgen = self.get_sgen()
        phase_sgens = self._get_sgen_on_phase(phase)
        s = sgen.loc[phase_sgens.index, f"p_{phase}_mw"] * apparent[phase_sgens.index]
        min_p = 0.8 * s
        max_q = np.sqrt(s**2 - min_p**2)
        q = reactive[phase_sgens.index] * max_q
        p = np.sqrt(s**2 - q**2)

        return p, q

    def _set_sgen_q_mvar(self, actions):
        apparent = actions[:, 0]
        reactive = actions[:, 1]
        apparent_proportion = self._map_to_range(apparent, 1.0, -1.0, 1.0, 0.0)
        reactive_proportion = self._map_to_range(reactive, 1.0, -1.0, 1.0, -1.0)

        sgen = self.get_sgen()
        actions = actions[: len(sgen)]

        if self._is_3ph():
            a = self._get_sgen_on_phase("a")
            b = self._get_sgen_on_phase("b")
            c = self._get_sgen_on_phase("c")

            p_a, q_a = self._apply_actions(
                "a", apparent_proportion, reactive_proportion
            )
            p_b, q_b = self._apply_actions(
                "b", apparent_proportion, reactive_proportion
            )
            p_c, q_c = self._apply_actions(
                "c", apparent_proportion, reactive_proportion
            )

            sgen.loc[a.index, "p_a_mw"] = p_a
            sgen.loc[a.index, "q_a_mvar"] = q_a
            sgen.loc[b.index, "p_b_mw"] = p_b
            sgen.loc[b.index, "q_b_mvar"] = q_b
            sgen.loc[c.index, "p_a_mw"] = p_c
            sgen.loc[c.index, "q_c_mvar"] = q_c

        else:
            # TODO: this doesn't work
            s = sgen["p_mw"] * apparent_proportion
            q = s * reactive_proportion
            p = np.sqrt(s**2 - q**2)
            sgen["q_mvar"] = q
            sgen["p_mw"] = p

    def _get_sgen_mw(self, index):
        sgen = self.get_sgen().iloc[index]
        if self._is_3ph():
            phase = sgen["name"]
            return sgen[f"p_{phase}_mw"]
        return sgen["p_mw"]

    def _get_sgen_mvar(self, index):
        sgen = self.get_sgen().iloc[index]
        if self._is_3ph():
            phase = sgen["name"]
            return sgen[f"q_{phase}_mvar"]
        return sgen["q_mvar"]

    def _get_res_bus(self):
        if self._is_3ph():
            return self.powergrid.res_bus_3ph
        else:
            return self.powergrid.res_bus

    def _get_state_3ph(self):
        state = []
        if "demand" in self.state_space:
            # Add per phase power
            state += list(
                self.powergrid.res_bus_3ph["p_a_mw"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus_3ph["p_b_mw"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus_3ph["p_c_mw"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus_3ph["q_a_mvar"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus_3ph["q_b_mvar"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus_3ph["q_c_mvar"].sort_index().to_numpy(copy=True)
            )
        if "pv" in self.state_space:
            sgen = self.get_sgen()
            state += list(sgen["p_a_mw"].sort_index().to_numpy(copy=True))
            state += list(sgen["p_b_mw"].sort_index().to_numpy(copy=True))
            state += list(sgen["p_c_mw"].sort_index().to_numpy(copy=True))

        if "reactive" in self.state_space:
            state += list(sgen["q_a_mvar"].sort_index().to_numpy(copy=True))
            state += list(sgen["q_b_mvar"].sort_index().to_numpy(copy=True))
            state += list(sgen["q_c_mvar"].sort_index().to_numpy(copy=True))

        if "vm_pu" in self.state_space:
            res_bus = self._get_res_bus()
            state += list(res_bus["vm_a_pu"].sort_index().to_numpy(copy=True))
            state += list(res_bus["vm_b_pu"].sort_index().to_numpy(copy=True))
            state += list(res_bus["vm_c_pu"].sort_index().to_numpy(copy=True))

        if "va_degree" in self.state_space:
            state += list(res_bus["va_a_degree"].sort_index().to_numpy(copy=True))
            state += list(res_bus["va_b_degree"].sort_index().to_numpy(copy=True))
            state += list(res_bus["va_c_degree"].sort_index().to_numpy(copy=True))
        state = np.array(state)
        return state

    def _get_state(self):
        state = []
        if "demand" in self.state_space:
            state += list(
                self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
            )
            state += list(
                self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
            )
        if "pv" in self.state_space:
            state += list(self.powergrid.sgen["p_mw"].sort_index().to_numpy(copy=True))
        if "reactive" in self.state_space:
            state += list(
                self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True)
            )
        if "vm_pu" in self.state_space:
            state += list(
                self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
            )
        if "va_degree" in self.state_space:
            state += list(
                self.powergrid.res_bus["va_degree"].sort_index().to_numpy(copy=True)
            )
        state = np.array(state)
        return state

    def _get_obs_distributed_3ph(self, clusters):
        obs_zone_dict = dict()
        zone_list = list()
        obs_len_list = list()
        for i in range(len(self.get_sgen())):
            obs = list()
            zone_buses, zone, pv, q, sgen_bus = clusters[f"sgen{i}"]
            zone_list.append(zone)
            if not (zone in obs_zone_dict.keys()):
                copy_zone_buses = copy.deepcopy(zone_buses)
                if "demand" in self.state_space:
                    copy_zone_buses.loc[sgen_bus][f"p_{zone}_mw"] += pv
                    copy_zone_buses.loc[sgen_bus][f"q_{zone}_mvar"] += q
                    obs += list(
                        copy_zone_buses.loc[:, f"p_{zone}_mw"].to_numpy(copy=True)
                    )
                    obs += list(
                        copy_zone_buses.loc[:, f"q_{zone}_mvar"].to_numpy(copy=True)
                    )
                if "pv" in self.state_space:
                    obs += list(copy_zone_buses.loc[:, f"p_{zone}_mw"].to_numpy(copy=True))
                if "reactive" in self.state_space:
                    obs += list(copy_zone_buses.loc[:, f"q_{zone}_mvar"].to_numpy(copy=True))
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, f"vm_{zone}_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    # transform the voltage phase to radian
                    obs += list(
                        zone_buses.loc[:, f"va_{zone}_degree"].to_numpy(copy=True)
                        * np.pi
                        / 180
                    )
                obs_zone_dict[zone] = np.array(obs)
            obs_len_list.append(obs_zone_dict[zone].shape[0])
        return obs_zone_dict, zone_list, obs_len_list

    def _get_obs_distributed(self, clusters):
        obs_zone_dict = dict()
        zone_list = list()
        obs_len_list = list()
        for i in range(len(self.get_sgen())):
            obs = list()
            zone_buses, zone, pv, q, sgen_bus = clusters[f"sgen{i}"]
            zone_list.append(zone)
            if not (zone in obs_zone_dict.keys()):
                if "demand" in self.state_space:
                    copy_zone_buses = copy.deepcopy(zone_buses)
                    copy_zone_buses.loc[sgen_bus]["p_mw"] += pv
                    copy_zone_buses.loc[sgen_bus]["q_mvar"] += q
                    obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                    obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                if "pv" in self.state_space:
                    obs.append(pv)
                if "reactive" in self.state_space:
                    obs.append(q)
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    # transform the voltage phase to radian
                    obs += list(
                        zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180
                    )
                obs_zone_dict[zone] = np.array(obs)
            obs_len_list.append(obs_zone_dict[zone].shape[0])
        return obs_zone_dict, zone_list, obs_len_list
