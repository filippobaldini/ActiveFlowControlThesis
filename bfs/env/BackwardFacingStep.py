import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.FlowSolver import FlowSolver
from env.probes import PressureProbeValues, VelocityProbeValues, TotalrecirculationArea
from dolfin import *


class RingBuffer:

    "A 1D ring buffer using numpy arrays to keep track of data with a maximum length"

    def __init__(self, length):
        self.data = np.zeros(length, dtype="f")
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer in case you exceed the length you override restarting from the beginning"
        x_index = (self.index + np.arange(x.size)) % self.data.size

        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer (all of it)"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


class BackwardFacingStep(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        geometry_params,
        flow_params,
        solver_params,
        output_params,
        optimization_params,
        inspection_params,
        n_iter_make_ready=None,
        verbose=0,
        size_history=2000,
        reward_function="plain_drag",
        size_time_state=50,
        number_steps_execution=50,
        simu_name="Simu",
    ):
        super(BackwardFacingStep, self).__init__()

        # Store parameters
        self.geometry_params = geometry_params
        self.flow_params = flow_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params
        self.n_iter_make_ready = n_iter_make_ready
        self.verbose = verbose
        self.size_history = size_history
        self.reward_function = reward_function
        self.size_time_state = size_time_state
        self.number_steps_execution = number_steps_execution
        self.simu_name = simu_name
        self.episode_number = 0
        self.epoch_step = 0
        self.epoch_size = optimization_params["step_per_epoch"]

        min_value_Q = self.optimization_params["min_value_jet_Q"]
        max_value_Q = self.optimization_params["max_value_jet_Q"]

        min_value_frequency = self.optimization_params["min_value_jet_frequency"]
        max_value_frequency = self.optimization_params["max_value_jet_frequency"]

        # Create arrays for low and high bounds
        self.low = np.array([min_value_Q, min_value_frequency], dtype=np.float32)
        self.high = np.array([max_value_Q, max_value_frequency], dtype=np.float32)

        # Initialize the action and observation spaces
        self.num_control_terms = len(self.optimization_params["control_terms"])
        if self.geometry_params["set_freq"]:
            self.num_control_terms -= 1
            self.low = np.array(
                [self.optimization_params["min_value_jet_Q"]], dtype=np.float32
            )
            self.high = np.array(
                [self.optimization_params["max_value_jet_Q"]], dtype=np.float32
            )

        print("Nb of control terms:", self.num_control_terms)
        self.action_space = spaces.Box(
            low=self.low,
            high=self.high,
            shape=(self.num_control_terms,),
            dtype=np.float32,
        )

        num_probes = len(self.output_params["locations"])

        if self.output_params["probe_type"] == "pressure":
            base_dim = (
                num_probes * self.optimization_params["num_steps_in_pressure_history"]
            )
        elif self.output_params["probe_type"] == "velocity":
            base_dim = (
                2
                * num_probes
                * self.optimization_params["num_steps_in_pressure_history"]
            )
        else:
            raise ValueError("Invalid probe_type specified.")

        # If training parametrico: add 1 dimension for viscosity
        if self.flow_params["parametric"]:
            obs_shape = (base_dim + 1,)
        else:
            obs_shape = (base_dim,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Initialize other variables
        self.start_class()

    def start_class(self):
        # Initialize environment variables
        self.episode_number = 0
        self.epoch_step = 0
        self.solver_step = 0
        self.history_parameters = {}

        # Initialize buffers for action terms
        for contr_term in self.optimization_params["control_terms"]:
            self.history_parameters["control_for_{}".format(contr_term)] = RingBuffer(
                self.size_history
            )

        # Number of control terms
        self.history_parameters["number_of_control_terms"] = self.num_control_terms

        # Initialize buffers for probes
        for crrt_probe in range(len(self.output_params["locations"])):
            if self.output_params["probe_type"] == "pressure":
                self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )
            elif self.output_params["probe_type"] == "velocity":
                self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )
                self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(
                    self.size_history
                )

        self.history_parameters["number_of_probes"] = len(
            self.output_params["locations"]
        )
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        if self.n_iter_make_ready is None:
            # initialize by hand parameters that were not present in the file
            if not "number_of_probes" in self.history_parameters:
                self.history_parameters["number_of_probes"] = 0
            if not "number_of_control_terms" in self.history_parameters:
                self.history_parameters["number_of_control_terms"] = len(
                    self.geometry_params["control_terms"]
                )
            if not "recirc_area" in self.history_parameters:
                self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

            # if not the same number of probes, reset them one by one
            if not self.history_parameters["number_of_probes"] == len(
                self.output_params["locations"]
            ):
                for crrt_probe in range(len(self.output_params["locations"])):
                    if self.output_params["probe_type"] == "pressure":
                        self.history_parameters["probe_{}".format(crrt_probe)] = (
                            RingBuffer(self.size_history)
                        )
                    elif self.output_params["probe_type"] == "velocity":
                        self.history_parameters["probe_{}_u".format(crrt_probe)] = (
                            RingBuffer(self.size_history)
                        )
                        self.history_parameters["probe_{}_v".format(crrt_probe)] = (
                            RingBuffer(self.size_history)
                        )

                self.history_parameters["number_of_probes"] = len(
                    self.output_params["locations"]
                )

                self.resetted_number_probes = True

        # Initialize flow solver
        self.flow = FlowSolver(
            self.flow_params, self.geometry_params, self.solver_params
        )

        # Setup probes
        if self.output_params["probe_type"] == "pressure":
            self.ann_probes = PressureProbeValues(
                self.flow, self.output_params["locations"]
            )
        elif self.output_params["probe_type"] == "velocity":
            self.ann_probes = VelocityProbeValues(
                self.flow, self.output_params["locations"]
            )

        self.area_probe = TotalrecirculationArea(self.flow.u_, 0.0)

        # No flux from jets for starting
        if self.geometry_params["set_freq"]:
            self.Qs = np.zeros(1)
            self.frequencies = np.ones(1)
        else:
            [self.Qs, self.frequencies] = np.zeros(
                len(self.optimization_params["control_terms"])
            )

        self.action = np.zeros(len(self.optimization_params["control_terms"]))

        if self.n_iter_make_ready is None:

            # Let's start in a random position of the vortex shading
            if self.optimization_params["random_start"]:
                rd_advancement = np.random.randint(650)
                for j in range(rd_advancement):
                    if self.geometry_params["wall_jets"]:
                        if self.geometry_params["set_freq"]:
                            self.u_, self.p_ = self.flow.two_jets_evolve(
                                np.concatenate((self.Qs, self.frequencies))
                            )
                        else:
                            self.u_, self.p_ = self.flow.two_jets_evolve(
                                [self.Qs, self.frequencies]
                            )

                    # iteration to initialize solution variables

                    else:
                        if self.geometry_params["set_freq"]:
                            self.u_, self.p_ = self.flow.evolve(
                                np.concatenate((self.Qs, self.frequencies))
                            )
                        else:
                            self.u_, self.p_ = self.flow.evolve(
                                [self.Qs, self.frequencies]
                            )
                print(
                    "Simulated {} iterations before starting the control".format(
                        rd_advancement
                    )
                )

            if self.geometry_params["wall_jets"]:
                if self.geometry_params["set_freq"]:
                    self.u_, self.p_ = self.flow.two_jets_evolve(
                        np.concatenate((self.Qs, self.frequencies))
                    )
                else:
                    self.u_, self.p_ = self.flow.two_jets_evolve(
                        [self.Qs, self.frequencies]
                    )

            # iteration to initialize solution variables

            else:
                if self.geometry_params["set_freq"]:
                    self.u_, self.p_ = self.flow.evolve(
                        np.concatenate((self.Qs, self.frequencies))
                    )
                else:
                    self.u_, self.p_ = self.flow.evolve([self.Qs, self.frequencies])

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()

            self.recirc_area = self.area_probe.sample(self.u_, self.p_)
            self.A_init = self.area_probe.sample(self.u_, self.p_)

        # Ready to use
        self.ready_to_use = True

    def step(self, action):

        if action.shape == ():
            action = action.reshape((self.num_control_terms,))

        if self.verbose > 1:
            print("--- call step ---")

        # Ensure action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        self.previous_action = self.action
        self.action = action
        self.Qs = np.array(action[0])
        if self.geometry_params["set_freq"]:
            self.frequencies = np.array(self.geometry_params["fixed_freq"])
        else:
            self.frequencies = np.array(action[1])
        print("Action :", action)

        istantaneous_reward = 0.0

        # Execute several numerical integration steps
        for crrt_action_nbr in range(self.number_steps_execution):
            # Apply smooth control if specified
            k_interp = 10
            if self.optimization_params["smooth_control"]:
                if crrt_action_nbr < k_interp:
                    alpha = (crrt_action_nbr + 1) / k_interp
                    interp_action = (
                        1 - alpha
                    ) * self.previous_action + alpha * self.action
                else:
                    interp_action = self.action
            else:
                interp_action = self.action

            self.Qs = interp_action[0]
            self.frequencies = (
                interp_action[1]
                if not self.geometry_params["set_freq"]
                else self.frequencies
            )
            action = np.array([self.Qs, self.frequencies])

            if self.geometry_params["wall_jets"]:
                self.u_, self.p_ = self.flow.two_jets_evolve(action)
            # Evolve the simulation
            else:
                self.u_, self.p_ = self.flow.evolve(action)

            # Increment solver step
            self.solver_step += self.number_steps_execution

            # Sample probes and recirculation area
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            istantaneous_reward += self.compute_reward()  # /self.A_init

            # Write to history buffers
            self.write_history_parameters()

        # Get the next state

        next_state = self.build_observation()

        self.epoch_step += 1

        reward = istantaneous_reward / self.number_steps_execution

        # Check if the episode is done
        self.done = self.episode_done()

        # Additional info
        info = {}

        if self.verbose > 1:
            print("--- done step ---")

        return next_state, reward, self.done, info

    def episode_done(self):
        done = self.epoch_step >= self.epoch_size
        return done

    def compute_reward(self):
        u = self.u_
        p = self.p_

        recirc_area = self.area_probe.sample(u, p)
        kinetic_energy = assemble(0.5 * inner(u, u) * dx)
        enstrophy = assemble(0.5 * inner(curl(u), curl(u)) * dx)

        # Optional: normalize each term if needed
        w_area = 100.0
        w_energy = 0.001
        w_enstrophy = 0.01

        if self.reward_function == "mixed_functional":
            return -(
                w_area * recirc_area
                + w_energy * kinetic_energy
                + w_enstrophy * enstrophy
            )
        else:
            return -recirc_area

    def reset(self, seed):
        self.start_class()
        self.done = 0
        self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
        # next_state = np.array(self.probes_values)
        next_state = self.build_observation()
        self.episode_number += 1
        self.epoch_step = 0
        info = {}
        return next_state, info

    def build_observation(self):
        state = []
        num_steps = self.optimization_params["num_steps_in_pressure_history"]

        if self.output_params["probe_type"] == "velocity":
            for i in range(len(self.output_params["locations"])):
                u_hist = self.history_parameters[f"probe_{i}_u"].get()
                v_hist = self.history_parameters[f"probe_{i}_v"].get()
                state.extend(u_hist[-num_steps:])
                state.extend(v_hist[-num_steps:])
        elif self.output_params["probe_type"] == "pressure":
            for i in range(len(self.output_params["locations"])):
                p_hist = self.history_parameters[f"probe_{i}"].get()
                state.extend(p_hist[-num_steps:])
        else:
            raise ValueError("Unsupported probe type.")

        # ⬇️ Aggiunta viscosità (parametric training)
        if self.flow_params.get("parametric", False):
            mu = float(self.flow.viscosity)
            state.append(mu)

        return np.array(state, dtype=np.float32)

    def write_history_parameters(self):

        # save actuation values
        self.history_parameters["control_for_Qs"].extend(self.Qs)
        self.history_parameters["control_for_frequency"].extend(self.frequencies)

        # save probes values
        if self.output_params["probe_type"] == "pressure":
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}".format(crrt_probe)].extend(
                    self.probes_values[crrt_probe]
                )
        elif self.output_params["probe_type"] == "velocity":
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}_u".format(crrt_probe)].extend(
                    self.probes_values[2 * crrt_probe]
                )
                self.history_parameters["probe_{}_v".format(crrt_probe)].extend(
                    self.probes_values[2 * crrt_probe + 1]
                )

        # save rewarda value
        self.history_parameters["recirc_area"].extend(np.array(self.recirc_area))
