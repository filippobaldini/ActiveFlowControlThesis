import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.FlowSolver import FlowSolver
from env.probes import PressureProbeValues, VelocityProbeValues, TotalrecirculationArea
from dolfin import *


def zero_sum_and_bound(action, max_val=1.0):
    action = action - np.mean(action)  # enforce zero net mass flux
    max_magnitude = np.max(np.abs(action))
    if max_magnitude > max_val:
        action = (action / max_magnitude) * max_val
    return action


class RingBuffer:
    "A 1D ring buffer using numpy arrays"

    def __init__(self, length):
        self.data = np.zeros(length, dtype="f")
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


class cylinderenv(gym.Env):
    # metadata = {'render.modes': ['human']}

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
        super(cylinderenv, self).__init__()

        # Store parameters
        self.geometry_params = geometry_params
        self.flow_params = flow_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params
        self.verbose = verbose
        self.n_iter_make_ready = n_iter_make_ready
        self.size_history = size_history
        self.reward_function = reward_function
        self.size_time_state = size_time_state
        self.number_steps_execution = number_steps_execution
        self.simu_name = simu_name
        self.episode_number = 0
        self.epoch_step = 0
        self.epoch_size = optimization_params["step_per_epoch"]

        # Initialize the action and observation spaces
        self.num_control_terms = self.geometry_params["num_control_jets"]

        self.low = np.full(
            (self.num_control_terms,),
            self.optimization_params["min_value_jet_Q"],
            dtype=np.float32,
        )
        self.high = np.full(
            (self.num_control_terms,),
            self.optimization_params["max_value_jet_Q"],
            dtype=np.float32,
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
            obs_shape = (
                num_probes * self.optimization_params["num_steps_in_pressure_history"],
            )
        elif self.output_params["probe_type"] == "velocity":
            obs_shape = (
                2
                * num_probes
                * self.optimization_params["num_steps_in_pressure_history"],
            )
        else:
            raise ValueError("Invalid probe_type specified.")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Initialize other variables
        self.start_class()


    def start_class(self):
        self.solver_step = 0

        self.area_probe = None

        self.history_parameters = {}

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

        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        # ------------------------------------------------------------------------
        # create the flow simulation object
        self.flow = FlowSolver(
            self.flow_params, self.geometry_params, self.solver_params
        )

        # ------------------------------------------------------------------------
        # Setup probes
        if self.output_params["probe_type"] == "pressure":
            self.ann_probes = PressureProbeValues(
                self.flow, self.output_params["locations"]
            )

        elif self.output_params["probe_type"] == "velocity":
            self.ann_probes = VelocityProbeValues(
                self.flow, self.output_params["locations"]
            )
        else:
            raise RuntimeError("unknown probe type")

        # Setup drag measurement
        self.drag, self.lift = self.flow.compute_drag_lift()

        # ------------------------------------------------------------------------
        # No flux from jets for starting
        self.Qs = np.zeros(self.geometry_params["num_control_jets"])
        self.action = np.zeros(self.geometry_params["num_control_jets"])

        # ----------------------------------------------------------------------
        # if reading from disk, show to check everything ok
        if self.n_iter_make_ready is None:
            # Let's start in a random position of the vortex shading
            if self.optimization_params["random_start"]:
                rd_advancement = np.random.randint(650)
                for j in range(rd_advancement):
                    self.flow.evolve(self.Qs)
                print(
                    "Simulated {} iterations before starting the control".format(
                        rd_advancement
                    )
                )

            self.u_, self.p_ = self.flow.evolve(self.Qs)

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.flow.compute_drag_lift()[0]
            self.lift = self.flow.compute_drag_lift()[1]

            self.initial_drag = self.drag
            self.initial_lift = self.lift

            self.write_history_parameters()

        self.ready_to_use = True

    def step(self, action):

        if action.shape == ():
            action = action.reshape((self.num_control_terms,))

        if self.verbose > 1:
            print("--- call step ---")

        self.epoch_step += 1
        print("Step number:", self.epoch_step)

        # Ensure action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # print("Action :", action)

        self.previous_action = self.action

        print("Jet action before zero-net:", action)

        # Enforce zero net mass flow rate if specified
        if self.optimization_params["zero_net_Qs"]:

            action = zero_sum_and_bound(
                action, max_val=self.optimization_params["max_value_jet_Q"]
            )

            print("Jet action after net:", action)

        self.action = action

        # Initialize reward
        reward = 0.0

        # Execute several numerical integration steps
        for crrt_action_nbr in range(self.number_steps_execution):
            # Apply smooth control if specified
            if self.optimization_params["smooth_control"]:
                control_progress = (crrt_action_nbr + 1) / self.number_steps_execution
                action = self.previous_action + control_progress * (
                    self.action - self.previous_action
                )
                # print("Smooth action:", action)

            self.u_, self.p_ = self.flow.evolve(action)

            # Increment solver step
            self.solver_step += self.number_steps_execution

            # Sample probes and recirculation area
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()

            # # Write to history buffers
            reward += self.compute_reward()

            self.write_history_parameters()

        # Get the next state
        next_state = self.build_observation()

        # Compute reward
        step_reward = reward / self.number_steps_execution

        # Check if the episode is done
        self.done = self.episode_done()

        cd, cl = self.flow.compute_drag_lift()

        # Additional info
        info = {"drag": cd, "lift": cl}

        if self.verbose > 1:
            print("--- done step ---")

        terminated = self.done
        truncated = False  # or add logic for timeouts etc.

        return next_state, step_reward, terminated, truncated, info

    def episode_done(self):
        done = self.epoch_step >= self.epoch_size
        return done

    def compute_reward(self):
        lambda_ = 0.01
        eta = 0.05
        drag_coeff, lift_coeff = self.flow.compute_drag_lift()
        if np.isnan(drag_coeff):
            with XDMFFile("results/broken_u.xdmf") as f:
                f.write_checkpoint(self.flow.u_, "u", 0, XDMFFile.Encoding.HDF5)
            with XDMFFile("results/broken_p.xdmf") as f:
                f.write_checkpoint(self.flow.p_, "p", 0, XDMFFile.Encoding.HDF5)
        jet_penalty = np.sum(np.abs(self.action))
        reward = -(drag_coeff + eta * lift_coeff**2 + lambda_ * jet_penalty)
        return reward

    def reset(self, *, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.start_class()

        self.done = 0
        self.probes_values = self.ann_probes.sample(
            self.flow.u_n, self.flow.p_n
        ).flatten()
        next_state = self.build_observation()
        self.episode_number += 1
        self.epoch_step = 0
        cd, cl = self.flow.compute_drag_lift()
        info = {"drag": cd, "lift": cl}

        # assert isinstance(next_state, np.ndarray), f"Expected ndarray, got {type(next_state)}"
        # assert next_state.dtype == np.float32, f"Expected float32, got {next_state.dtype}"
        # assert next_state.shape == self.observation_space.shape, f"Expected shape {self.observation_space.shape}, got {next_state.shape}"

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

        return np.array(state, dtype=np.float32)

    def zero_sum_and_bound(action, max_val=1.0):
        action = action - np.mean(action)  # enforce zero net mass flux
        max_magnitude = np.max(np.abs(action))
        if max_magnitude > max_val:
            action = (action / max_magnitude) * max_val
        return action

    def write_history_parameters(self):

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
