import numpy as np
import os
import sys
import shutil
import csv
import time
from dolfin import Expression
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from agents.td3 import TD3  # Import your TD3 agent
from agents.utils import (
    ReplayBuffer,
)  # Assuming you have a ReplayBuffer implemented in utils.py
from env.BackwardFacingStep import BackwardFacingStep  # Import your custom environment

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--control", type=str, default="amplitude")  # per BFS
args = parser.parse_args()
control_mode = args.control 

if control_mode == "amplitude":
    set_freq = 1
elif control_mode == "ampfreq":
    set_freq = 0

# Save the parameters to a JSON file
def save_args(data, dir):

    # Path for the args.json file
    args_file_path = os.path.join(dir, "hyperparameters.json")

    # Save args as JSON
    with open(args_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Parameters saved to {args_file_path}")


hyperparameters = {
    "ENV_NAME": "BackwardFacingStep",  # Replace with your environment name
    "SEED": 42,
    "EPISODES": 500,
    "ACTUATIONS": 80,
    "BATCH_SIZE": 64,
    "DISCOUNT": 0.98,
    "TAU": 0.005,
    "POLICY_NOISE": 0.2,
    "NOISE_CLIP": 0.1,
    "POLICY_FREQ": 2,
    "NOISE_CLIP_FLAG": True,
    "H_DIM": 512,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-4,
    "PARAMETRIC": False,
    "WALL_JETS": False,
    "EARLY_STOPPING": True,  # Early stopping flag
    "EARLY_STOPPING_PATIENCE": 150,  # Number of episodes to wait before stopping
    "REWARD_FUNCTION": "recirc_area",
}

# Access hyperparameters
ENV_NAME = hyperparameters["ENV_NAME"]
SEED = hyperparameters["SEED"]
EPISODES = hyperparameters["EPISODES"]
ACTUATIONS = hyperparameters["ACTUATIONS"]
BATCH_SIZE = hyperparameters["BATCH_SIZE"]
DISCOUNT = hyperparameters["DISCOUNT"]
TAU = hyperparameters["TAU"]
POLICY_NOISE = hyperparameters["POLICY_NOISE"]
NOISE_CLIP = hyperparameters["NOISE_CLIP"]
POLICY_FREQ = hyperparameters["POLICY_FREQ"]
NOISE_CLIP_FLAG = hyperparameters["NOISE_CLIP_FLAG"]
H_DIM = hyperparameters["H_DIM"]
LR_A = hyperparameters["LR_ACTOR"]
LR_C = hyperparameters["LR_CRITIC"]
PARAMETRIC = hyperparameters["PARAMETRIC"]
WALL_JETS = hyperparameters["WALL_JETS"]
REWARD_FUNCTION = hyperparameters["REWARD_FUNCTION"]



simulation_duration = 2.0  # cfd simulation length
dt = 0.0005

time_steps = int(simulation_duration / dt)

geometry_params = {
    "total_length": 3,
    "frequency": 1,
    "total_height": 0.3,
    "length_before_control": 0.95,
    "control_width": 0.05,
    "step_height": 0.1,
    "coarse_size": 0.1,
    "coarse_distance": 0.5,
    "box_size": 0.05,
    # central point of control_width (used in the jet_bcs function)
    "set_freq": set_freq,
    "fixed_freq": 0.01,
    "tuning_parameters": [6.0, 1.0, 0.0],
    "clscale": 1,
    "wall_jets": WALL_JETS,
}


def profile(mesh, degree=2):
    bot = mesh.coordinates().min(axis=0)[1] + 0.1
    top = mesh.coordinates().max(axis=0)[1]
    H = top - bot

    Um = 1.5

    return Expression(
        ("-4*Um*(x[1]-bot)*(x[1]-top)/H/H", "0"),
        bot=bot,
        top=top,
        H=H,
        Um=Um,
        degree=degree,
    )


flow_params = {
    "mu": 1e-3,
    "mu_list": [3e-4, 1.5e-3, 2.5e-4, 5e-4, 7.5e-4, 1e-3],
    "rho": 1,
    "inflow_profile": profile,
    "u_init": "mesh/init/u_init.xdmf",
    "p_init": "mesh/init/p_init.xdmf",
    "parametric": PARAMETRIC,
}


solver_params = {
    "dt": dt,
    "solver_type": "lu",  # choose between lu(direct) and la_solve(iterative)
    "preconditioner_step_1": "default",
    "preconditioner_step_2": "amg",
    "preconditioner_step_3": "jacobi",
    "la_solver_step_1": "gmres",
    "la_solver_step_2": "gmres",
    "la_solver_step_3": "cg",
}


# initialization of the list containing the coordinates of the probes
list_position_probes = []
# collocate the probes in the more critical region for the recirculation area:
# that is the area below the step.
# It would be likely a good possible improvement to place some probes also in the upper area
positions_probes_for_grid_x = np.linspace(1, 2, 27)[1:-1]
positions_probes_for_grid_y = np.linspace(0, 0.1, 6)[1:-1]


for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))

output_params = {"locations": list_position_probes, "probe_type": "velocity"}

optimization_params = {
    "num_steps_in_pressure_history": 2,
    "step_per_epoch": ACTUATIONS,
    "min_value_jet_Q": -1.0,
    "max_value_jet_Q": 1.0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": False,  # (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
    "zero_net_Qs": False,
    "random_start": 1,
}

inspection_params = {
    "plot": False,
    "step": 50,
    "dump": 100,
    "range_pressure_plot": [-2.0, 1],
    "show_all_at_reset": False,
    "single_run": False,
}

reward_function = REWARD_FUNCTION

verbose = 5

number_steps_execution = int(time_steps / ACTUATIONS)


# Get the current date and time
current_time = datetime.now()

# Format the timestamp as a string
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

env = BackwardFacingStep(
    flow_params=flow_params,
    geometry_params=geometry_params,
    solver_params=solver_params,
    optimization_params=optimization_params,
    inspection_params=inspection_params,
    reward_function=reward_function,
    output_params=output_params,
    verbose=verbose,
    number_steps_execution=number_steps_execution,
)

np.random.seed(SEED)


state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
max_action = env.high[0]

# env = gym.wrappers.FlattenObservation(env)

# Initialize TD3 agent
agent = TD3(state_dim, action_dim, env.action_space, LR_A, LR_C, h_dim=H_DIM)

run_name = f"BackwardFacingSteptd3__{timestamp}"


if geometry_params["set_freq"]:
    run_name = f"BackwardFacingSteptd3__no_freq__{timestamp}"


if flow_params["parametric"]:
    run_name = f"BackwardFacingSteptd3__parametric__{timestamp}"
    if geometry_params["set_freq"]:
        run_name = f"BackwardFacingSteptd3__parametric__no_freq__{timestamp}"


# Create the model directory (run directory)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(root, "runs", run_name)
os.makedirs(model_dir, exist_ok=True)

save_args(hyperparameters, model_dir)

# Copy 'plot_episode_reward.py' to 'model_dir'
source_file = os.path.join(os.getcwd(), "bfs", "utils", "plot_episode_reward.py")
destination_file = os.path.join(model_dir, "plot_episode_reward.py")

try:
    shutil.copy(source_file, destination_file)
    print(f"Copied {source_file} to {destination_file}")
except FileNotFoundError:
    print(f"File {source_file} not found.")
except Exception as e:
    print(f"Error copying file: {e}")

# Initialize the CSV file in the run directory
csv_file_path = os.path.join(model_dir, "episode_rewards.csv")
csv_file = open(csv_file_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Episode", "Reward"])

iterations = 4
# Initialize replay buffer
replay_buffer = ReplayBuffer(state_dim, action_dim)

best_reward = -np.inf
start_time = time.time()

cfd_time = 0.0
opt_time = 0.0

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    tot_recirculation_area = 0.0

    policy = []

    for step in range(ACTUATIONS):
        cfd_start_time = time.time()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        cfd_end_time = time.time()
        cfd_time += cfd_end_time - cfd_start_time
        replay_buffer.add(state, action, next_state, reward, done)

        policy.append(action)

        state = next_state
        episode_reward += reward  # /ACTUATIONS

        tot_recirculation_area += env.recirc_area

        if done:
            break

        opt_start_time = time.time()
        if replay_buffer.size > 4 * BATCH_SIZE:
            agent.train(
                replay_buffer,
                iterations,
                BATCH_SIZE,
                DISCOUNT,
                POLICY_NOISE,
                NOISE_CLIP,
                POLICY_FREQ,
                NOISE_CLIP_FLAG,
            )
        opt_end_time = time.time()
        opt_time += opt_end_time - opt_start_time

    # Early stopping condition
    if hyperparameters["EARLY_STOPPING"]:

        if episode_reward > best_reward:

            best_reward = episode_reward
            best_policy = policy
            best_episode = episode
            agent.save("model_best", model_dir)

        elif episode - best_episode >= hyperparameters["EARLY_STOPPING_PATIENCE"]:
            print(
                f"Early stopping at episode {episode} with best reward {best_reward} at episode {best_episode}."
            )
            break

    print(f"Episode {episode} done, Total Reward: {episode_reward}")
    writer.writerow([episode, episode_reward])

env.close()

agent.save("model_last", model_dir)

best_path = os.path.join(model_dir, "best_policy.csv")

# Write the list to the CSV file
with open(best_path, "w", newline="") as csvfile:
    writer1 = csv.writer(csvfile)
    # Write the best reward and episode number at the beginning
    writer1.writerow(["Best Reward", best_reward])
    writer1.writerow(["Best Episode", best_episode])
    writer1.writerow([])  # Add an empty row for separation

    i = 1
    writer1.writerow(["Step", "Action"])
    for item in best_policy:
        writer1.writerow([i, item])
        i += 1

end_time = time.time()
total_training_time = end_time - start_time

# Compute hours, minutes, and seconds
hours_total, rem = divmod(total_training_time, 3600)
minutes_total, seconds_total = divmod(rem, 60)

# Convert to hours, minutes, and seconds
cfd_time_hours, rem = divmod(cfd_time, 3600)
cfd_time_minutes, cfd_time_seconds = divmod(rem, 60)
opt_time_hours, rem = divmod(opt_time, 3600)
opt_time_minutes, opt_time_seconds = divmod(rem, 60)

# Formatting the times
formatted_total_time = "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours_total), int(minutes_total), seconds_total
)
formatted_cfd_time = "CFD time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(cfd_time_hours), int(cfd_time_minutes), cfd_time_seconds
)
formatted_opt_time = "Optimization time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(opt_time_hours), int(opt_time_minutes), opt_time_seconds
)
# Specify the file path
timing_file_path = os.path.join(model_dir, "total_timing_data.txt")

# Write the timing data to the file
with open(timing_file_path, "w") as file:
    file.write(
        formatted_total_time
        + "\n"
        + formatted_cfd_time
        + "\n"
        + formatted_opt_time
        + "\n"
    )

# Print the formatted time
print(
    "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours_total), int(minutes_total), seconds_total
    )
)

print(f"📦 Saved outputs to: {model_dir}")
