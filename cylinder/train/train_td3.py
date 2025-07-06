import numpy as np
import os
import sys
import shutil
import csv
import time
from dolfin import Expression
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from agents.td3 import TD3  # Import  TD3 agent
from agents.utils import ReplayBuffer  # Import ReplayBuffer
from env.cylinderenv import cylinderenv  # Import your custom environment


# Save the parameters to a JSON file
def save_args(data, dir):

    # Path for the args.json file
    args_file_path = os.path.join(dir, "hyperparameters.json")

    # Save args as JSON
    with open(args_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Parameters saved to {args_file_path}")



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--control", type=str, default="2jets", choices=["2jets", "3jets"], help="Control mode for the cylinder")
args = parser.parse_args()
control_mode = args.control

if control_mode == "2jets":
    num_control_jets = 2
elif control_mode == "3jets":
    num_control_jets = 3
else:
    raise ValueError(f"Unknown control mode for cylinder: {control_mode}")


hyperparameters = {
    "ENV_NAME": "cylinderenv",  # Replace with your environment name
    "SEED": 42,
    "EPISODES": 250,
    "ACTUATIONS": 80,
    "BATCH_SIZE": 64,
    "DISCOUNT": 0.98,
    "TAU": 0.005,
    "POLICY_NOISE": 0.05,
    "NOISE_CLIP": 0.2,
    "POLICY_FREQ": 4,
    "NOISE_CLIP_FLAG": True,
    "H_DIM": 512,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-4,
    "PARAMETRIC": False,
    "REWARD_FUNCTION": "weighted_drag",
    "EARLY_STOPPING_PATIENCE": 150,  # Set to None to disable early stopping
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
REWARD_FUNCTION = hyperparameters["REWARD_FUNCTION"]
EARLY_STOPPING_PATIENCE = hyperparameters["EARLY_STOPPING_PATIENCE"]

simulation_duration = 15.0  # cfd simulation length
dt = 0.001  # cfd time step

time_steps = int(simulation_duration / dt)

geometry_params = {
    "frequency": 1,
    #    'control_terms' : ['Qs','frequency'],
    "num_control_jets": num_control_jets,
    "radius": 0.25,  # fixed cylinder radius
    "center": [0.0, 0.0],  # center of the cylinder
    "width": np.pi / 18,  # width of the jet
    # 'set_freq': 1,
    "clscale": 1,
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
    "mu": 5e-3,
    "rho": 1.0,
    "inflow_profile": profile,
    "U_in": 1.5,
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

# Parameters
center = np.array([0.0, 0.0])  # Center of the cylinder
radii = [0.3, 0.4]  # Cylinder radius + 2 outer rings
num_probes_per_ring = 32  # Resolution of angular sampling


# Generate probe positions
list_position_probes = []

for r in radii:
    angles = np.linspace(0, 2 * np.pi, num_probes_per_ring, endpoint=False)
    for theta in angles:
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        list_position_probes.append(np.array([x, y]))


x_coords = np.linspace(-0.15, 0.27, 6)  # Start before the cylinder
y_coords = np.linspace(-0.6, 0.6, 7)
for x in x_coords:
    for y in y_coords:
        if (x**2 + y**2) > 0.6**2:  # Only keep points outside the cylinder
            list_position_probes.append(np.array([x, y]))

# Wake probes (downstream)
x_coords = np.linspace(0.3, 3.3, 10)  # Start after the cylinder
y_coords = np.linspace(-0.6, 0.6, 7)
for x in x_coords:
    for y in y_coords:
        list_position_probes.append(np.array([x, y]))

# Store for your probe object or training env
output_params = {"locations": list_position_probes, "probe_type": "pressure"}

optimization_params = {
    "num_steps_in_pressure_history": 2,
    "step_per_epoch": ACTUATIONS,
    "min_value_jet_Q": -1.0,
    "max_value_jet_Q": 1.0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": True,  # (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
    "zero_net_Qs": True,
    "random_start": True,
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

env = cylinderenv(
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

run_name = f"cylinderwakeTD3_{control_mode}_{timestamp}"


# if geometry_params["set_freq"]:
#         run_name = f"BackwardFacingSteptd3__no_freq__{timestamp}"


# if flow_params["parametric"]:
#     run_name = f"BackwardFacingSteptd3__parametric__{timestamp}"
#     if geometry_params["set_freq"]:
#         run_name = f"BackwardFacingSteptd3__parametric__no_freq__{timestamp}"


# Create the model directory (run directory)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(root, "runs", run_name)
os.makedirs(model_dir, exist_ok=True)


save_args(hyperparameters, model_dir)

# Copy 'plot_episode_reward.py' to 'model_dir'
source_file = os.path.join(root, "utils/plot_episode_reward.py")
destination_file = os.path.join(model_dir, "plot_episode_reward.py")

try:
    shutil.copy(source_file, destination_file)
    print(f"Copied {source_file} to {destination_file}")
except FileNotFoundError:
    print(f"File {source_file} not found.")
except Exception as e:
    print(f"Error copying file: {e}")


# Copy 'plot_episode_reward.py' to 'model_dir'
source_drag_file = os.path.join(os.getcwd(), "utils/plot_drag_lift.py")
destination_drag_file = os.path.join(model_dir, "plot_drag_lift.py")

try:
    shutil.copy(source_file, destination_file)
    print(f"Copied {source_drag_file} to {destination_drag_file}")
except FileNotFoundError:
    print(f"File {source_file} not found.")
except Exception as e:
    print(f"Error copying file: {e}")


# Initialize the CSV file in the run directory
csv_file_path = os.path.join(model_dir, "episode_rewards.csv")
csv_file = open(csv_file_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Episode", "Reward"])

csv_coeffs_path = os.path.join(model_dir, "drag_lift.csv")
csv_coeff_file = open(csv_coeffs_path, "w", newline="")
writerdl = csv.writer(csv_coeff_file)
writerdl.writerow(["Episode", "Cd", "Cl"])

iterations = 4
# Initialize replay buffer
replay_buffer = ReplayBuffer(state_dim, action_dim)

best_reward = -np.inf
start_time = time.time()

baseline_reward = 0.0
episode_cd = 0.0
episode_cl = 0.0
baseline_drag = []
# Calculate baseline reward
for step in range(ACTUATIONS):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_state, reward, done, _, infos = env.step(action)
    cd0, cl0 = env.flow.compute_drag_lift()
    baseline_drag.append(cd0)
    episode_cd += cd0
    episode_cl += cl0
    print(f"Cd: {cd0}, Cl: {cl0}")

    baseline_reward += reward / ACTUATIONS

print(f"Baseline reward: {baseline_reward:.4f}")

baseline_drag_path = os.path.join(model_dir, "baseline_drag_steps.csv")


with open(baseline_drag_path, "w", newline="") as csvfile:
    writer3 = csv.writer(csvfile)
    i = 1
    writer3.writerow(["Step", "Drag"])
    for item in baseline_drag:
        writer3.writerow([i, item])
        i += 1

episode_cd = episode_cd / ACTUATIONS  # Average CD over the episode
print(f"Baseline CD: {episode_cd:.4f}")
episode_cl = episode_cl / ACTUATIONS  # Average CL over the episode
print(f"Baseline CL: {episode_cl:.4f}")

writer.writerow([0, baseline_reward])
writerdl.writerow([0, episode_cd, episode_cl])

optimization_time = 0.0
cfd_time = 0.0

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0

    episode_cd = 0.0
    episode_cl = 0.0

    drag_list = []
    policy = []

    for step in range(ACTUATIONS):
        cfd_time_start = time.time()

        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        cfd_time_end = time.time()
        cfd_time += cfd_time_end - cfd_time_start
        replay_buffer.add(state, action, next_state, reward, done)

        policy.append(action)

        cd, cl = env.flow.compute_drag_lift()

        episode_cd += cd
        episode_cl += cl

        print(f"Reward: {reward}, CD: {cd}")

        drag_list.append(cd)

        state = next_state
        episode_reward += reward / ACTUATIONS

        if done:
            break

        episode_optimization_time_start = time.time()

        if replay_buffer.size > 5 * BATCH_SIZE:

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

        episode_optimization_time_end = time.time()
        episode_optimization_time = (
            episode_optimization_time_end - episode_optimization_time_start
        )
        optimization_time += episode_optimization_time

    # Early stopping condition
    if EARLY_STOPPING_PATIENCE is not None:

        if episode_reward > best_reward:

            best_reward = episode_reward
            best_policy = policy
            best_episode = episode
            best_drag = drag_list
            print(f"New best reward: {best_reward} at episode {episode}. Saving model.")
            agent.save("model_best", model_dir)

        elif episode - best_episode >= hyperparameters["EARLY_STOPPING_PATIENCE"]:
            print(
                f"Early stopping at episode {episode} with best reward {best_reward} at episode {best_episode}."
            )
            break

    episode_cd = episode_cd / ACTUATIONS  # Average CD over the episode
    episode_cl = episode_cl / ACTUATIONS  # Average CL over the episode
    print(
        f"Episode: {episode+1}, Reward: {episode_reward}, CD: {episode_cd}, CL: {episode_cl}"
    )
    writer.writerow([episode + 1, episode_reward])
    writerdl.writerow([episode + 1, episode_cd, episode_cl])


agent.save("model_", model_dir)


env.close()

best_path = os.path.join(model_dir, "best_policy.csv")
drag_path = os.path.join(model_dir, "drag_steps.csv")


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

with open(drag_path, "w", newline="") as csvfile:
    writer2 = csv.writer(csvfile)
    # Write the best reward and episode number at the beginning
    writer2.writerow(["Best Reward", best_reward])
    writer2.writerow(["Best Episode", best_episode])
    writer2.writerow([])  # Add an empty row for separation

    i = 1
    writer2.writerow(["Step", "Drag"])
    for item in best_drag:
        writer2.writerow([i, item])
        i += 1

# Compute hours, minutes, and seconds
hours_total, rem = divmod(total_training_time, 3600)
minutes_total, seconds_total = divmod(rem, 60)

# Convert to hours, minutes, and seconds
optim_hours, rem = divmod(optimization_time, 3600)
optim_minutes, optim_seconds = divmod(rem, 60)

cfd_hours, rem = divmod(cfd_time, 3600)
cfd_minutes, cfd_seconds = divmod(rem, 60)

formatted_optim_time = "Optimization time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(optim_hours), int(optim_minutes), optim_seconds
)

formatted_cfd_time = "CFD time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(cfd_hours), int(cfd_minutes), cfd_seconds
)

# Formatting the times
formatted_total_time = "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours_total), int(minutes_total), seconds_total
)

# Specify the file path
timing_file_path = os.path.join(model_dir, "total_timing_data.txt")

# Write the timing data to the file
with open(timing_file_path, "w") as file:
    file.write(
        formatted_total_time
        + "\n"
        + formatted_optim_time
        + "\n"
        + formatted_cfd_time
        + "\n"
    )

# Print the formatted time
print(
    "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours_total), int(minutes_total), seconds_total
    )
)
