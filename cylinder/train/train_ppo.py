import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from datetime import datetime
import json
import sys
import shutil
from dolfin import Expression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, scale_action

from env.cylinderenv import cylinderenv


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
    "ENV_NAME": "BackwardFacingStep",  # Replace with your environment name
    "SEED": 42,
    "CUDA": False,
    "TORCH_DETERMINISTIC": True,
    "EPISODES": 250,
    "ACTUATIONS": 80,
    "BATCH_SIZE": 32,
    "DISCOUNT": 0.95,
    "GAE_LAMBDA": 0.95,
    "H_DIM": 512,
    "LR": 1e-4,
    "NB_MINIBATCHES": 4,
    "ENT_COEF": 0.05,
    "VF_COEF": 0.5,
    "CLIP_COEF": 0.2,
    "UPDATE_EPOCHS": 5,
    "CLIP_VLOSS": True,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "NORM_ADV": True,
    "ANNEAL_LR": False,
    "PATIENCE": 150,
    "PARAMETRIC": False,
    "WALL_JETS": False,
    "REWARD_FUNCTION": "drag_lift_penalization",
    "EARLY_STOPPING": True,
    "SAVE_BEST_MODEL": True,
    "SAVE_MODEL": True,
}

# Access hyperparameters
ENV_NAME = hyperparameters["ENV_NAME"]
SEED = hyperparameters["SEED"]
CUDA = hyperparameters["CUDA"]
TORCH_DETERMINISTIC = hyperparameters["TORCH_DETERMINISTIC"]
EPISODES = hyperparameters["EPISODES"]
ACTUATIONS = hyperparameters["ACTUATIONS"]
BATCH_SIZE = hyperparameters["BATCH_SIZE"]
DISCOUNT = hyperparameters["DISCOUNT"]
GAE_LAMBDA = hyperparameters["GAE_LAMBDA"]
H_DIM = hyperparameters["H_DIM"]
LR = hyperparameters["LR"]
NB_MINIBATCHES = hyperparameters["NB_MINIBATCHES"]
ENT_COEF = hyperparameters["ENT_COEF"]
VF_COEF = hyperparameters["VF_COEF"]
CLIP_COEF = hyperparameters["CLIP_COEF"]
UPDATE_EPOCHS = hyperparameters["UPDATE_EPOCHS"]
CLIP_VLOSS = hyperparameters["CLIP_VLOSS"]
MAX_GRAD_NORM = hyperparameters["MAX_GRAD_NORM"]
TARGET_KL = hyperparameters["TARGET_KL"]
NORM_ADV = hyperparameters["NORM_ADV"]
ANNEAL_LR = hyperparameters["ANNEAL_LR"]
PATIENCE = hyperparameters["PATIENCE"]
PARAMETRIC = hyperparameters["PARAMETRIC"]
WALL_JETS = hyperparameters["WALL_JETS"]
REWARD_FUNCTION = hyperparameters["REWARD_FUNCTION"]
EARLY_STOPPING = hyperparameters["EARLY_STOPPING"]
SAVE_BEST_MODEL = hyperparameters["SAVE_BEST_MODEL"]
SAVE_MODEL = hyperparameters["SAVE_MODEL"]

simulation_duration = 25.0  # duree en secondes de la simulation
dt = 0.001


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
    "rho": 1,
    "inflow_profile": profile,
    "u_init": "mesh/u_init.xdmf",
    "p_init": "mesh/p_init.xdmf",
    "U_in": 1.5,
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

output_params = {"locations": list_position_probes, "probe_type": "pressure"}

optimization_params = {
    "num_steps_in_pressure_history": 2,
    "step_per_epoch": ACTUATIONS,
    "min_value_jet_Q": -1.0e0,
    "max_value_jet_Q": 1.0e0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": True,  # (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
    "zero_net_Qs": True,
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

number_steps_execution = int((simulation_duration / dt) / ACTUATIONS)

# Get the current date and time
current_time = datetime.now()

# Format the timestamp as a string
timestamp = current_time.strftime("%Y%m%d_%H%M%S")


envs = cylinderenv(
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


MINIBATCH_SIZE = int(BATCH_SIZE / NB_MINIBATCHES)
run_name = f"cylinderwakePP0_{control_mode}_{timestamp}"


# Create the model directory (run directory)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(root, "runs", run_name)
os.makedirs(model_dir, exist_ok=True)

# Initialize cumulative timing variables
total_cfd_time = 0.0
total_optimization_time = 0.0

# Copy 'plot_episode_reward.py' to 'model_dir'
source_file = os.path.join(os.getcwd(), "utils/plot_episode_reward.py")
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

save_args(hyperparameters, model_dir)

# TRY NOT TO MODIFY: seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = TORCH_DETERMINISTIC

device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")


print("nb of cntrls terms:", envs.num_control_terms)

# Set seeds
envs.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

assert isinstance(
    envs.action_space, gym.spaces.Box
), "only continuous action space is supported"

agent = PPOAgent(envs.observation_space, envs.action_space, h_dim=H_DIM).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-4)

# ALGO Logic: Storage setup
obs = torch.zeros((ACTUATIONS, 1) + envs.observation_space.shape).to(device)
actions = torch.zeros((ACTUATIONS, 1) + envs.action_space.shape).to(device)
logprobs = torch.zeros((ACTUATIONS, 1)).to(device)
rewards = torch.zeros((ACTUATIONS, 1)).to(device)
dones = torch.zeros((ACTUATIONS, 1)).to(device)
values = torch.zeros((ACTUATIONS, 1)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=SEED)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(1).to(device)

# Initialize the CSV file in the run directory
csv_file_path = os.path.join(model_dir, "episode_rewards.csv")
csv_file = open(csv_file_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Episode", "Reward"])


episode_reward = 0
best_reward = 0


print("Starting reference experiment with zero control action")
for step in range(0, ACTUATIONS):

    # ALGO LOGIC: action logic
    action = torch.zeros((envs.num_control_terms,))

    # TRY NOT TO MODIFY: execute the game and log data.
    # next_obs, terminations, reward, truncations, infos = envs.step(action.cpu().numpy())
    next_obs, reward, terminations, _, info = envs.step(action.cpu().numpy())
    print(reward)
    print(terminations)
    next_done = terminations  # np.logical_or(terminations, truncations)
    rewards[step] = torch.tensor(reward).to(device).view(-1)
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
        [next_done]
    ).to(device)

    episode_reward += reward / ACTUATIONS
    if next_done:
        print("Episode is done!!!")
        print("Episode Reward no control : ", episode_reward)
        # if args.log:
        #     wandb.log({'train/training_reward': episode_reward})
        #     episode_reward = 0


writer.writerow(["0", episode_reward])

if SAVE_BEST_MODEL:
    # Initialize variables for early stopping
    best_reward = float("-inf")  # Best reward seen so far
    patience = 10  # Number of iterations to wait for improvement
    num_iterations_no_improve = 0  # Iterations since the last improvement


print("Starting the real training of the ppo agent")
for iteration in range(1, EPISODES + 1):
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    print("Iteration:", iteration)
    # Annealing the rate if instructed to do so.
    if ANNEAL_LR:
        frac = 1.0 - (iteration - 1.0) / EPISODES
        lrnow = frac * LR
        optimizer.param_groups[0]["lr"] = lrnow
    episode_reward = 0.

    policy = []

    cfd_start = time.time()
    episode_cd = 0.0
    episode_cl = 0.0

    for step in range(0, ACTUATIONS):
        print("Step:", step+1)
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        action_np = action.cpu().numpy()
        action_scaled = scale_action(action_np, envs)

        policy.append(action_scaled)

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, terminations, reward, truncations, infos = envs.step(action.cpu().numpy())
        next_obs, reward, terminations, _, info = envs.step(action_scaled)

        cd,cl = envs.flow.compute_drag_lift()
        episode_cd += cd
        episode_cl += cl

        print(f"Reward: {reward}, CD: {cd}, CL: {cl}")
        # print(terminations)
        next_done = terminations  # np.logical_or(terminations, truncations)
        print(next_done)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            [next_done]
        ).to(device)

        episode_reward += reward / ACTUATIONS


    # bootstrap value if not done
    cfd_end = time.time()
    total_cfd_time += cfd_end - cfd_start

    f"Episode: {iteration+1}, Reward: {episode_reward}, CD: {episode_cd}, CL: {episode_cl}"

    writer.writerow([iteration, episode_reward])
    if SAVE_BEST_MODEL:
        # At the end of each iteration, check if the episode reward improved
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_policy = policy
            # Save the best model
            model_path = os.path.join(model_dir, f"cylinder_best_model.cleanrl_model")
            torch.save(agent.state_dict(), model_path)

    policy_start = time.time()

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(ACTUATIONS)):
            if t == ACTUATIONS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + DISCOUNT * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + DISCOUNT * GAE_LAMBDA * nextnonterminal * lastgaelam
            )
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(BATCH_SIZE)
    clipfracs = []
    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if NORM_ADV:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if CLIP_VLOSS:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if TARGET_KL is not None and approx_kl > TARGET_KL:
            break
    policy_end = time.time()
    total_optimization_time += policy_end - policy_start

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # Specify the file path in the same directory as before
best_path = os.path.join(model_dir, "best_policy.csv")

# Write the list to the CSV file
with open(best_path, "w", newline="") as csvfile:
    writer1 = csv.writer(csvfile)
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

hours_cfd, rem = divmod(total_cfd_time, 3600)
minutes_cfd, seconds_cfd = divmod(rem, 60)

hours_opt, rem = divmod(total_optimization_time, 3600)
minutes_opt, seconds_opt = divmod(rem, 60)

ratio_cfd = total_cfd_time / total_training_time
ratio_opt = total_optimization_time / total_training_time

csv_file.close()
csvfile.close()

# Formatting the times
formatted_total_time = "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours_total), int(minutes_total), seconds_total
)

formatted_cfd_time = "Total CFD simulation time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours_cfd), int(minutes_cfd), seconds_cfd
)

formatted_opt_time = "Total optimization time: {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours_opt), int(minutes_opt), seconds_opt
)

# Specify the file path
timing_file_path = os.path.join(model_dir, "total_timing_data.txt")

# Write the timing data to the file
with open(timing_file_path, "w") as file:
    file.write(formatted_total_time + "\n")
    file.write(formatted_cfd_time + "\n")
    file.write(formatted_opt_time + "\n")
    if ratio_cfd is not None:
        file.write("Ratio CFD/Total training time: {:.6f}\n".format(ratio_cfd))
    else:
        file.write("Ratio CFD/Total training time: N/A (Total training time is zero)\n")
    if ratio_opt is not None:
        file.write("Ratio Optimization/Total training time: {:.6f}\n".format(ratio_opt))
    else:
        file.write(
            "Ratio Optimization/Total training time: N/A (Total training time is zero)\n"
        )

# Print the formatted time
print(
    "Total training time: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours_total), int(minutes_total), seconds_total
    )
)


if SAVE_MODEL:

    # Construct the full model path
    model_path = os.path.join(model_dir, f"cylinder_{num_control_jets}jets.cleanrl_model")

    # Save the model
    try:
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved successfully at {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


envs.close()
# writer.close()
