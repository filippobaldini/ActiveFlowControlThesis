import numpy as np
import torch
import csv
from dolfin import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, scale_action
from env.BackwardFacingStep import BackwardFacingStep
from datetime import datetime

# Load hyperparameters
EPISODES = 250
ACTUATIONS = 80
SEED = 42
H_DIM = 512
DT = 0.0005

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--control", type=str, default="ampfreq")  # per BFS
parser.add_argument("--run-name", type=str, required=True)
args = parser.parse_args()
control_mode = args.control 

if control_mode == "amplitude":
    set_freq = 1
elif control_mode == "ampfreq":
    set_freq = 0


# Setup geometry, flow, solver, and optimization params
# (same as in your training setup)
# Copy them from your `train_ppo.py` if not already modularized

simulation_duration = 2.0  # in seconds
dt = 0.0005

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
    "set_freq": set_freq,
    "fixed_freq": 0.01,
    "tuning_parameters": [6.0, 1.0, 0.0],
    "clscale": 1,
    "wall_jets": False,
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
    "mu_list": [5e-4],
    "rho": 1,
    "inflow_profile": profile,
    "u_init": "mesh/init/u_init.xdmf",
    "p_init": "mesh/init/p_init.xdmf",
    "parametric": False,
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
    "min_value_jet_Q": -1.0e0,
    "max_value_jet_Q": 1.0e0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": False,  # (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
    "zero_net_Qs": False,
    "random_start": 0,
}

inspection_params = {
    "plot": False,
    "step": 50,
    "dump": 100,
    "range_pressure_plot": [-2.0, 1],
    "show_all_at_reset": False,
    "single_run": False,
}

reward_function = "recirculation_area"

# Create environment
env = BackwardFacingStep(
    flow_params=flow_params,
    geometry_params=geometry_params,
    solver_params=solver_params,
    optimization_params=optimization_params,
    inspection_params=inspection_params,
    reward_function="recirculation_area",
    output_params=output_params,
    verbose=1,
    number_steps_execution=int((2.0 / DT) / ACTUATIONS),
)

# Load trained PPO model
agent = PPOAgent(env.observation_space, env.action_space, h_dim=H_DIM)


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "runs", args.run_name, "training.cleanrl_model")
path = os.path.join(base_dir, "runs", args.run_name)
agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
agent.eval()

# Rollout parameters
output_dir = os.path.join(path, "rollout_inspection")
os.makedirs(output_dir, exist_ok=True)

# Open CSV file for logging
csv_path = os.path.join(output_dir, "recirc_area_per_action.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Step", "Reward", "Recirculation Area", "Action"])

# Open XDMF files for u and p
u_outfile = XDMFFile(MPI.comm_world, f"{output_dir}/u_inspection.xdmf")
p_outfile = XDMFFile(MPI.comm_world, f"{output_dir}/p_inspection.xdmf")
for f in (u_outfile, p_outfile):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True


print("\n🔵 Starting zero-control rollout")

csv_path_zero = os.path.join(output_dir, "recirc_area_per_action_zero.csv")
csv_file_zero = open(csv_path_zero, "w", newline="")
csv_writer_zero = csv.writer(csv_file_zero)
csv_writer_zero.writerow(["Step", "Reward", "Recirculation Area", "Action"])


obs_zero, _ = env.reset(seed=SEED)
zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
obs_zero = torch.Tensor(obs_zero)
total_zero_reward = 0.0

for step in range(ACTUATIONS):
    obs_zero, reward, done, _ = env.step(zero_action)
    obs_zero = torch.Tensor(obs_zero)

    recirc_area = reward
    total_zero_reward += reward

    csv_writer_zero.writerow([step, reward, recirc_area, zero_action.tolist()])
    print(
        f"[ZERO] Step {step}: reward={reward:.4f}, recirc_area={recirc_area:.4f}, action={zero_action}"
    )


csv_file_zero.close()


print(f"🟦 Zero-control rollout finished. Total reward: {total_zero_reward:.4f}")
print(f"📂 Results saved in {output_dir}")

# Run rollout
obs, _ = env.reset(seed=SEED)
obs = torch.Tensor(obs)
total_reward = 0.0

for step in range(ACTUATIONS):
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs)
    action_np = action.cpu().numpy()
    scaled_action = scale_action(action_np, env)

    obs, reward, done, _ = env.step(scaled_action)
    obs = torch.Tensor(obs)

    recirc_area = reward
    total_reward += reward

    csv_writer.writerow([step, reward, recirc_area, scaled_action.tolist()])
    print(
        f"Step {step}: reward={reward:.4f}, recirc_area={recirc_area:.4f}, action={scaled_action}"
    )

    u_outfile.write_checkpoint(
        env.u_, "velocity", step, XDMFFile.Encoding.HDF5, append=True
    )
    p_outfile.write_checkpoint(
        env.p_, "pressure", step, XDMFFile.Encoding.HDF5, append=True
    )

print(f"\n✅ Rollout finished. Total reward: {total_reward:.4f}")
csv_file.close()
u_outfile.close()
p_outfile.close()
print(f"\n📂 Results saved in {output_dir}")


import matplotlib.pyplot as plt
import pandas as pd

# Paths to the CSV files
# Correct paths to the CSV files
ppo_path = os.path.join(output_dir, "recirc_area_per_action.csv")
zero_path = os.path.join(output_dir, "recirc_area_per_action_zero.csv")

# Load both CSVs
df_ppo = pd.read_csv(ppo_path)
df_zero = pd.read_csv(zero_path)

# Plotting
plt.figure(figsize=(8, 4))

plt.plot(
    df_zero["Step"],
    -df_zero["Recirculation Area"],
    label="Zero Control",
    linestyle="--",
    color="darkgreen",
)
plt.plot(
    df_ppo["Step"],
    -df_ppo["Recirculation Area"],
    label="PPO Policy",
    color="darkorange",
)

plt.title("Recirculation Area over Control Step", fontsize=14)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Recirculation Area", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
