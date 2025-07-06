import numpy as np
import torch
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from dolfin import *

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from agents.ppo import PPOAgent, scale_action
from env.cylinderenv import cylinderenv


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--control", type=str, default="2jets", choices=["2jets", "3jets"], help="Control mode for the cylinder")
parser.add_argument("--run-name", type=str, required=True)  # solo per inspection
args = parser.parse_args()
control_mode = args.control

if control_mode == "2jets":
    num_control_jets = 2
elif control_mode == "3jets":
    num_control_jets = 3
else:
    raise ValueError(f"Unknown control mode for cylinder: {control_mode}")


# --- Setup ---
EPISODES = 250
ACTUATIONS = 80
SEED = 42
DT = 0.0005
simulation_duration = 5.0

# Geometry
geometry_params = {
    "frequency": 1,
    "num_control_jets": num_control_jets,
    "radius": 0.25,
    "center": [0.0, 0.0],
    "width": np.pi / 18,
    "clscale": 1,
}


# Flow
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
    "parametric": False,
}

# Solver
solver_params = {
    "dt": DT,
    "solver_type": "lu",
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
    "min_value_jet_Q": -1.0,
    "max_value_jet_Q": 1.0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs"],
    "smooth_control": True,
    "zero_net_Qs": True,
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

reward_function = "weighted_drag"
number_steps_execution = int((simulation_duration / DT) / ACTUATIONS)

# --- Environment & Model ---
env = cylinderenv(
    flow_params=flow_params,
    geometry_params=geometry_params,
    solver_params=solver_params,
    optimization_params=optimization_params,
    inspection_params=inspection_params,
    reward_function=reward_function,
    output_params=output_params,
    verbose=1,
    number_steps_execution=number_steps_execution,
)

agent = PPOAgent(
    observation_space=env.observation_space, action_space=env.action_space, h_dim=512
)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "runs", args.run_name, "cylinder_best_model.cleanrl_model")
path = os.path.join(base_dir, "runs", args.run_name)

agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
agent.eval()

output_dir = os.path.join(path, "rollout_inspection")
os.makedirs(output_dir, exist_ok=True)

# --- Zero Control Rollout ---
print("\n🔵 Starting zero-control rollout")
csv_path_zero = os.path.join(output_dir, "recirc_area_per_action_zero.csv")
with open(csv_path_zero, "w", newline="") as fzero:
    writer = csv.writer(fzero)
    writer.writerow(["Step", "Reward", "CD", "CL", "Action"])
    obs, _ = env.reset(seed=SEED)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    total_zero_reward = 0.0
    for step in range(ACTUATIONS):
        obs, reward, done, _, _ = env.step(zero_action)
        total_zero_reward += reward
        cd, cl = env.flow.compute_drag_lift()
        writer.writerow([step, reward, cd, cl, zero_action.tolist()])
        print(f"[ZERO] Step {step}: reward={reward:.4f}")
print(f"🟦 Zero-control rollout finished. Total reward: {total_zero_reward:.4f}")

# --- PPO Rollout ---
print("\n🟠 Starting PPO rollout")
csv_path = os.path.join(output_dir, "recirc_area_per_action.csv")
u_outfile = XDMFFile(MPI.comm_world, os.path.join(output_dir, "u_inspection.xdmf"))
p_outfile = XDMFFile(MPI.comm_world, os.path.join(output_dir, "p_inspection.xdmf"))
for f in (u_outfile, p_outfile):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True

with open(csv_path, "w", newline="") as fppo:
    writer = csv.writer(fppo)
    writer.writerow(["Step", "Reward", "CD", "CL", "Action"])
    obs, _ = env.reset(seed=SEED)
    obs = torch.Tensor(obs)
    total_reward = 0.0
    for step in range(ACTUATIONS):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        action_np = action.cpu().numpy()
        scaled_action = scale_action(action_np, env)
        obs, reward, done, _, _ = env.step(scaled_action)
        cd, cl = env.flow.compute_drag_lift()
        obs = torch.Tensor(obs)
        total_reward += reward
        writer.writerow([step, reward, cd, cl, scaled_action.tolist()])
        print(f"[PPO] Step {step}: reward={reward:.4f}")
        u_outfile.write_checkpoint(
            env.u_, "velocity", step, XDMFFile.Encoding.HDF5, append=True
        )
        p_outfile.write_checkpoint(
            env.p_, "pressure", step, XDMFFile.Encoding.HDF5, append=True
        )
u_outfile.close()
p_outfile.close()
print(f"✅ PPO rollout finished. Total reward: {total_reward:.4f}")

# --- Plotting ---
df_ppo = pd.read_csv(csv_path)
df_zero = pd.read_csv(csv_path_zero)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(
    df_zero["Step"],
    df_zero["CD"],
    label="Zero Control",
    linestyle="--",
    color="darkgreen",
)
plt.plot(df_ppo["Step"], df_ppo["CD"], label="TD3 Policy", color="darkorange")
plt.xlabel("Step")
plt.ylabel("Drag Coefficient (Cd)")
plt.title("Cd over Time")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    df_zero["Step"],
    df_zero["CL"],
    label="Zero Control",
    linestyle="--",
    color="darkgreen",
)
plt.plot(df_ppo["Step"], df_ppo["CL"], label="TD3 Policy", color="darkorange")
plt.xlabel("Step")
plt.ylabel("Lift Coefficient (Cl)")
plt.title("Cl over Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cd_cl_comparison.png"))
plt.show()
