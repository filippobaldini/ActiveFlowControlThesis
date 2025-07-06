import numpy as np
import csv
from dolfin import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.td3 import TD3
from env.BackwardFacingStep import BackwardFacingStep
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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

# === Parameters ===
ACTUATIONS = 80
SEED = 42
DT = 0.0005

simulation_duration = 2.0
dt = 0.0005

# === Geometry and flow setup ===
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
    "rho": 1,
    "inflow_profile": profile,
    "u_init": "mesh/init/u_init.xdmf",
    "p_init": "mesh/init/p_init.xdmf",
    "parametric": False,
}

solver_params = {
    "dt": dt,
    "solver_type": "lu",
    "preconditioner_step_1": "default",
    "preconditioner_step_2": "amg",
    "preconditioner_step_3": "jacobi",
    "la_solver_step_1": "gmres",
    "la_solver_step_2": "gmres",
    "la_solver_step_3": "cg",
}

positions_probes_for_grid_x = np.linspace(1, 2, 27)[1:-1]
positions_probes_for_grid_y = np.linspace(0, 0.1, 6)[1:-1]
list_position_probes = [
    np.array([x, y])
    for x in positions_probes_for_grid_x
    for y in positions_probes_for_grid_y
]

output_params = {"locations": list_position_probes, "probe_type": "velocity"}

optimization_params = {
    "num_steps_in_pressure_history": 2,
    "step_per_epoch": ACTUATIONS,
    "min_value_jet_Q": -1.0,
    "max_value_jet_Q": 1.0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": False,
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

# === Environment ===
env = BackwardFacingStep(
    flow_params=flow_params,
    geometry_params=geometry_params,
    solver_params=solver_params,
    optimization_params=optimization_params,
    inspection_params=inspection_params,
    reward_function=reward_function,
    output_params=output_params,
    verbose=1,
    number_steps_execution=int((simulation_duration / dt) / ACTUATIONS),
)

# === Load TD3 best model ===
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.join( base_dir,"runs", args.run_name)
agent = TD3(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_space=env.action_space,
)

agent.load("model_best", path)

# === Output setup ===
output_dir = os.path.join(path, "rollout_inspection")
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "recirc_area_per_action.csv")
csv_zero_path = os.path.join(output_dir, "recirc_area_per_action_zero.csv")

u_outfile = XDMFFile(MPI.comm_world, f"{output_dir}/u_inspection.xdmf")
p_outfile = XDMFFile(MPI.comm_world, f"{output_dir}/p_inspection.xdmf")
for f in (u_outfile, p_outfile):
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True

# === Zero-control rollout ===
print("\n🔵 Starting zero-control rollout")
csv_file_zero = open(csv_zero_path, "w", newline="")
csv_writer_zero = csv.writer(csv_file_zero)
csv_writer_zero.writerow(["Step", "Reward", "Recirculation Area", "Action"])

obs_zero, _ = env.reset(seed=SEED)
zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
total_zero_reward = 0.0

for step in range(ACTUATIONS):
    obs_zero, reward, done, _ = env.step(zero_action)
    recirc_area = env.recirc_area
    csv_writer_zero.writerow([step, reward, recirc_area, zero_action.tolist()])
    print(
        f"[ZERO] Step {step}: reward={reward:.4f}, recirc_area={recirc_area:.4f}, action={zero_action}"
    )
    total_zero_reward += reward

csv_file_zero.close()
print(f"🟦 Zero-control rollout finished. Total reward: {total_zero_reward:.4f}")

# === TD3 rollout ===
print("\n🟠 Starting TD3 rollout")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Step", "Reward", "Recirculation Area", "Action"])

state, _ = env.reset(seed=SEED)
total_reward = 0.0

for step in range(ACTUATIONS):
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)

    recirc_area = env.recirc_area
    csv_writer.writerow([step, reward, recirc_area, action.tolist()])
    print(
        f"Step {step}: reward={reward:.4f}, recirc_area={recirc_area:.4f}, action={action}"
    )

    u_outfile.write_checkpoint(
        env.u_, "velocity", step, XDMFFile.Encoding.HDF5, append=True
    )
    p_outfile.write_checkpoint(
        env.p_, "pressure", step, XDMFFile.Encoding.HDF5, append=True
    )
    total_reward += reward

csv_file.close()
u_outfile.close()
p_outfile.close()

print(f"✅ TD3 rollout finished. Total reward: {total_reward:.4f}")
print(f"📂 Results saved in {output_dir}")

# === Plot comparison ===
df_td3 = pd.read_csv(csv_path)
df_zero = pd.read_csv(csv_zero_path)

plt.figure(figsize=(8, 4))
plt.plot(
    df_zero["Step"],
    -df_zero["Reward"],
    label="Zero Control",
    linestyle="--",
    color="darkgreen",
)
plt.plot(df_td3["Step"], -df_td3["Reward"], label="TD3 Policy", color="darkorange")
plt.title("Recirculation Area over Control Step")
plt.xlabel("Step")
plt.ylabel("Recirculation Area")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
