
import numpy as np
import matplotlib.pyplot as plt
import torch
from dolfin import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.BackwardFacingStep import BackwardFacingStep
from agents.ppo import PPOAgent, scale_action

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

# ---- Config ----

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "runs", args.run_name, "training.cleanrl_model")
path = os.path.join(base_dir, "runs", args.run_name)


mu_list = [1e-3, 6.66e-4, 5e-4, 3.33e-4]  # Order matters for plotting
ACTUATIONS = 80
DT = 0.0005
SEED = 42
H_DIM = 512
simulation_duration = 2.0  # in seconds
dt = DT
total_steps = int((2.0 / DT) / ACTUATIONS)
results_dir = os.path.join(path, "rollout_mu_inspection")
os.makedirs(results_dir, exist_ok=True)



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

verbose = 5

number_steps_execution = int((simulation_duration / dt) / ACTUATIONS)

# Dummy env for loading agent
dummy_env = BackwardFacingStep(
    geometry_params,
    flow_params,
    solver_params,
    output_params,
    optimization_params,
    inspection_params,
    number_steps_execution=number_steps_execution,
)
agent = PPOAgent(dummy_env.observation_space, dummy_env.action_space, h_dim=H_DIM)
agent.load_state_dict(torch.load(model_path, map_location="cpu"))
agent.eval()

for mu in mu_list:
    print(f"\n🧪 Testing μ = {mu:.1e}")
    flow_params = {
        "mu": mu,
        "rho": 1.0,
        "inflow_profile": profile,
        "u_init": "mesh/init/u_init.xdmf",
        "p_init": "mesh/init/p_init.xdmf",
        "parametric": False,
    }

    env = BackwardFacingStep(
        geometry_params,
        flow_params,
        solver_params,
        output_params,
        optimization_params,
        inspection_params,
        number_steps_execution=number_steps_execution,
    )

    mu_tag = f"mu_{mu:.0e}"
    mu_dir = os.path.join(results_dir, mu_tag)
    os.makedirs(mu_dir, exist_ok=True)

    # XDMF writers for u, p
    u_out = XDMFFile(MPI.comm_world, os.path.join(mu_dir, "u_control.xdmf"))
    p_out = XDMFFile(MPI.comm_world, os.path.join(mu_dir, "p_control.xdmf"))
    for f in (u_out, p_out):
        f.parameters["flush_output"] = True
        f.parameters["functions_share_mesh"] = True

    ### 1. Zero-control run
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs_zero, _ = env.reset(seed=SEED)
    recirc_zero = []

    zero_episode_reward = 0.0

    for zero_step in range(ACTUATIONS):
        obs_zero, reward, _, _ = env.step(zero_action)
        recirc_zero.append(-reward)
        print(f"Step {zero_step+1}/{ACTUATIONS}, Reward: {reward:.4f}")
        zero_episode_reward += reward

    print(f"Total Reward (Zero Control): {zero_episode_reward:.4f}")

    ### 2. Controlled run
    obs, _ = env.reset(seed=SEED)
    obs = torch.Tensor(obs)
    recirc_ctrl = []

    episode_reward = 0.0

    for step in range(ACTUATIONS):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        scaled_action = scale_action(action.cpu().numpy(), env)

        obs, reward, _, _ = env.step(scaled_action)
        obs = torch.Tensor(obs)

        episode_reward += reward

        print(f"Step {step+1}/{ACTUATIONS}, Reward: {reward:.4f}")

        recirc_ctrl.append(-reward)
        u_out.write_checkpoint(
            env.u_, "velocity", step, XDMFFile.Encoding.HDF5, append=True
        )
        p_out.write_checkpoint(
            env.p_, "pressure", step, XDMFFile.Encoding.HDF5, append=True
        )

    print(
        f"Total Reward: {episode_reward:.4f}, Non controlled Reward: {zero_episode_reward:.4f}"
    )

    u_out.close()
    p_out.close()
    env.close()

    ### Plot recirc vs step
    steps = np.arange(ACTUATIONS)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, recirc_zero, label="Zero Control", linestyle="--", color="green")
    plt.plot(steps, recirc_ctrl, label="Controlled", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Recirculation Area")
    plt.title(f"Recirculation Area per Step (μ = {mu:.0e})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mu_dir, f"recirc_vs_step_{mu_tag}.png"))
    plt.close()
