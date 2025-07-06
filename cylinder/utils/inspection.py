import numpy as np
import torch
from cylinderenv import cylinderenv
from dolfin import *
from td3 import TD3  # Assuming this is your TD3 class
import os


hyperparameters = {
    "ENV_NAME": "BackwardFacingStep",  # Replace with your environment name
    "SEED": 42,
    "EPISODES": 250,
    "ACTUATIONS": 100,
    "BATCH_SIZE": 64,
    "DISCOUNT": 0.95,
    "TAU": 0.005,
    "POLICY_NOISE": 0.5,
    "NOISE_CLIP": 0.2,
    "POLICY_FREQ": 2,
    "NOISE_CLIP_FLAG": True,
    "H_DIM": 512,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "PARAMETRIC": False,
    "WALL_JETS": False,
    "REWARD_FUNCTION": "weighted_drag",
    "NB_JETS": 3,
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
NB_JETS = hyperparameters["NB_JETS"]

simulation_duration = 5.0  # cfd simulation length
dt = 0.0005

time_steps = int(simulation_duration / dt)

geometry_params = {
    "frequency": 1,
    #    'control_terms' : ['Qs','frequency'],
    "num_control_jets": NB_JETS,
    "jet_angle": [0.0, 105.0, 255.0],  # jet angles
    "radius": 0.25,  # fixed cylinder radius
    "center": [0.0, 0.0],  # center of the cylinder
    "width": np.pi / 18,  # width of the jet
    # central point of control_width (used in the jet_bcs function)
    # 'set_freq': 1,
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
num_probes_per_ring = 64  # Resolution of angular sampling


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
        if (x**2 + y**2) > 0.25**2:  # Only keep points outside the cylinder
            list_position_probes.append(np.array([x, y]))

# Wake probes (downstream)
x_coords = np.linspace(0.3, 5.0, 10)  # Start after the cylinder
y_coords = np.linspace(-0.6, 0.6, 7)
for x in x_coords:
    for y in y_coords:
        list_position_probes.append(np.array([x, y]))

# Store for your probe object or training env
output_params = {"locations": list_position_probes, "probe_type": "pressure"}

optimization_params = {
    "num_steps_in_pressure_history": 1,
    "step_per_epoch": ACTUATIONS,
    "min_value_jet_Q": -1.0,
    "max_value_jet_Q": 1.0,
    "min_value_jet_frequency": 0.0,
    "max_value_jet_frequency": 10.0,
    "control_terms": ["Qs", "frequency"],
    "smooth_control": False,  # (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
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

# Path to saved model
model_dir = "runs/cylinderwakeTD3__20250425_143841"  # or wherever you saved
model_name = "model_best"

# Load environment
env = cylinderenv(
    flow_params=flow_params,
    geometry_params=geometry_params,
    solver_params=solver_params,
    output_params=output_params,
    optimization_params=optimization_params,
    inspection_params=inspection_params,
    verbose=1,
)

# Set seed and reset environment
state, _ = env.reset(seed=42)

# Load trained agent
agent = TD3(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_space=env.action_space,
)

agent.load(model_name, model_dir)

# Run controlled rollout
total_reward = 0
done = False
step = 0

# Create output directory
output_dir = os.path.join(model_dir, "rollout")

os.makedirs(output_dir, exist_ok=True)

rollout_actuation_nb = 250


# Open XDMF files for writing
u_outfile = XDMFFile(
    MPI.comm_world, f"{output_dir}/u_rollout_{rollout_actuation_nb}steps.xdmf"
)
p_outfile = XDMFFile(
    MPI.comm_world, f"{output_dir}/p_rollout_{rollout_actuation_nb}steps.xdmf"
)

# Configure output settings
u_outfile.parameters["flush_output"] = True
u_outfile.parameters["functions_share_mesh"] = True
p_outfile.parameters["flush_output"] = True
p_outfile.parameters["functions_share_mesh"] = True


while step < rollout_actuation_nb:
    # Select action from policy (no exploration noise)
    action = agent.select_action(np.array(state))
    state, reward, done, info = env.step(action)

    # Save velocity and pressure to XDMF at this step
    u_outfile.write_checkpoint(
        env.u_, "velocity", step, XDMFFile.Encoding.HDF5, append=True
    )
    p_outfile.write_checkpoint(
        env.p_, "pressure", step, XDMFFile.Encoding.HDF5, append=True
    )

    print(f"Step {step}: reward={reward:.4f}, action={action}")
    total_reward += reward
    step += 1

print(f"\n✅ Finished controlled rollout. Total episode reward: {total_reward:.4f}")

u_outfile.close()
p_outfile.close()
print(f"\n✅ XDMF output saved to {output_dir}/")
