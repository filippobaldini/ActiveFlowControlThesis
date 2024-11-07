import torch
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from BackwardFacingStep import BackwardFacingStep
from FlowSolver import FlowSolver
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


verbose = 3
simulation_duration = 0.5
dt = 0.0005
nb_actuations = 80
# Number of time steps
num_steps = int(simulation_duration / dt)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h_dim = 512
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, h_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(h_dim, np.prod(envs.action_space.shape)), std=0.01),
            #nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        print("Input x shape:", x.shape)
        action_mean = self.actor_mean(x)
        print("action_mean shape:", action_mean.shape)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        print("action_logstd shape:", action_logstd.shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = torch.clamp(probs.sample(), min=-1.0, max=1.0)
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# function to impose inflow boundary condition
# parabolic profile normalized with maximum velocity U_in

def profile(mesh, degree):

    bot = mesh.coordinates().min(axis=0)[1]+0.1
    top = mesh.coordinates().max(axis=0)[1]

    # width of inlet channel
    H = top - bot

    # 
    U_in = 1.5

    return Expression(('-4*U_in*(x[1]-bot)*(x[1]-top)/H/H',
                    '0'), bot=bot, top=top, H=H, U_in=U_in, degree=degree)

def squeeze_scale_action(action, env):
    action_env = action.squeeze(0).cpu().numpy()  # Shape: (action_dim,)
    low = env.low          # Array of lower bounds for each action component
    high = env.high        # Array of upper bounds for each action component
    scaled_action = low + (0.5 * (action_env + 1.0) * (high - low))
    return np.clip(scaled_action, low, high)

# Define flow and solver parameters (as before)

geometry_params = {'total_length': 3,
                    'frequency': 1,
                    'total_height' : 0.3,
                    'length_before_control' : 0.95,
                    'control_width' : 0.05,
                    'step_height' : 0.1,
                    'coarse_size': 0.1,
                    'coarse_distance': 0.5,
                    'box_size': 0.05,
                    #central point of control_width (used in the jet_bcs function)
                    'set_freq': 0,
                    'tuning_parameters' : [4.0,1.0,0.0],
                    'clscale': 1,
                    }

flow_params = {'mu': 1E-3,
                  'rho': 1,
                  'inflow_profile': profile,
                  'u_init' : 'mesh/u_init.xdmf',
                  'p_init' : 'mesh/p_init.xdmf'}


solver_params = {'dt': dt,
                    'solver_type': 'lu', # choose between lu(direct) and la_solve(iterative)
                    'preconditioner_step_1': 'default',
                    'preconditioner_step_2': 'amg',
                    'preconditioner_step_3': 'jacobi',
                    'la_solver_step_1': 'gmres',
                    'la_solver_step_2': 'gmres',
                    'la_solver_step_3': 'cg'}


#initialization of the list containing the coordinates of the probes
list_position_probes = []
# we decided to collocate the probes in the more critical region for the recirculation area:
# that is the area below the step.
# It would be likely a good possible improvement to place some probes also in the upper area
positions_probes_for_grid_x = np.linspace(1,2,27)[1:-1]
positions_probes_for_grid_y = np.linspace(0,0.1,6)[1:-1]


for crrt_x in positions_probes_for_grid_x:
    for crrt_y in positions_probes_for_grid_y:
        list_position_probes.append(np.array([crrt_x, crrt_y]))

output_params = {'locations': list_position_probes,
                 'probe_type': 'velocity'
                    }

optimization_params = {"num_steps_in_pressure_history": 1,
                       "step_per_epoch": nb_actuations,
                       "min_value_jet_Q": -1.e0,
                       "max_value_jet_Q": 1.e0,
                       "min_value_jet_frequency": 0.0,
                       "max_value_jet_frequency": 10.,
                       'control_terms': ['Qs','frequency'],
                       "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80), # 80/0.0005*...=0.1
                       "zero_net_Qs": False,
                       "random_start": 0}

inspection_params = {"plot": False,
                    "step": 50,
                    "dump": 100,
                    "range_pressure_plot": [-2.0, 1],
                    "show_all_at_reset": False,
                    "single_run": False
                    }

reward_function = 'recirculation_area'

# Instantiate the flow solver
flow_solver = FlowSolver(flow_params, geometry_params, solver_params)

# Instantiate the environment (needed for action and observation spaces)
env = BackwardFacingStep(flow_params=flow_params,
                         geometry_params=geometry_params,
                         solver_params=solver_params,
                         optimization_params=optimization_params,
                         inspection_params=inspection_params,
                         reward_function=reward_function,
                         output_params=output_params,
                         verbose=verbose)

# Instantiate the agent
agent = Agent(env).to(device)

# Load the saved model
model_path = "runs/BackwardFacingStepv0__2_train__1/2_train.cleanrl_model"  # Replace with your actual model path
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()


# Initialize observation
obs = env.reset(seed=1)[0]  # Assuming env.reset() returns (obs, info)

# Initialize data storage
obs_list = []
action_list = []
reward_list = []

# Initialize XDMF files for u and p
u_xdmf_file = XDMFFile(MPI.comm_world, "output/u.xdmf")
p_xdmf_file = XDMFFile(MPI.comm_world, "output/p.xdmf")
u_xdmf_file.parameters["flush_output"] = True
u_xdmf_file.parameters["functions_share_mesh"] = True
p_xdmf_file.parameters["flush_output"] = True
p_xdmf_file.parameters["functions_share_mesh"] = True

# Optional: Set write interval
write_interval = 1  # Adjust as needed

for step in range(num_steps):
    # Get the current observation from the flow solver
    obs = flow_solver.get_observation(env)
    
    # Convert observation to tensor and add batch dimension
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    
    # Get action from the agent
    with torch.no_grad():
        action_batched, _, _, _ = agent.get_action_and_value(obs_tensor)
    
    action_np = squeeze_scale_action(action_batched,env)
    
    # Apply the control action to the flow solver
    jet_bc_values = action_np  # [Q, frequency]
    
    # Evolve the flow solver by one time step with the control action
    u_, p_ = flow_solver.evolve(jet_bc_values)
    
  
    u_xdmf_file.write(u_, step)
    p_xdmf_file.write(p_, step)
    
    # Optionally, compute reward
    reward = -env.area_probe.sample(u_,p_)
    
    # Collect data
    obs_list.append(obs)
    action_list.append(action_np)
    reward_list.append(reward)
    
    # Optionally, print progress
    print(f"Step: {step}, Action: {action_np}, Reward: {reward}")

# Close XDMF files
u_xdmf_file.close()
p_xdmf_file.close()

# Convert collected data to arrays
obs_array = np.array(obs_list)
action_array = np.array(action_list)
reward_array = np.array(reward_list)

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(reward_array, label='Reward')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()