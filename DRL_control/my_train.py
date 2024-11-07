import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import tyro
from typing import Callable
import wandb


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    log: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    # wandb_project_name: str = "cleanRL"
    # """the wandb's project name"""
    # wandb_entity: str = None
    # """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BackwardFacingStepv0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 80#2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 50#10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    seed: int = 1
    """seed"""

    h_dim: int = 512
    """hidden layer neurons"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    ):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

def squeeze_scale_action(action, env):
    action_env = action.squeeze(0).cpu().numpy()  # Shape: (action_dim,)
    low = env.low          # Array of lower bounds for each action component
    high = env.high        # Array of upper bounds for each action component
    scaled_action = low + (0.5 * (action_env + 1.0) * (high - low))
    return np.clip(scaled_action, low, high)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h_dim = args.h_dim
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


import sys
import os
import shutil
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from BackwardFacingStep import BackwardFacingStep
import numpy as np
from dolfin import Expression
import math

import os
cwd = os.getcwd()

nb_actuations = 80 # Number of neural network actuations per episode

simulation_duration = 2.0 #duree en secondes de la simulation
dt = 0.0005


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


def profile(mesh, degree = 2):
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]
    H = top - bot

    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                    '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)


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

verbose = 3

number_steps_execution = int((simulation_duration/dt)/nb_actuations)



envs = BackwardFacingStep(flow_params=flow_params,
                                         geometry_params=geometry_params,
                                         solver_params=solver_params,
                                         optimization_params=optimization_params,
                                         inspection_params=inspection_params,
                                         reward_function=reward_function,
                                         output_params=output_params,
                                         verbose=verbose)



if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    seed = args.seed

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    # )
    #envs = gym.vector.SyncVectorEnv([env_2d_cylinder])
    #envs = [env_2d_cylinder]
    # envs_list = []
    # for i in range(args.num_envs):
    #     envs_list.append(lambda: env_backwardstep)
    # print("Spawning {} environments".format(args.num_envs))
    # envs = gym.vector.SyncVectorEnv(envs_list)
    # envs.eval = False

    # Set seeds
    envs.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("Starting reference experiment with zero control action")
    for step in range(0, args.num_steps):
        episode_reward = 0
        # ALGO LOGIC: action logic
        action = torch.zeros((envs.num_control_terms,))
        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, terminations, reward, truncations, infos = envs.step(action.cpu().numpy())
        next_obs, reward, terminations, info = envs.step(action.cpu().numpy())
        print(reward)
        print(terminations)
        next_done = terminations  # np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor([terminations], dtype=torch.float32).to(device)
        step += 1
        episode_reward += reward
        print('n_step:',envs.epoch_step)
        if next_done:
            envs.reset(seed= args.seed)
            print("Episode is done!!!")
            if args.log:
                wandb.log({'train/training_reward': episode_reward})
                episode_reward = 0

    print("Starting the real training of the ppo agent")
    for iteration in range(1, args.num_iterations + 1):
        print("Iteration", iteration)
        
        # Resetting the environment
        next_obs, _ = envs.reset(seed= args.seed)

        # Adjust next_obs
        next_obs = torch.Tensor(next_obs).to(device)
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
        print("Adjusted next_obs shape:", next_obs.shape)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        episode_reward = 0
        #starting the epoch
        for step in range(0, args.num_steps):
            print("Step", step)
            global_step += args.num_envs
            print("Global Step", global_step)
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                print("Action", action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            action_scaled = squeeze_scale_action(action,envs)
            
            
            print("Action passed to env:", action_scaled)
            print("Action passed to env shape:", action_scaled.shape)

            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, terminations, reward, truncations, infos = envs.step(action.cpu().numpy())
            next_obs, reward, terminations, info = envs.step(action_scaled)
            print(reward)
            #print(terminations)
            print("Raw next_obs shape:", np.array(next_obs).shape)
            print("Raw reward:", reward)
            print("Raw terminations:", terminations)

            print("Raw next_obs shape:", np.array(next_obs).shape)

            # Adjust next_obs
            next_obs = torch.Tensor(next_obs).to(device)
            if next_obs.dim() == 1:
                next_obs = next_obs.unsqueeze(0)
            print("Adjusted next_obs shape:", next_obs.shape)

            # Adjust next_done
            next_done = torch.tensor([terminations], dtype=torch.float32).to(device)

            # Adjust rewards
            rewards[step] = torch.tensor([reward], dtype=torch.float32).to(device).view(-1)

            episode_reward += reward
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            if next_done:
                print("Episode is done!!!")
                print("Reward with zero control", episode_reward)
                if args.log:
                    wandb.log({'train/training_reward': episode_reward,
                              })
                    episode_reward = 0
                    episode_drag = 0
                    episode_lift = 0
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.log:
            wandb.log({'losses/value_loss': v_loss.item(),
                       'losses/policy_loss': pg_loss.item(),
                       'losses/entropy': entropy_loss.item(),
                       'losses/old_approx_kl': old_approx_kl.item(),
                       'losses/approx_kl': approx_kl.item(),
                       'losses/clipfrac': np.mean(clipfracs),
                       'losses/explained_variance': explained_var,
                       'charts/learning_rate': optimizer.param_groups[0]["lr"]})

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        #from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )

        print('eval_reward',episodic_returns.mean())


        if args.log:
            wandb.log({'eval/eval_reward': episodic_returns.mean()})

        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub
        #
        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    # writer.close()