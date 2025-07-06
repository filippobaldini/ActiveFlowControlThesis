import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space, h_dim=10):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, action_dim)

        # Directly use environment bounds here
        self.register_buffer("action_low", torch.FloatTensor(action_space.low))
        self.register_buffer("action_high", torch.FloatTensor(action_space.high))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))

        action = self.action_low + (
            0.5 * (x + 1.0) * (self.action_high - self.action_low)
        )
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim=10):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.l3 = nn.Linear(h_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, h_dim)
        self.l5 = nn.Linear(h_dim, h_dim)
        self.l6 = nn.Linear(h_dim, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        action_space,
        lr_A=3e-4,
        lr_C=3e-4,
        h_dim=512,
        tau=0.05,
        device="cpu",
    ):

        self.actor = Actor(state_dim, action_dim, action_space, h_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_space, h_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_A)

        self.critic = Critic(state_dim, action_dim, h_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, h_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_C)

        self.tau = tau
        self.device = device

        self.epoch_count = 0

    def select_action(self, state):
        # if isinstance(state, tuple):
        #     state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).detach().cpu().numpy().flatten()

    # regular TD3 is stateless; add this to conform to API
    def reset(self):
        pass

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        noise_clip_flag=False,
        log=True,
    ):
        self.epoch_count += 1
        for it in range(iterations):
            # self.epoch_count += 1

            # Sample replay buffer
            x, u, y, r, d = replay_buffer.sample(batch_size)
            state = x
            action = u
            next_state = y
            reward = r
            done = 1.0 - d

            if noise_clip_flag:
                noise = (
                    torch.FloatTensor(action.size())
                    .data.normal_(0, policy_noise)
                    .to(self.device)
                )
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (
                    self.actor_target(next_state) + noise
                )  # clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor_target(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()
            # import ipdb; ipdb.set_trace()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                # log training losses
                # if log:
                #     if self.epoch_count % 1 == 0:
                # wandb.log({'train/critic_loss': critic_loss,
                #            'train/actor_loss': actor_loss})

    def save(self, filename, directory):
        # Save best actor and critic separately
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

        # Also save full agent (for 'last' model inspection)
        torch.save(self, f"{directory}/{filename}_all.pth")

    def load(self, filename, directory):
        if not torch.cuda.is_available():
            self.actor.load_state_dict(
                torch.load(
                    "%s/%s_actor.pth" % (directory, filename), map_location="cpu"
                )
            )
            self.critic.load_state_dict(
                torch.load(
                    "%s/%s_critic.pth" % (directory, filename), map_location="cpu"
                )
            )
        else:
            self.actor.load_state_dict(
                torch.load("%s/%s_actor.pth" % (directory, filename))
            )
            self.critic.load_state_dict(
                torch.load("%s/%s_critic.pth" % (directory, filename))
            )

    def load_all(self, filename, directory):
        if not torch.cuda.is_available():
            return torch.load(
                "%s/%s_all.pth" % (directory, filename),
                map_location="cpu",
                weights_only=False,
            )
        else:
            return torch.load("%s/%s_all.pth" % (directory, filename))
