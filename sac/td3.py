# Method 1

# import os
# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from sac.utils import soft_update, hard_update
# from sac.model import DeterministicPolicy, QNetwork

# class TD3(object):
#     def __init__(self, num_inputs, action_space, args):
#         self.gamma = args.gamma
#         self.tau = args.tau
#         self.policy_delay = args.policy_delay
#         self.device = torch.device("cuda" if args.cuda else "cpu")
#         self.action_space_size = action_space.shape[0]
#         self.alpha=args.alpha
#         # Initialize the Q-Networks and Policy
#         self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

#         # Target Networks
#         self.critic_target_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.critic_target_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         hard_update(self.critic_target_1, self.critic_1)
#         hard_update(self.critic_target_2, self.critic_2)

#         # Optimizers
#         self.critic_optim_1 = Adam(self.critic_1.parameters(), lr=args.lr)
#         self.critic_optim_2 = Adam(self.critic_2.parameters(), lr=args.lr)
#         self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

#         self.total_it = 0

#     def select_action(self, state, evaluate=False):
#         state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
#         action = self.policy(state)
#         if not evaluate:
#             action = action + torch.normal(0, 0.1 * (action.max() - action.min()))  # Add noise for exploration
#         return action.detach().cpu().numpy()[0]

#     def update_parameters(self, memory, batch_size, updates):
#     # Sample a batch from memory
#         state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

#         state_batch = torch.FloatTensor(state_batch).to(self.device)
#         next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
#         action_batch = torch.FloatTensor(action_batch).to(self.device)
#         reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
#         done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

#         # Update Critic Networks
#         with torch.no_grad():
#             # Target Policy Smoothing
#             mean = torch.tensor(0.0, dtype=torch.float32).to(self.device)
#             std = 0.2  # Standard deviation for noise
#             noise = torch.clamp(torch.normal(mean, std, size=action_batch.size()), -0.5, 0.5)  # Add noise
#             next_action = self.policy(next_state_batch) + noise
#             next_action = next_action.clamp(-1, 1)

#             # Get target Q-values
#             target_q1 = self.critic_target_1(next_state_batch, next_action)
#             target_q2 = self.critic_target_2(next_state_batch, next_action)

#             # Extract Q-values if they are tuples
#             target_q1 = target_q1[0] if isinstance(target_q1, tuple) else target_q1
#             target_q2 = target_q2[0] if isinstance(target_q2, tuple) else target_q2

#             # Compute target Q-value
#             target_q = reward_batch + (1 - done_batch) * self.gamma * torch.min(target_q1, target_q2)

#         # Get current Q-values
#         q1 = self.critic_1(state_batch, action_batch)
#         q2 = self.critic_2(state_batch, action_batch)

#         # Extract Q-values if they are tuples
#         q1 = q1[0] if isinstance(q1, tuple) else q1
#         q2 = q2[0] if isinstance(q2, tuple) else q2

#         # Compute losses
#         critic_loss_1 = F.mse_loss(q1, target_q)
#         critic_loss_2 = F.mse_loss(q2, target_q)

#         # Optimize critics
#         self.critic_optim_1.zero_grad()
#         critic_loss_1.backward()
#         self.critic_optim_1.step()

#         self.critic_optim_2.zero_grad()
#         critic_loss_2.backward()
#         self.critic_optim_2.step()

#         # Initialize policy_loss
#         policy_loss = torch.tensor(0.0).to(self.device)

#         # Update Policy if required
#         if self.total_it % self.policy_delay == 0:
#             policy_output = self.policy(state_batch)
            
#             # Get Q-values from both critics for the policy output
#             q1_policy = self.critic_1(state_batch, policy_output)
#             q2_policy = self.critic_2(state_batch, policy_output)
            
#             # Extract Q-values if they are tuples
#             q1_policy = q1_policy[0] if isinstance(q1_policy, tuple) else q1_policy
#             q2_policy = q2_policy[0] if isinstance(q2_policy, tuple) else q2_policy

#             # Calculate clipped double Q-learning loss
#             min_q = torch.min(q1_policy, q2_policy)
#             # Calculate the policy loss with clipping
#             policy_loss = -(min_q - self.alpha * torch.log(policy_output)).mean()  # Assuming policy_output is a probability distribution

#             # Optimize policy
#             self.policy_optim.zero_grad()
#             policy_loss.backward()
#             self.policy_optim.step()

#             # Soft update of target networks
#             soft_update(self.critic_target_1, self.critic_1, self.tau)
#             soft_update(self.critic_target_2, self.critic_2, self.tau)

#         self.total_it += 1
#         return critic_loss_1.item(), critic_loss_2.item(), policy_loss.item(), 0, 0  # Adjust entropy and alpha as needed





#     # Save model parameters
    # def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
    #     if not os.path.exists('checkpoints/'):
    #         os.makedirs('checkpoints/')
    #     if ckpt_path is None:
    #         ckpt_path = "checkpoints/td3_checkpoint_{}_{}".format(env_name, suffix)
    #     print('Saving models to {}'.format(ckpt_path))
    #     torch.save({'policy_state_dict': self.policy.state_dict(),
    #                  'critic_1_state_dict': self.critic_1.state_dict(),
    #                  'critic_2_state_dict': self.critic_2.state_dict(),
    #                  'critic_target_1_state_dict': self.critic_target_1.state_dict(),
    #                  'critic_target_2_state_dict': self.critic_target_2.state_dict(),
    #                  'critic_optim_1_state_dict': self.critic_optim_1.state_dict(),
    #                  'critic_optim_2_state_dict': self.critic_optim_2.state_dict(),
    #                  'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # # Load model parameters
    # def load_checkpoint(self, ckpt_path, evaluate=False):
    #     print('Loading models from {}'.format(ckpt_path))
    #     if ckpt_path is not None:
    #         checkpoint = torch.load(ckpt_path)
    #         self.policy.load_state_dict(checkpoint['policy_state_dict'])
    #         self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    #         self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    #         self.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
    #         self.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
    #         self.critic_optim_1.load_state_dict(checkpoint['critic_optim_1_state_dict'])
    #         self.critic_optim_2.load_state_dict(checkpoint['critic_optim_2_state_dict'])
    #         self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    #         if evaluate:
    #             self.policy.eval()
    #             self.critic_1.eval()
    #             self.critic_2.eval()
    #             self.critic_target_1.eval()
    #             self.critic_target_2.eval()
    #         else:
    #             self.policy.train()
    #             self.critic_1.train()
    #             self.critic_2.train()
    #             self.critic_target_1.train()
    #             self.critic_target_2.train()



#Method 2

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from sac.utils import soft_update, hard_update
# from sac.model import DeterministicPolicy, QNetwork

# class OUNoise:
#     """Ornstein-Uhlenbeck process for action noise."""
#     def __init__(self, action_dim, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2):
#         self.mu = mu * torch.ones(action_dim)
#         self.theta = theta
#         self.sigma = sigma
#         self.dt = dt
#         self.action_dim = action_dim
#         self.reset()

#     def reset(self):
#         self.state = self.mu.clone()

#     def sample(self):
#         x = self.state
#         dx = (self.theta * (self.mu - x) * self.dt + 
#               self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(self.action_dim))  # Ensure dt is a Tensor
#         self.state = x + dx
#         return self.state



# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from sac.utils import soft_update, hard_update
# from sac.model import DeterministicPolicy, QNetwork

# class OUNoise:
#     """Ornstein-Uhlenbeck process for action noise."""
#     def __init__(self, action_dim, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2):
#         self.mu = mu * torch.ones(action_dim)
#         self.theta = theta
#         self.sigma = sigma
#         self.dt = dt
#         self.action_dim = action_dim
#         self.reset()

#     def reset(self):
#         self.state = self.mu.clone()

#     def sample(self):
#         x = self.state
#         dx = (self.theta * (self.mu - x) * self.dt + 
#               self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(self.action_dim))  # Ensure dt is a Tensor
#         self.state = x + dx
#         return self.state


# class TD3(object):
#     def __init__(self, num_inputs, action_space, args):
#         self.gamma = args.gamma
#         self.tau = args.tau
#         self.policy_delay = args.policy_delay
#         self.device = torch.device("cuda" if args.cuda else "cpu")
#         self.action_space_size = action_space.shape[0]
#         self.alpha = args.alpha
#         self.noise = OUNoise(self.action_space_size)  # Initialize Ornstein-Uhlenbeck noise

#         # Initialize the Q-Networks and Policy
#         self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

#         # Target Networks
#         self.critic_target_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         self.critic_target_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
#         hard_update(self.critic_target_1, self.critic_1)
#         hard_update(self.critic_target_2, self.critic_2)

#         # Optimizers
#         self.critic_optim_1 = Adam(self.critic_1.parameters(), lr=args.lr)
#         self.critic_optim_2 = Adam(self.critic_2.parameters(), lr=args.lr)
#         self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

#         self.total_it = 0

#     def select_action(self, state, evaluate=False):
#         state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
#         action = self.policy(state)
#         if not evaluate:
#             action += torch.tensor(self.noise.sample(), dtype=torch.float32).to(self.device)  # Add noise for exploration
#         return action.detach().cpu().numpy()[0]

#     def update_parameters(self, memory, batch_size, updates):
#         # Sample a batch from memory
#         state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

#         state_batch = torch.FloatTensor(state_batch).to(self.device)
#         next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
#         action_batch = torch.FloatTensor(action_batch).to(self.device)
#         reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
#         done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

#         # Update Critic Networks
#         with torch.no_grad():
#             # Target Policy Smoothing
#             noise = torch.clamp(torch.normal(0.0, 0.2, size=action_batch.size()), -0.5, 0.5)  # Add noise
#             next_action = self.policy(next_state_batch) + noise
#             next_action = next_action.clamp(-1, 1)

#             # Get target Q-values
#             target_q1 = self.critic_target_1(next_state_batch, next_action)
#             target_q2 = self.critic_target_2(next_state_batch, next_action)

#             target_q1 = self.get_q_values(target_q1)
#             target_q2 = self.get_q_values(target_q2)

#             # Compute target Q-value
#             target_q = reward_batch + (1 - done_batch) * self.gamma * torch.min(target_q1, target_q2)

#         # Get current Q-values
#         q1 = self.get_q_values(self.critic_1(state_batch, action_batch))
#         q2 = self.get_q_values(self.critic_2(state_batch, action_batch))

#         # Compute losses
#         critic_loss_1 = F.mse_loss(q1, target_q)
#         critic_loss_2 = F.mse_loss(q2, target_q)

#         # Optimize critics
#         self.critic_optim_1.zero_grad()
#         critic_loss_1.backward()
#         self.critic_optim_1.step()

#         self.critic_optim_2.zero_grad()
#         critic_loss_2.backward()
#         self.critic_optim_2.step()

#         # Update Policy if required
#         policy_loss = torch.tensor(0.0).to(self.device)
#         if self.total_it % self.policy_delay == 0:
#             policy_output = self.policy(state_batch)

#             # Get Q-values from both critics for the policy output
#             q1_policy = self.get_q_values(self.critic_1(state_batch, policy_output))
#             q2_policy = self.get_q_values(self.critic_2(state_batch, policy_output))

#             # Calculate clipped double Q-learning loss
#             min_q = torch.min(q1_policy, q2_policy)
#             # Calculate the policy loss with clipping
#             policy_loss = -(min_q - self.alpha * torch.log(policy_output)).mean()  # Adjust as needed for your output distribution

#             # Optimize policy
#             self.policy_optim.zero_grad()
#             policy_loss.backward()
#             self.policy_optim.step()

#             # Soft update of target networks
#             soft_update(self.critic_target_1, self.critic_1, self.tau)
#             soft_update(self.critic_target_2, self.critic_2, self.tau)

#         self.total_it += 1
#         return critic_loss_1.item(), critic_loss_2.item(), policy_loss.item(),0,0

#     def get_q_values(self, q_value):
#         """Extract Q-values from tuples if necessary."""
#         return q_value[0] if isinstance(q_value, tuple) else q_value

#     # Save model parameters
#     def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
#         if not os.path.exists('checkpoints/'):
#             os.makedirs('checkpoints/')
#         if ckpt_path is None:
#             ckpt_path = f"checkpoints/td3_checkpoint_{env_name}_{suffix}"
#         print(f'Saving models to {ckpt_path}')
#         torch.save({
#             'policy_state_dict': self.policy.state_dict(),
#             'critic_1_state_dict': self.critic_1.state_dict(),
#             'critic_2_state_dict': self.critic_2.state_dict(),
#             'critic_target_1_state_dict': self.critic_target_1.state_dict(),
#             'critic_target_2_state_dict': self.critic_target_2.state_dict(),
#             'critic_optim_1_state_dict': self.critic_optim_1.state_dict(),
#             'critic_optim_2_state_dict': self.critic_optim_2.state_dict(),
#             'policy_optimizer_state_dict': self.policy_optim.state_dict()
#         }, ckpt_path)

#     # Load model parameters
#     def load_checkpoint(self, ckpt_path, evaluate=False):
#         print(f'Loading models from {ckpt_path}')
#         if ckpt_path is not None:
#             checkpoint = torch.load(ckpt_path)
#             self.policy.load_state_dict(checkpoint['policy_state_dict'])
#             self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
#             self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
#             self.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
#             self.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
#             self.critic_optim_1.load_state_dict(checkpoint['critic_optim_1_state_dict'])
#             self.critic_optim_2.load_state_dict(checkpoint['critic_optim_2_state_dict'])
#             self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
#         if evaluate:
#             self.policy.eval()

# Method 3


import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.utils import soft_update, hard_update
from sac.model import DeterministicPolicy, QNetwork
from sac.replay_memory import HERReplayMemory  # Import the HER memory

class TD3(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_delay = args.policy_delay
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.action_space_size = action_space.shape[0]
        self.alpha = args.alpha
        self.memory = HERReplayMemory(args.replay_size, args.seed)  # Initialize HER memory
        # Initialize the Q-Networks and Policy
        self.critic_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

        # Target Networks
        self.critic_target_1 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target_2 = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target_1, self.critic_1)
        hard_update(self.critic_target_2, self.critic_2)

        # Optimizers
        self.critic_optim_1 = Adam(self.critic_1.parameters(), lr=args.lr)
        self.critic_optim_2 = Adam(self.critic_2.parameters(), lr=args.lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.policy(state)
        if not evaluate:
            action = action + torch.normal(0, 0.1 * (action.max() - action.min()))  # Add noise for exploration
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample HER transitions from memory
        her_transitions = self.memory.sample_her_transitions(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch = zip(*her_transitions)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Update Critic Networks
        with torch.no_grad():
            # Target Policy Smoothing
            mean = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            std = 0.2  # Standard deviation for noise
            noise = torch.clamp(torch.normal(mean, std, size=action_batch.size()), -0.5, 0.5)  # Add noise
            next_action = self.policy(next_state_batch) + noise
            next_action = next_action.clamp(-1, 1)

            # Get target Q-values
            target_q1 = self.critic_target_1(next_state_batch, next_action)
            target_q2 = self.critic_target_2(next_state_batch, next_action)

            target_q1 = target_q1[0] if isinstance(target_q1, tuple) else target_q1
            target_q2 = target_q2[0] if isinstance(target_q2, tuple) else target_q2

            # Compute target Q-value
            target_q = reward_batch + (1 - done_batch) * self.gamma * torch.min(target_q1, target_q2)

        # Get current Q-values
        q1 = self.critic_1(state_batch, action_batch)
        q2 = self.critic_2(state_batch, action_batch)

        q1 = q1[0] if isinstance(q1, tuple) else q1
        q2 = q2[0] if isinstance(q2, tuple) else q2

        # Compute losses
        critic_loss_1 = F.mse_loss(q1, target_q)
        critic_loss_2 = F.mse_loss(q2, target_q)

        # Optimize critics
        self.critic_optim_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optim_1.step()

        self.critic_optim_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optim_2.step()

        # Initialize policy_loss
        policy_loss = torch.tensor(0.0).to(self.device)

        # Update Policy if required
        if self.total_it % self.policy_delay == 0:
            policy_output = self.policy(state_batch)

            # Get Q-values from both critics for the policy output
            q1_policy = self.critic_1(state_batch, policy_output)
            q2_policy = self.critic_2(state_batch, policy_output)

            q1_policy = q1_policy[0] if isinstance(q1_policy, tuple) else q1_policy
            q2_policy = q2_policy[0] if isinstance(q2_policy, tuple) else q2_policy

            # Calculate clipped double Q-learning loss
            min_q = torch.min(q1_policy, q2_policy)
            policy_loss = -(min_q - self.alpha * torch.log(policy_output)).mean()

            # Optimize policy
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Soft update of target networks
            soft_update(self.critic_target_1, self.critic_1, self.tau)
            soft_update(self.critic_target_2, self.critic_2, self.tau)

        self.total_it += 1
        return critic_loss_1.item(), critic_loss_2.item(), policy_loss.item(), 0, 0  # Adjust entropy and alpha as needed


    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/td3_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                     'critic_1_state_dict': self.critic_1.state_dict(),
                     'critic_2_state_dict': self.critic_2.state_dict(),
                     'critic_target_1_state_dict': self.critic_target_1.state_dict(),
                     'critic_target_2_state_dict': self.critic_target_2.state_dict(),
                     'critic_optim_1_state_dict': self.critic_optim_1.state_dict(),
                     'critic_optim_2_state_dict': self.critic_optim_2.state_dict(),
                     'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
            self.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
            self.critic_optim_1.load_state_dict(checkpoint['critic_optim_1_state_dict'])
            self.critic_optim_2.load_state_dict(checkpoint['critic_optim_2_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic_1.eval()
                self.critic_2.eval()
                self.critic_target_1.eval()
                self.critic_target_2.eval()
            else:
                self.policy.train()
                self.critic_1.train()
                self.critic_2.train()
                self.critic_target_1.train()
                self.critic_target_2.train()

