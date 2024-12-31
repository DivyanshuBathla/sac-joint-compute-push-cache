import argparse
import datetime
import numpy as np
import itertools
import torch
from sac.td3 import TD3
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory
from envs.MultiTaskCore import MultiTaskCore
from tool.data_loader import load_data
from config import system_config

# Argument parsing
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="MultiTaskCore",
                    help='Wireless Comm environment (default: MultiTaskCore)')
parser.add_argument('--exp-case', default="case3",
                    help='The experiment configuration case (default: case 3)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='Target smoothing coefficient (τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='Learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                    help='Maximum number of steps (default: 5000001)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='Hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                    help='Value target update per number of updates per step (default: 1000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='Size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='Run on CUDA (default: False)')
parser.add_argument('--policy_noise', type=float, default=0.2, 
                    help="Noise added to target policy during critic update")
parser.add_argument('--noise_clip', type=float, default=0.5, 
                    help="Range to clip target policy noise")
parser.add_argument('--policy_delay', type=int, default=2, 
                    help="Number of updates before policy network is updated")

args = parser.parse_args()

# Environment Initialization
task_num = system_config['F']
maxp = system_config['maxp']
task_utils = load_data('./data/task' + str(task_num) + '_utils.csv')
task_set_ = task_utils.tolist()
At = np.squeeze(load_data('data/samples' + str(task_num) + '_maxp' + str(maxp) + '.csv'))
channel_snrs = load_data('./data/dynamic_snrs.csv')

env = MultiTaskCore(init_sys_state=[0] * (2 * task_num) + [1], task_set=task_set_, requests=At,
                    channel_snrs=channel_snrs, exp_case=args.exp_case)

# Seed Initialization
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent Initialization
agent = TD3(env.observation_space.shape[0],  # state_dim
            env.action_space.shape[0],       # action_dim
            env.action_space.high[0],        # max_action
            args) 

# Tensorboard Writer Initialization
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                  args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Replay Memory Initialization
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
result_trans = []
result_comp = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        # Sample Action
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        # Update Agent Parameters
        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                # Update parameters of all the networks
                try:
                    critic_1_loss, critic_2_loss, policy_loss, additional_loss = agent.update_parameters(memory, args.batch_size)

                    # Log losses
                    if critic_1_loss is not None:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    else:
                        print("Critic 1 loss is None.")

                    if critic_2_loss is not None:
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    else:
                        print("Critic 2 loss is None.")

                    # Log policy loss only if it's not None
                    if policy_loss is not None:
                        writer.add_scalar('loss/policy', policy_loss, updates)
                    else:
                        print("Policy loss is None. Check policy update.")

                    updates += 1

                except Exception as e:
                    print(f"Error during parameter update: {e}")

        # Environment Step
        next_state, reward, done, info = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, mask)  # Append transition to memory
        state = next_state

    if total_numsteps > args.num_steps:
        break

    # Logging
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    # Evaluation
    eval_freq = 10
    if i_episode % eval_freq == 0 and args.eval is True:
        avg_reward = 0.
        avg_trans_cost = 0.
        avg_compute_cost = 0.
        episodes = 10
        done_step = 0
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            trans_cost = 0
            compute_cost = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                try:
                    trans_cost += info['observe_detail'][0]
                    compute_cost += info['observe_detail'][1]
                    done_step += 1
                except KeyError:
                    pass  # Handle missing keys gracefully
                state = next_state

            avg_reward += episode_reward
            avg_trans_cost += trans_cost
            avg_compute_cost += compute_cost

        avg_reward /= episodes
        avg_trans_cost /= done_step if done_step > 0 else 1  # Avoid division by zero
        avg_compute_cost /= done_step if done_step > 0 else 1  # Avoid division by zero
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Total Steps: {}, Avg. Reward: {}, Avg. Trans Cost: {}, Avg. Compute Cost: {}, Glb. Step: {}".format(
            episodes, int(done_step), round(avg_reward, 2), round(avg_trans_cost, 2), round(avg_compute_cost, 2), env.global_step))
        print("----------------------------------------")
        result_trans.append(avg_trans_cost)
        result_comp.append(avg_compute_cost)
        
        # Calculate average costs over last 10 episodes
        if len(result_trans) > 10:
            print_avg_trans = np.average(np.asarray(result_trans[-10:]))
            print_avg_comp = np.average(np.asarray(result_comp[-10:]))
        else:
            print_avg_trans = np.average(np.asarray(result_trans))
            print_avg_comp = np.average(np.asarray(result_comp))

        print("Final Avg Results for last 100 episodes: Avg. Trans Cost: {}, Avg. Compute Cost: {}".format(
            round(print_avg_trans, 2), round(print_avg_comp, 2)))
        print("----------------------------------------")
        writer.add_scalar('avg_cost/trans_cost', round(print_avg_trans, 2), i_episode)
        writer.add_scalar('avg_cost/comp_cost', round(print_avg_comp, 2), i_episode)

# Save Models
torch.save(agent.state_dict(), f"models/{args.env_name}_{args.policy}.pt")
print("Models saved successfully!")
writer.close()
