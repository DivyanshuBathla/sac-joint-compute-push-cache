import argparse
import datetime
import numpy as np
import itertools
import torch
from sac.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory
from envs.MultiTaskCore import MultiTaskCore
from tool.data_loader import load_data
from config import system_config

# Parse arguments
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
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=5000001, metavar='N',
                    help='maximum number of steps (default: 5000001)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                    help='Value target update per no. of updates per step (default: 1000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--baseline', default="PTDFC",
                    help='Baseline algorithm: PTDFC | DFC | DFNC | MFU-LFU | MRU-LRU (default: PTDFC)')
args = parser.parse_args()

# Environment setup
task_num = system_config['F']
maxp = system_config['maxp']
task_utils = load_data(f'./data/task{task_num}_utils.csv')
task_set_ = task_utils.tolist()
At = np.squeeze(load_data(f'data/samples{task_num}_maxp{maxp}.csv'))
channel_snrs = load_data('./data/dynamic_snrs.csv')

if args.baseline in ["PTDFC", "DFC", "DFNC"]:
    env = MultiTaskCore(init_sys_state=[0] * (2 * task_num) + [1], task_set=task_set_, requests=At,
                        channel_snrs=channel_snrs, exp_case=args.exp_case)
elif args.baseline in ["MFU-LFU", "MRU-LRU"]:
    # Placeholder for MFU-LFU or MRU-LRU specific environment setup
    pass

env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent setup
if args.baseline in ["PTDFC", "DFC", "DFNC"]:
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
elif args.baseline in ["MFU-LFU", "MRU-LRU"]:
    # Placeholder for different agent initialization if needed
    pass

# Tensorboard setup
writer = SummaryWriter(
    f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.policy}_' +
    ('autotune' if args.automatic_entropy_tuning else ''))

# Memory setup
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
        if args.baseline in ["PTDFC", "DFC", "DFNC"]:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state

        elif args.baseline in ["MFU-LFU", "MRU-LRU"]:
            # Implement logic for MFU-LFU or MRU-LRU baselines
            pass

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

    if i_episode % 10 == 0 and args.eval:
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
                except:
                    pass
                state = next_state

            avg_reward += episode_reward
            avg_trans_cost += trans_cost
            avg_compute_cost += compute_cost

        avg_reward /= episodes
        avg_trans_cost /= done_step
        avg_compute_cost /= done_step
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Total Steps: {int(done_step)}, Avg. Reward: {round(avg_reward, 2)}, Avg. Trans Cost: {round(avg_trans_cost, 2)}, Avg. Compute Cost: {round(avg_compute_cost, 2)}, Glb. Step: {env.global_step}")
        print("----------------------------------------")
        result_trans.append(avg_trans_cost)
        result_comp.append(avg_compute_cost)
        if len(result_trans) > 10:
            print_avg_trans = np.average(np.asarray(result_trans[-10:]))
            print_avg_comp = np.average(np.asarray(result_comp[-10:]))
        else:
            print_avg_trans = np.average(np.asarray(result_trans))
            print_avg_comp = np.average(np.asarray(result_comp))
        print(f"Final Avg Results for last 10 episodes: Avg. Trans Cost: {round(print_avg_trans, 2)}, Avg. Compute Cost: {round(print_avg_comp, 2)}")
        print("----------------------------------------")
        writer.add_scalar('avg_cost/trans_cost', round(print_avg_trans, 2), i_episode)
        writer.add_scalar('avg_cost/comp_cost', round(print_avg_comp, 2), i_episode)

# Close the tensorboard writer
writer.close()
