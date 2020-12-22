# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent
from collections import deque, namedtuple, OrderedDict
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
import copy
import torch.nn.functional as F
from problem2.DDPG_soft_updates import soft_updates

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
	def __init__(self, maximum_length):
		self.buffer = deque(maxlen=maximum_length)

	def append(self, experience):
		self.buffer.append(experience)

	def __len__(self):
		return len(self.buffer)

	def sample_batch(self, n):
		if n > len(self.buffer):
			raise IndexError('Tried to sample too many elements from the buffer!')
		indices = np.random.choice(len(self.buffer), size=n, replace=False)
		batch = [self.buffer[i] for i in indices]
		states, actions, rewards, next_states, dones = zip(*batch)
		return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class Actor(nn.Module): # Fix action network. Append action to state
	def __init__(self, input_size, output_size):
		super(Actor, self).__init__()
		self.lin1 = nn.Linear(input_size, 400)
		self.lin2 = nn.Linear(400, 200)
		self.lin2 = nn.Linear(200, output_size)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		return F.tanh(self.lin3(x))
	
	def getLayers(self):
		return [self.lin1.state_dict(), self.lin2.state_dict(), self.lin3.state_dict()]

	def setLayers(self, layers):
		self.lin1.load_state_dict(layers[0])
		self.lin2.load_state_dict(layers[1])
		self.lin3.load_state_dict(layers[2])

class Critic(nn.Module):
	def __init__(self, input_size, output_size):
		super(Critic, self).__init__()
		self.lin1 = nn.Linear(input_size, 400)
		self.lin2 = nn.Linear(400, 200)
		self.lin2 = nn.Linear(200, output_size)

	def forward(self, s, a):
		x = F.relu(np.concatenate(self.lin1(s), a))
		x = F.relu(self.lin2(x))
		return F.tanh(self.lin3(x))
	
	def getLayers(self):
		return [self.lin1.state_dict(), self.lin2.state_dict(), self.lin3.state_dict()]

	def setLayers(self, layers):
		self.lin1.load_state_dict(layers[0])
		self.lin2.load_state_dict(layers[1])
		self.lin3.load_state_dict(layers[2])

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def plotEpisodes(episode_reward_list, episode_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE):
	# Plot Rewards and steps
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
	ax[0].plot([i for i in range(1, N_EPISODES+1)], episode_reward_list, label='Episode reward')
	ax[0].plot([i for i in range(1, N_EPISODES+1)], running_average(
		episode_reward_list, N_EP_RUNNING_AVERAGE), label='Avg. episode reward')
	ax[0].set_xlabel('Episodes')
	ax[0].set_ylabel('Total reward')
	ax[0].set_title('Total Reward vs Episodes')
	ax[0].legend()
	ax[0].grid(alpha=0.3)

	ax[1].plot([i for i in range(1, N_EPISODES+1)], episode_number_of_steps, label='Steps per episode')
	ax[1].plot([i for i in range(1, N_EPISODES+1)], running_average(
		episode_number_of_steps, N_EP_RUNNING_AVERAGE), label='Avg. number of steps per episode')
	ax[1].set_xlabel('Episodes')
	ax[1].set_ylabel('Total number of steps')
	ax[1].set_title('Total number of steps vs Episodes')
	ax[1].legend()
	ax[1].grid(alpha=0.3)
	plt.show()

if __name__ == "__main__":
	# Import and initialize Mountain Car Environment
	env = gym.make('LunarLanderContinuous-v2')
	env.reset()
	'''
	Discount factor γ;
	buffer size L;
	number of episodes TE;
	target networks update constant τ;
	policy update frequency d
	'''

	# Parameters

	BUFFER_LENGTH = 30000
	A_LEARNING_RATE = 0.00005
	C_LEARNING_RATE = 0.0005
	BATCH_SIZE = 64 # 4 − 128
	DISCOUNT_FACTOR = 0.99 # Value of the discount factor
	TARGET_NETWORK_UPDATE_CONSTANT = 0.001
	d = 2

	N_episodes = 300               # Number of episodes to run for training
	DIM_STATE = len(env.observation_space.high) # State dimensionality
	m = len(env.action_space.high) # dimensionality of the action
	CLIPPING_VALUE = 1.0 # 0.5 - 2

	Z = N_episodes * 0.9
	N_EP_RUNNING_AVERAGE = 50 # Running average of 50 episodes
	
	EPSILON_MIN = 0.05
	EPSILON_MAX = 0.99
	#Target Network Hyperparameters
	TARGET_NETWORK_FREQUENCY_UPDATE = BUFFER_LENGTH/BATCH_SIZE#1000

	buffer = ExperienceReplayBuffer(BUFFER_LENGTH)
	actor_network = Actor(DIM_STATE, m)
	actor_target_network = Actor(DIM_STATE, m)
	actor_target_network.setLayers(actor_network.getLayers())
	
	critic_network = Critic(DIM_STATE, m)
	critic_target_network = Critic(DIM_STATE, m)
	critic_target_network.setLayers(critic_network.getLayers())

	actor_optimizer = optim.Adam(actor_network.parameters(), lr=A_LEARNING_RATE)
	critic_optimizer = optim.Adam(critic_network.parameters(), lr=C_LEARNING_RATE)
	
	# Reward
	episode_reward_list = []  # Used to save episodes reward
	episode_number_of_steps = []

	# Agent initialization
	agent = RandomAgent(m)

	steps = 0
	# Training process
	EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
	n = 0
	for i in EPISODES:
		# Reset enviroment data
		epsilon = max(EPSILON_MIN, EPSILON_MAX - (EPSILON_MAX-EPSILON_MIN)*i/(Z-1))
		done = False
		state = env.reset()
		total_episode_reward = 0.
		t = 0
		while not done:
			state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
			if random.random() > epsilon: # Greedy
				action = agent.forward(state) #TODO
			else: # Explore
				action = agent.forward(state)
			next_state, reward, done, _ = env.step(action) # Take step in the environment using the action 
			buffer.append(Experience(state, action, reward, next_state, done)) # Add experience to memory

			if len(buffer) >= BATCH_SIZE: # Start training if there is enough data in the memory
				states, actions, rewards, next_states, dones = buffer.sample_batch(n=BATCH_SIZE) # Sample a batch
				
				# Convert to Tensors
				index_tensor = torch.tensor(np.arange(0, BATCH_SIZE, 1))
				states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
				actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
				rewards_tensor = torch.tensor(rewards, requires_grad=False, dtype=torch.float32)
				next_states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
				dones_tensor = torch.tensor(dones, requires_grad=False, dtype=torch.int64)
				
				# TODO
				# Compute target values
				y = rewards_tensor +  DISCOUNT_FACTOR * (1 - dones_tensor) * critic_target_network.forward(next_states_tensor)
				# TODO
				# update w using SGD


				if t % TARGET_NETWORK_FREQUENCY_UPDATE == 0:
					break
					# TODO
					# Update theta using SGD
					#
					# Soft update target networks


			# Update episode reward
			total_episode_reward += reward

			# Update state for next iteration
			state = next_state
			t+= 1

		n = 0.9 * 
		# Append episode reward
		episode_reward_list.append(total_episode_reward)
		episode_number_of_steps.append(t)
		# Close environment
		env.close()

		# Updates the tqdm update bar with fresh information
		# (episode number, total reward of the last episode, total number of Steps
		# of the last episode, average reward, average number of steps)
		EPISODES.set_description(
			"Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
			i, total_episode_reward, t,
			running_average(episode_reward_list, n_ep_running_average)[-1],
			running_average(episode_number_of_steps, n_ep_running_average)[-1]))

	plotEpisodes(episode_reward_list, episode_number_of_steps, N_episodes, N_EP_RUNNING_AVERAGE)