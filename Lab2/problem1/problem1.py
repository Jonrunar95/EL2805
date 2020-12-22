### IMPORT PACKAGES ###
import numpy as np
import gym
from collections import deque, namedtuple, OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from tqdm import trange
from DQN_agent import RandomAgent
import random
import copy
import torch.nn.functional as F


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

class DQN(nn.Module):
	def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
		super(DQN, self).__init__()
		self.lin1 = nn.Linear(input_size, neurons_per_layer)
		self.lin2 = nn.Linear(neurons_per_layer, neurons_per_layer)
		self.lin3 = nn.Linear(neurons_per_layer, neurons_per_layer)
		self.lin4 = nn.Linear(neurons_per_layer, output_size)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		return self.lin4(x)
	
	def getLayers(self):
		return [self.lin1.state_dict(), self.lin2.state_dict(), self.lin3.state_dict(), self.lin4.state_dict()]

	def setLayers(self, layers):
		self.lin1.load_state_dict(layers[0])
		self.lin2.load_state_dict(layers[1])
		self.lin3.load_state_dict(layers[2])
		self.lin4.load_state_dict(layers[3])

def running_average(x, N):
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

def train(BUFFER_LENGTH, LEARNING_RATE, HIDDEN_LAYERS, NEURON_PER_LAYER, BATCH_SIZE, DISCOUNT_FACTOR):
	env = gym.make('LunarLander-v2')
	env.reset()
	
	CLIPPING_VALUE = 1.0 # 0.5 - 2

	N_EPISODES = 500 # 100 - 1000
	Z = N_EPISODES * 0.9
	N_EP_RUNNING_AVERAGE = 50 # Running average of 50 episodes
	#MDP Hyperparameters
	EPSILON_MIN = 0.05
	EPSILON_MAX = 0.99
	N_ACTIONS = env.action_space.n # Number of available actions
	DIM_STATE = len(env.observation_space.high) # State dimensionality
	#Target Network Hyperparameters
	TARGET_NETWORK_FREQUENCY_UPDATE = BUFFER_LENGTH/BATCH_SIZE#1000

	buffer = ExperienceReplayBuffer(BUFFER_LENGTH)
	network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
	target_network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
	target_network.setLayers(network.getLayers())

	optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

	# We will use these variables to compute the average episodic reward and
	# the average number of steps per episode
	episode_reward_list = []       # this list contains the total reward per episode
	episode_number_of_steps = []   # this list contains the number of steps per episode

	# Random agent initialization 
	agent = RandomAgent(N_ACTIONS)

	### Training process

	steps = 0 # Total number of steps so far
	EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
	for i in EPISODES:
		epsilon = max(EPSILON_MIN, EPSILON_MAX - (EPSILON_MAX-EPSILON_MIN)*i/(Z-1))
		
		# Reset enviroment data and initialize variables
		done = False
		state = env.reset()
		total_episode_reward = 0.
		t = 0
		while not done:
			state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
			Q_sA = network.forward(state_tensor) # Calculate Q(s, A) for all awailable actions using the policy network
			if random.random() > epsilon: # Greedy
				action = Q_sA.max(1)[1].item()
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
				
				output_values = network.forward(states_tensor)[index_tensor, actions_tensor] # Compute Q(s, a) of the policy network given the states and actions
				
				target_q_values = target_network.forward(next_states_tensor).max(1)[0] # Compute max_a(Q(next_s, A)) using the target network
				
				target_values = rewards_tensor +  DISCOUNT_FACTOR * (1 - dones_tensor) * target_q_values # Calculate y_i
				
				loss = F.mse_loss(output_values, target_values.detach()) # Detach the target_network from the backpropagation
				
				optimizer.zero_grad() # Set gradients to 0
				
				loss.backward() # Compute gradient
				#make_dot(loss).render("attached", format="png") # Vizualize the backpropagation
				
				nn.utils.clip_grad_norm_(network.parameters(), CLIPPING_VALUE) # Clip gradient norm

				optimizer.step() # Perform backward pass (backpropagation)

			
			total_episode_reward += reward # Update episode reward

			
			state = next_state # Update state for next iteration
			t+= 1
			steps += 1
			if steps % TARGET_NETWORK_FREQUENCY_UPDATE == 0: # Update the Target Network periodically
				target_network.setLayers(network.getLayers())

				

		# Append episode reward and total number of steps
		episode_reward_list.append(total_episode_reward)
		episode_number_of_steps.append(t)

		'''
		if (i+1) % N_EPISODES == 0: # Save the neural_network
			name = 'neural-network-1.pt'
			torch.save(network, name)
		'''

		# Close environment
		env.close()

		# Updates the tqdm update bar with fresh information
		# (episode number, total reward of the last episode, total number of Steps
		# of the last episode, average reward, average number of steps)
		EPISODES.set_description(
			"Episode {} - Reward: {:.1f} - Avg. Reward: {:.1f} - Epsilon {:.2f} - Steps {} - TNFU {}".format(
			i, total_episode_reward, running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
			epsilon, steps, TARGET_NETWORK_FREQUENCY_UPDATE))


	
	plotEpisodes(episode_reward_list, episode_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE)

def trainManyEpisodes(BUFFER_LENGTH, LEARNING_RATE, HIDDEN_LAYERS, NEURON_PER_LAYER, BATCH_SIZE, DISCOUNT_FACTOR):
	episodes_reward_lists = []
	episodes_number_of_steps = []
	N_EPISODES_LIST = [300, 500, 700]#[100, 300, 500, 700, 900] # 100 - 1000
	for N_EPISODES in N_EPISODES_LIST:
		env = gym.make('LunarLander-v2')
		env.reset()
		
		CLIPPING_VALUE = 1.0 # 0.5 - 2

		Z = N_EPISODES * 0.9
		N_EP_RUNNING_AVERAGE = 50 # Running average of 50 episodes
		#MDP Hyperparameters
		EPSILON_MIN = 0.05
		EPSILON_MAX = 0.99
		N_ACTIONS = env.action_space.n # Number of available actions
		DIM_STATE = len(env.observation_space.high) # State dimensionality
		#Target Network Hyperparameters
		TARGET_NETWORK_FREQUENCY_UPDATE = BUFFER_LENGTH/BATCH_SIZE#1000

		buffer = ExperienceReplayBuffer(BUFFER_LENGTH)
		network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
		target_network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
		target_network.setLayers(network.getLayers())

		optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

		# We will use these variables to compute the average episodic reward and
		# the average number of steps per episode
		episode_reward_list = []       # this list contains the total reward per episode
		episode_number_of_steps = []   # this list contains the number of steps per episode
		# Random agent initialization 
		agent = RandomAgent(N_ACTIONS)

		### Training process

		
		steps = 0 # Total number of steps so far
		EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
		for i in EPISODES:
			epsilon = max(EPSILON_MIN, EPSILON_MAX - (EPSILON_MAX-EPSILON_MIN)*i/(Z-1))
			
			# Reset enviroment data and initialize variables
			done = False
			state = env.reset()
			total_episode_reward = 0.
			t = 0
			while not done:
				state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
				Q_sA = network.forward(state_tensor) # Calculate Q(s, A) for all awailable actions using the policy network
				if random.random() > epsilon: # Greedy
					action = Q_sA.max(1)[1].item()
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
					
					output_values = network.forward(states_tensor)[index_tensor, actions_tensor] # Compute Q(s, a) of the policy network given the states and actions
					
					target_q_values = target_network.forward(next_states_tensor).max(1)[0] # Compute max_a(Q(next_s, A)) using the target network
					
					target_values = rewards_tensor +  DISCOUNT_FACTOR * (1 - dones_tensor) * target_q_values # Calculate y_i
					
					loss = F.mse_loss(output_values, target_values.detach()) # Detach the target_network from the backpropagation
					
					optimizer.zero_grad() # Set gradients to 0
					
					loss.backward() # Compute gradient
					#make_dot(loss).render("attached", format="png") # Vizualize the backpropagation
					
					nn.utils.clip_grad_norm_(network.parameters(), CLIPPING_VALUE) # Clip gradient norm

					optimizer.step() # Perform backward pass (backpropagation)

				
				total_episode_reward += reward # Update episode reward

				
				state = next_state # Update state for next iteration
				t+= 1
				steps += 1
				if steps % TARGET_NETWORK_FREQUENCY_UPDATE == 0: # Update the Target Network periodically
					target_network.setLayers(network.getLayers())

					

			# Append episode reward and total number of steps
			episode_reward_list.append(total_episode_reward)
			episode_number_of_steps.append(t)

			# Close environment
			env.close()

			# Updates the tqdm update bar with fresh information
			# (episode number, total reward of the last episode, total number of Steps
			# of the last episode, average reward, average number of steps)
			EPISODES.set_description(
				"Episode {} - Reward: {:.1f} - Avg. Reward: {:.1f} - Epsilon {:.2f} - Steps {} - TNFU {}".format(
				i, total_episode_reward, running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
				epsilon, steps, TARGET_NETWORK_FREQUENCY_UPDATE))
		
		episodes_reward_lists.append(episode_reward_list)
		episodes_number_of_steps.append(episode_number_of_steps)
	print(episodes_reward_lists)
	print(episodes_number_of_steps)
	plotManyEpisodes(episodes_reward_lists, episodes_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE)

def plotManyEpisodes(episode_reward_list, episode_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE):
	# Plot Rewards and steps
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
	for ep in range(len(episode_reward_list)):
		#ax[0].plot([i for i in range(1, len(episode_reward_list[ep])+1)], episode_reward_list[ep], label='Episode reward')
		ax[0].plot([i for i in range(1, len(episode_reward_list[ep])+1)], running_average(
			episode_reward_list[ep], N_EP_RUNNING_AVERAGE), label='Avg. reward for ' + str(len(episode_reward_list[ep])) + ' episodes')
		ax[0].set_xlabel('Episodes')
		ax[0].set_ylabel('Total reward')
		ax[0].set_title('Total Reward vs Episodes')
		ax[0].legend()
		ax[0].grid(alpha=0.3)

		#ax[1].plot([i for i in range(1, len(episode_number_of_steps[ep])+1)], episode_number_of_steps[ep], label='Steps per episode')
		ax[1].plot([i for i in range(1, len(episode_number_of_steps[ep])+1)], running_average(
			episode_number_of_steps[ep], N_EP_RUNNING_AVERAGE), label='Avg. number of steps for ' + str(len(episode_reward_list[ep])) + ' episodes')
		ax[1].set_xlabel('Episodes')
		ax[1].set_ylabel('Total number of steps')
		ax[1].set_title('Total number of steps vs Episodes')
		ax[1].legend()
		ax[1].grid(alpha=0.3)

	plt.show()

def trainManyBufferSizes(BUFFER_LENGTH, LEARNING_RATE, HIDDEN_LAYERS, NEURON_PER_LAYER, BATCH_SIZE, DISCOUNT_FACTOR):
	episodes_reward_lists = []
	episodes_number_of_steps = []
	BUFFER_LENGTHS = [10000, 20000, 30000]
	for BUFFER_LENGTH in BUFFER_LENGTHS:
		env = gym.make('LunarLander-v2')
		env.reset()
		
		CLIPPING_VALUE = 1.0 # 0.5 - 2

		N_EP_RUNNING_AVERAGE = 50 # Running average of 50 episodes
		N_EPISODES = 500 # 100 - 1000
		#MDP Hyperparameters
		EPSILON_MIN = 0.05
		EPSILON_MAX = 0.99
		N_ACTIONS = env.action_space.n # Number of available actions
		DIM_STATE = len(env.observation_space.high) # State dimensionality
		#Target Network Hyperparameters
		TARGET_NETWORK_FREQUENCY_UPDATE = BUFFER_LENGTH/BATCH_SIZE#1000

		buffer = ExperienceReplayBuffer(BUFFER_LENGTH)
		network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
		target_network = DQN(DIM_STATE, N_ACTIONS, HIDDEN_LAYERS, NEURON_PER_LAYER)
		target_network.setLayers(network.getLayers())

		optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

		# We will use these variables to compute the average episodic reward and
		# the average number of steps per episode
		episode_reward_list = []       # this list contains the total reward per episode
		episode_number_of_steps = []   # this list contains the number of steps per episode
		# Random agent initialization 
		agent = RandomAgent(N_ACTIONS)

		### Training process

		
		steps = 0 # Total number of steps so far
		EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
		Z = N_EPISODES * 0.9
		for i in EPISODES:
			epsilon = max(EPSILON_MIN, EPSILON_MAX - (EPSILON_MAX-EPSILON_MIN)*i/(Z-1))
			
			# Reset enviroment data and initialize variables
			done = False
			state = env.reset()
			total_episode_reward = 0.
			t = 0
			while not done:
				state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
				Q_sA = network.forward(state_tensor) # Calculate Q(s, A) for all awailable actions using the policy network
				if random.random() > epsilon: # Greedy
					action = Q_sA.max(1)[1].item()
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
					
					output_values = network.forward(states_tensor)[index_tensor, actions_tensor] # Compute Q(s, a) of the policy network given the states and actions
					
					target_q_values = target_network.forward(next_states_tensor).max(1)[0] # Compute max_a(Q(next_s, A)) using the target network
					
					target_values = rewards_tensor +  DISCOUNT_FACTOR * (1 - dones_tensor) * target_q_values # Calculate y_i
					
					loss = F.mse_loss(output_values, target_values.detach()) # Detach the target_network from the backpropagation
					
					optimizer.zero_grad() # Set gradients to 0
					
					loss.backward() # Compute gradient
					#make_dot(loss).render("attached", format="png") # Vizualize the backpropagation
					
					nn.utils.clip_grad_norm_(network.parameters(), CLIPPING_VALUE) # Clip gradient norm

					optimizer.step() # Perform backward pass (backpropagation)

				
				total_episode_reward += reward # Update episode reward

				
				state = next_state # Update state for next iteration
				t+= 1
				steps += 1
				if steps % TARGET_NETWORK_FREQUENCY_UPDATE == 0: # Update the Target Network periodically
					target_network.setLayers(network.getLayers())

					

			# Append episode reward and total number of steps
			episode_reward_list.append(total_episode_reward)
			episode_number_of_steps.append(t)

			# Close environment
			env.close()

			# Updates the tqdm update bar with fresh information
			# (episode number, total reward of the last episode, total number of Steps
			# of the last episode, average reward, average number of steps)
			EPISODES.set_description(
				"Episode {} - Reward: {:.1f} - Avg. Reward: {:.1f} - Epsilon {:.2f} - Steps {} - TNFU {}".format(
				i, total_episode_reward, running_average(episode_reward_list, N_EP_RUNNING_AVERAGE)[-1],
				epsilon, steps, TARGET_NETWORK_FREQUENCY_UPDATE))
		
		episodes_reward_lists.append(episode_reward_list)
		episodes_number_of_steps.append(episode_number_of_steps)
	plotManyBufferSizes(episodes_reward_lists, episodes_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE, BUFFER_LENGTHS)

def plotManyBufferSizes(episode_reward_list, episode_number_of_steps, N_EPISODES, N_EP_RUNNING_AVERAGE, BUFFER_LENGTHS):
	# Plot Rewards and steps
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
	for ep in range(len(episode_reward_list)):
		#ax[0].plot([i for i in range(1, len(episode_reward_list[ep])+1)], episode_reward_list[ep], label='Episode reward')
		ax[0].plot([i for i in range(1, len(episode_reward_list[ep])+1)], running_average(
			episode_reward_list[ep], N_EP_RUNNING_AVERAGE), label='Buffer Size = ' + str(BUFFER_LENGTHS[ep]))
		ax[0].set_xlabel('Episodes')
		ax[0].set_ylabel('Total reward')
		ax[0].set_title('Total Reward vs Episodes')
		ax[0].legend()
		ax[0].grid(alpha=0.3)

		#ax[1].plot([i for i in range(1, len(episode_number_of_steps[ep])+1)], episode_number_of_steps[ep], label='Steps per episode')
		ax[1].plot([i for i in range(1, len(episode_number_of_steps[ep])+1)], running_average(
			episode_number_of_steps[ep], N_EP_RUNNING_AVERAGE), label='Buffer Size = ' + str(BUFFER_LENGTHS[ep]))
		ax[1].set_xlabel('Episodes')
		ax[1].set_ylabel('Total number of steps')
		ax[1].set_title('Total number of steps vs Episodes')
		ax[1].legend()
		ax[1].grid(alpha=0.3)

	plt.show()

def plot3DImage():
	try:
		model = torch.load('./Lab2/problem1/neural-network-1.pt')
		print('Network model: {}'.format(model))
	except:
		print('File neural-network-1.pth not found!')
		exit(-1)
	num = 100
	y = np.linspace(0, 1.5, num = num)
	w = np.linspace(-np.pi, np.pi, num = num)
	
	S = np.zeros((num*num, 8))
	for i in range(num):
		for j in range(num):
			S[i*num+j, 1] = y[i] 
			S[i*num+j, 4] = w[j]

	states_tensor = torch.tensor(S, requires_grad=False, dtype=torch.float32)
	Q_sa = model.forward(states_tensor).max(1)[0].detach().numpy()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(S[:, 1], S[:, 4], Q_sa, marker='o')

	ax.set_xlabel('Y')
	ax.set_ylabel('W')
	ax.set_zlabel('max(Q(S, W))')

	plt.show()

if __name__ == "__main__":
	# Import and initialize the discrete Lunar Laner Environment
	env = gym.make('LunarLander-v2')
	env.reset()
	BUFFER_LENGTH = 20000 # 5000 − 30000.
	LEARNING_RATE = 0.0005 # 10^-3 - 10^-4
	HIDDEN_LAYERS = 2 # Unused, always 2 hidden layers
	NEURON_PER_LAYER = 64 # 8 - 128
	BATCH_SIZE = 32 # 4 − 128
	DISCOUNT_FACTOR = 0.99 # Value of the discount factor

	#trainManyBufferSizes(BUFFER_LENGTH, LEARNING_RATE, HIDDEN_LAYERS, NEURON_PER_LAYER, BATCH_SIZE, DISCOUNT_FACTOR)

	plot3DImage()