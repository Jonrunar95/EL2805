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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

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

class Critic(nn.Module):
	def __init__(self, input_size, output_size=1):
		super(Critic, self).__init__()
		self.layer1 = nn.Linear(input_size, 400)
		self.layer2 = nn.Linear(400, 200)
		self.layer3 = nn.Linear(200, output_size)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		return x


class Actor(nn.Module):
	def __init__(self, input_size, output_size):
		super(Actor, self).__init__()
		self.layer_input = nn.Linear(input_size, 400)
		self.layer_mu1 = nn.Linear(400, 200)
		self.layer_mu2 = nn.Linear(200, output_size)
		self.layer_var1 = nn.Linear(400, 200)
		self.layer_var2 = nn.Linear(200, output_size)

	def forward(self, x):
		x = F.relu(self.layer_input(x))

		mu = F.relu(self.layer_mu1(x))
		mu_out = torch.tanh(self.layer_mu2(mu))

		var = F.relu(self.layer_mu1(x))
		var_out = torch.sigmoid(self.layer_mu2(var))

		return mu_out, var_out



def critic_backward(loss):
	# Set gradients to 0
	optimizer_critic.zero_grad() 
	
	# Compute gradient
	loss.backward() 
	#make_dot(loss).render("attached", format="png") # Vizualize the backpropagation

	# Clip gradients
	#nn.utils.clip_grad_norm_(critic_nn.parameters(), 1.0)

	# Perform backward pass
	optimizer_critic.step() 

def actor_backward(loss):
	# Set gradients to 0
	optimizer_actor.zero_grad() 
	
	# Compute gradient
	loss.backward() 
	#make_dot(loss).render("attached", format="png") # Vizualize the backpropagation

	# Clip gradients 
	#nn.utils.clip_grad_norm_(actor_nn.parameters(), 1.0)

	# Perform backward pass
	optimizer_actor.step() 
	

class Buffer():
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.nextActions = []
		self.actionProb = []

	def append(self, state, action, reward, nextAction, actionProb):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.nextActions.append(nextAction)
		self.actionProb.append(actionProb)
	
	def getBufferAt(self, i):
		if i > len(states) - 1:
			print("ERROR: Trying to fetch buffer entry at", i, "when buffer size:", len(states))
		return self.states[i], self.actions[i], self.rewards[i], self.nextActions[i], self.actionProb[i]

	def getLength(self):
		return len(self.states)



# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 1600              # Number of episodes to run for training
discount_factor = 0.99         # Value of gamma
n_ep_running_average = 50      # Running average of 20 episodes
m = len(env.action_space.high) # dimensionality of the action
M = 10                         # No. epochs
actor_lr = 10**(-5)            # Adam learning rate for actor
critic_lr = 10**(-3)		   # Adam learning rate for critic
epsilon = 0.2				   # Obj. func. clipping hyperparameter

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
agent = RandomAgent(m)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# Neural nets
dim_state = len(env.observation_space.high) # State dimensionality
n_actions = len(env.action_space.high) #env.action_space.high # Or action_space.n?

actor_nn = Actor(dim_state, n_actions)
critic_nn = Critic(dim_state)

optimizer_actor = optim.Adam(actor_nn.parameters(), lr = actor_lr)
optimizer_critic = optim.Adam(critic_nn.parameters(), lr = critic_lr)
epsilon = torch.tensor(epsilon)

for i in EPISODES:
	# Reset enviroment data
	done = False
	state = env.reset()
	total_episode_reward = 0.
	t = 0
	buffer = Buffer()

	while not done:
		# Take a random action
		#action = agent.forward(state)

		# Get state tensor
		state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)

		# Get mu and var from actor pi_theta
		mu, var = actor_nn.forward(state_tensor)

		# Get distribution
		var = torch.diag_embed(var)
		dist = MultivariateNormal(mu, var)

		# Sample action from multivariate normal
		action = dist.sample()
		action_prob = dist.log_prob(action).detach() # Detach this...
		actionNP = action.numpy()[0] # Convert action to numpy

		# Get next state and reward.  The done variable
		# will be True if you reached the goal position,
		# False otherwise
		next_state, reward, done, _ = env.step(actionNP)

		# Append z = (s_t, a_t, r_t, s_t+1, d_t) to the buffer
		buffer.append(state, action, reward, next_state, action_prob)

		# Update episode reward
		total_episode_reward += reward

		# Update state for next iteration
		state = next_state
		t+= 1

	# Compute target value G_i for each (s_i, a_i) in buffer, DOUBLE CHECK THIS!
	G = []
	for i in range(buffer.getLength()):
		y_sum = 0 
		for k in range(i, t):
			y_sum += discount_factor**(k-i) * buffer.rewards[k] 
		G.append(y_sum)
	G = torch.tensor(G, requires_grad=False, dtype=torch.float32).unsqueeze(dim=-1) # Think about this

	# Create tensor for all states in buffer
	states = torch.tensor(buffer.states, requires_grad=False, dtype=torch.float32)

	# Get value estimates from critic V_omega for all buffer states
	V_omega = critic_nn.forward(states)

	# Advantage estimation psi_i
	psi = G - V_omega
	psi = psi.detach().squeeze(dim=-1) # Squeeze yes/no?... probably doesnt matter

	# For each epoch
	for k in range(M):
		# Re-estimate V_omega
		V_omega = critic_nn.forward(states)

		# Calculate MSE loss
		critic_loss = F.mse_loss(V_omega, G) 

		# Backwards on critic
		critic_backward(critic_loss)

		# Re-estimate pi_theta
		actor_mus, actor_vars = actor_nn.forward(states)
		actor_vars = torch.diag_embed(actor_vars)

		# Calculate actor loss, DOUBLECHECK THIS!
		actor_loss = torch.zeros(1)
		for i in range(buffer.getLength()):
			actorDist = MultivariateNormal(actor_mus[i], actor_vars[i]) # Get actor distribution for mu_i & var_i
			action_prob = actorDist.log_prob(buffer.actions[i]) # Get prob. of the action for new actor distribution
			r_theta = torch.exp(action_prob - buffer.actionProb[i]) # pi_theta / pi_theta_old
			c_epsilon = torch.max(1 - epsilon, torch.min(r_theta, 1 + epsilon)) # clipping function
			actor_loss += torch.min(r_theta * psi[i], c_epsilon * psi[i])
		actor_loss = (-1 / torch.tensor(buffer.getLength())) * actor_loss

		# Backwards on actor
		actor_backward(actor_loss)


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


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
	episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
	episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
