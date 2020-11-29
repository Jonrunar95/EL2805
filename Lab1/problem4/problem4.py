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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
epsilon = 0.1 # Exploration prob. 
alpha = 0.2 # learning rate
lamb = 0.8 # Eligibility trace variable

# Reward
episode_reward_list = []  # Used to save episodes reward

# Eta, user defined... no idea what this should be
# Should take values in p = {0, 1, 2}
eta = np.array([
    [1,0],
    [0,1],
    [2,1],
    [1,2],
])

class ValueFunction:
    def __init__(self, eta, alpha=alpha, lamb=lamb, gamma=discount_factor):
        self.alpha = alpha # Learning rate
        self.lamb = lamb # Eligibility trace variable
        self.gamma = gamma # Discount factor
        self.eta = eta # User defined for basis functions!
        self.n_basisFunctions = eta.shape[0] # Number of basis functions used
        self.W = np.zeros((k, self.n_basisFunctions)) # Weights for each available action
        self.Z = np.zeros(self.W.shape) # Eligibility trace
        
    # Update value for given state and action
    def update(self, state, action, new_state, new_action, reward):

        # Update eligibility trace, not sure if this should be done AFTER weight updates
        for i in range(self.Z.shape[0]):
            if i == action:
                # CHECK W IN FORMULA, PROBABLY NOT RIGHT!!!
                self.Z[i] = (self.gamma * self.lamb * self.Z[i]) + (self.W[i]*self.Q(state, action))
                # Clipping to avoid exploding gradients
                self.Z[i] = np.clip(self.Z[i], -5, 5)
            else:
                self.Z[i] = np.clip(self.gamma * self.lamb * self.Z[i], -5, 5)
        
        # Calculate error
        d_t = reward + (self.gamma * self.Q(new_state, new_action)) - self.Q(state, action)
        # Update weights
        self.W += self.alpha * d_t * self.Z
        

    # Reset between episodes
    def reset(self):
        self.Z = np.zeros(self.W.shape)

    # Get value for given state and action
    def Q(self, state, action):
        return np.dot(self.W[action].T, self.basisFunctions(state))

    # Get basis functions for state
    def basisFunctions(self, state):
        phi = np.zeros(self.n_basisFunctions)
        for i in range(len(phi)):
            phi[i] = np.cos(np.pi*np.dot(self.eta[i].T, state))
        return phi
        


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def pickAction(state, vf, epsilon):
    # Do random action:
    if np.random.rand() <= epsilon:
        return np.random.randint(0, k)

    # Do greedy action:
    values = []
    actions = []
    # Get value for each action
    for i in range(k):
        values.append(vf.Q(state, i))
    # Only pick best actions
    for num,v in enumerate(values):
        if v == np.max(values):
            actions.append(num)
    
    #print("actions are", actions)

    # Random choice between best values
    return np.random.choice(actions)


# Training process
vf = ValueFunction(eta) # Value Function
for i in range(N_episodes):
    # Reset enviroment data
    vf.reset() # Reset the eligibility trace!
    done = False
    total_episode_reward = 0.

    # Get intial state and action
    state = scale_state_variables(env.reset())
    action = pickAction(state, vf, epsilon)

    while not done:
        # Take an action
        # env.action_space.n tells you the number of actions
        # available
        #action = np.random.randint(0, k)
        
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        # Update episode reward
        total_episode_reward += reward

        # Update value function
        next_action = pickAction(next_state, vf, epsilon)
        vf.update(state, action, next_state, next_action, reward)
            
        # Update state for next iteration
        state = next_state
        action = next_action

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    print("Episode", i, "reward:", total_episode_reward)

    # Close environment
    env.close()
    

# Plot Rewards
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()