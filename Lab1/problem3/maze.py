import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

	# Actions
	STAY       = 0
	MOVE_LEFT  = 1
	MOVE_RIGHT = 2
	MOVE_UP    = 3
	MOVE_DOWN  = 4

	# Give names to actions
	actions_names = {
		STAY: "stay",
		MOVE_LEFT: "move left",
		MOVE_RIGHT: "move right",
		MOVE_UP: "move up",
		MOVE_DOWN: "move down"
	}

	# Reward values
	ROB_BANK = 1
	CAUGHT_REWARD = -10

	def __init__(self, maze, start_state = (0,0,3,3), weights=None, random_rewards=False):
		""" Constructor of the environment Maze.
		"""
		self.maze                     = maze
		self.start_state 			  = start_state
		self.actions                  = self.__actions()
		self.states, self.map         = self.__states()
		self.n_actions                = len(self.actions)
		self.n_states                 = len(self.states)
		self.transition_probabilities = self.__transitions()
		self.rewards                  = self.__rewards(weights=weights, random_rewards=random_rewards)

	def __actions(self):
		actions = dict()
		actions[self.STAY]       = (0, 0) # 0
		actions[self.MOVE_LEFT]  = (0,-1) # 1
		actions[self.MOVE_RIGHT] = (0, 1) # 2
		actions[self.MOVE_UP]    = (-1,0) # 3
		actions[self.MOVE_DOWN]  = (1,0)  # 4
		return actions

	def __states(self):
		states = dict()
		map = dict()
		end = False
		s = 0
		for i1 in range(self.maze.shape[0]):
			for j1 in range(self.maze.shape[1]):
				for i2 in range(self.maze.shape[0]):
					for j2 in range(self.maze.shape[1]):
						states[s] = (i1, j1, i2, j2)
						map[(i1, j1, i2, j2)] = s
						s += 1
		return states, map

	def __transitions(self):
		""" Computes the transition probabilities for every state action pair.
			:return numpy.tensor transition probabilities: tensor of transition
			probabilities of dimension S*S*A
		"""
		# Initialize the transition probailities tensor (S,S,A)
		dimensions = (self.n_states,self.n_states,self.n_actions)
		transition_probabilities = np.zeros(dimensions)

		# Compute the transition probabilities.
		for s in range(self.n_states):
			police_moves = self.__police_moves(s)
			for a_p in range(self.n_actions):
				for a_m, prob in police_moves.items():
					if self.states[s][0:2] == self.states[s][2:]:
						transition_probabilities[self.map[self.start_state], s, a_p] = 1
					else:
						next_s = self.__move(s, a_p, a_m)
						transition_probabilities[next_s, s, a_p] = prob
		return transition_probabilities

	def __move(self, state, action_player, action_police):
		""" Makes a step in the maze, given a current position and an action.
			:return tuple next_cell: Position (i_1, j_1, i_2, j_2) on the maze that agent transitions to.
		"""
		# Compute the future position given current (state, action)
		row_player = self.states[state][0] + self.actions[action_player][0]
		col_player = self.states[state][1] + self.actions[action_player][1]
		if (row_player == -1) or (row_player == self.maze.shape[0]) or (col_player == -1) or (col_player == self.maze.shape[1]):
			row_player = self.states[state][0]
			col_player = self.states[state][1]

		row_police = self.states[state][2] + self.actions[action_police][0]
		col_police = self.states[state][3] + self.actions[action_police][1]
		if (row_police == -1) or (row_police == self.maze.shape[0]) or (col_police == -1) or (col_police == self.maze.shape[1]):
			print("Illegal police move:", self.states[state], self.actions_names[action_police])
			print("Resulting state:", row_player, col_player, row_police, col_police)
			print(self.maze.shape)
			row_police = self.states[state][2]
			col_police = self.states[state][3]
		return self.map[(row_player, col_player, row_police, col_police)]

	def __rewards(self, weights=None, random_rewards=None):
		rewards = np.zeros((self.n_states, self.n_actions))
		for s in range(self.n_states):
			police_moves = self.__police_moves(s)
			for a_p in range(self.n_actions):
				for a_m, prob in police_moves.items():
					next_s = self.__move(s, a_p, a_m)
					# # Impossible reward
					# if self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and a_p != self.STAY:
					# 	rewards[s, a_p] += self.IMPOSSIBLE_REWARD*prob
					# Rewrd for being caught
					if self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
						rewards[s, a_p] += self.CAUGHT_REWARD*prob
					# Reward for robbing a bank
					elif self.maze[self.states[next_s][0:2]] == 1:
						rewards[s, a_p] += self.ROB_BANK*prob
		return rewards

	def __police_moves(self, state, getAction=False):
		# Compute the future position given current (state)
		police_pos = self.states[state][2:]
		police_moves = dict()

		# Number of directions possible
		possibleDir = 0
		# Possible action list
		actionList = []

		# Check where police can move
		if police_pos[0] > 0:
			police_moves[self.MOVE_UP] = 1
			possibleDir += 1
			actionList.append(3)
		if police_pos[0] < self.maze.shape[0]-1:
			possibleDir += 1
			police_moves[self.MOVE_DOWN] = 1
			actionList.append(4)
		if police_pos[1] > 0:
			possibleDir += 1
			police_moves[self.MOVE_LEFT] = 1
			actionList.append(1)
		if police_pos[1] < self.maze.shape[1]-1:
			police_moves[self.MOVE_RIGHT] = 1
			possibleDir += 1
			actionList.append(2)

		if(getAction):
			return actionList
		
		# Set uniform probabilities
		for key in police_moves:
			police_moves[key] /= possibleDir
			
		return police_moves

	def move_state(self, state, action, policeAction):
		#make police move...
		return self.__move(state, action, policeAction)

	def getPoliceAction(self, state):
		return self.__police_moves(state, True)

# Q learning algorithm
def Q_learning(env, iters=None, start=None):
	n_states  = env.n_states
	n_actions = env.n_actions

	if start == None:
		start = env.start_state
	if iters == None:
		iters = 10000000

	# Required variables
	Q = np.zeros((n_states, n_actions)) # Q states
	n = np.ones((n_states, n_actions)) # No. of updates in Q, for step size

	reward = env.rewards
	
	gamma = 0.8 # Discount factor
	#epsilon = 0.2 # Exploration probability

	state = env.map[start]
	maxQ = []

	for i in range(iters):
		action = np.random.choice(list(env.actions.keys()))
		alpha = 1/(n[state, action]**(2/3)) # Step size
		#alpha = 0.8
		policeAction = np.random.choice(env.getPoliceAction(state))
		new_state = env.move_state(state, action, policeAction)
		Q[state, action] = Q[state, action] + alpha * (reward[state, action] + gamma * np.max(Q[new_state, :]) - Q[state, action])
		n[state, action] += 1
		state = new_state
		maxQ.append(np.max(Q))
	return Q, maxQ

# def Reward(env, state):
# 	player_pos = env.states[state][0:2]
# 	police_pos = env.states[state][2:]
	
# 	# Reward for being caught
# 	elif player_pos[0] == police_pos[0] and player_pos[1] == police_pos[1]:
# 		return env.CAUGHT_REWARD
# 	# Reward for robbing a bank
# 	elif env.maze[player_pos] == 1:
# 		return env.ROB_BANK
# 	return 0


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: LIGHT_PURPLE, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path, start):

	# Map a color to each cell in the maze
	col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: LIGHT_PURPLE, -6: LIGHT_RED, -1: LIGHT_RED}

	# Size of the maze
	rows,cols = maze.shape

	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))

	# Remove the axis ticks and add title title
	ax = plt.gca()
	ax.set_title('Policy simulation')
	ax.set_xticks([])
	ax.set_yticks([])

	# Give a color to each cell
	colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

	# Create figure of the size of the maze
	fig = plt.figure(1, figsize=(cols,rows))

	# Create a table to color
	grid = plt.table(cellText=None,
						cellColours=colored_maze,
						cellLoc='center',
						loc=(0,0),
						edges='closed')

	# Modify the hight and width of the cells in the table
	tc = grid.properties()['children']
	for cell in tc:
		cell.set_height(1.0/rows)
		cell.set_width(1.0/cols)

	# Update the color at each frame
	for i in range(len(path)):
		grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[maze[path[i-1][0:2]]])
		grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
		grid.get_celld()[(path[i-1][2:])].set_facecolor(col_map[maze[path[i-1][2:]]])
		grid.get_celld()[(path[i-1][2:])].get_text().set_text('')

		grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
		grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')
		grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_RED)
		grid.get_celld()[(path[i][2:])].get_text().set_text('Police')
		if i > 0:
			if path[i][0:2] == path[i][2:]:
				grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_RED)
				grid.get_celld()[(path[i][0:2])].get_text().set_text('Player is Caught')
				return
		display.display(fig)
		display.clear_output(wait=True)
		time.sleep(1)
