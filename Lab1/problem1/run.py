import numpy as np
import maze as mz

# Description of the maze as a numpy array
maze = np.array([
    [2, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 3, 0, 0],
])
# with the convention 
# 0 = empty cell
# 1 = obstacle
# 2 = beginning cell of the Maze
# 3 = end of the Maze, beginnig cell of the Minotaur

mz.draw_maze(maze)

# Create an environment maze
env = mz.Maze(maze)
#env.show()

# Finite horizon
horizon = 20
start  = (0, 0, 6, 5)

# Solve the MDP problem with dynamic programming 
V, policy= mz.dynamic_programming(env,horizon)
print(V[env.map[start]])
# Simulate the shortest path starting from position A
method = 'DynProg'

path = env.simulate(start, policy, method)
print(path)
'''
mz.animate_solution(maze, path)

# Discount Factor 
gamma   = 0.95
# Accuracy treshold 
epsilon = 0.0001
V, policy = mz.value_iteration(env, gamma, epsilon)

method = 'ValIter'
start  = (0,0,6,5)
path = env.simulate(start, policy, method)
'''