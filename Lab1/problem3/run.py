import numpy as np
import maze as mz

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
# with the convention 
# 0 = empty cell
# 1 = bank
mz.draw_maze(maze)

# Create an environment maze
env = mz.Maze(maze)
'''
# env.show()
# Finite horizon
horizon = 10
# Solve the MDP problem with dynamic programming 
V, policy= mz.dynamic_programming(env,horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start  = (0, 0, 2, 3)
path = env.simulate(start, policy, method)

mz.animate_solution(maze, path, start)

# Discount Factor 
gamma   = 0.95
# Accuracy treshold 
epsilon = 0.0001
V, policy = mz.value_iteration(env, gamma, epsilon)

method = 'ValIter'
start  = (0,0,1,2)
path = env.simulate(start, policy, method)
'''