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
# env.show()

# Finite horizon
horizon = 10
# Solve the MDP problem with dynamic programming 
V, policy= mz.dynamic_programming(env,horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start  = (0, 0, 6, 5)
path = env.simulate(start, policy, method)

mz.animate_solution(maze, path)