import numpy as np
import matplotlib.pyplot as plt
import maze as mz
'''--------Part A--------'''


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

'''--------Part B--------'''

# Create an environment maze
MinotaurStay = False
env = mz.Maze(maze, MinotaurStay)
#env.show()

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming 
V, policy= mz.dynamic_programming(env,horizon)

start  = (0,0,6,5)
value = np.flip(V[env.map[start]])
plt.figure()
plt.plot(value)
plt.ylabel("Value")
plt.xlabel("T")
plt.show()

# Simulate the shortest path starting from position A
method = 'DynProg'
path = env.simulate(start, policy, method)
print(path)

# Show the shortest path 
#mz.animate_solution(maze, path, start)

'''--------Part C--------'''

# Create an environment maze
MinotaurStay = False
env = mz.Maze(maze, MinotaurStay)
#env.show()

# Discount Factor 
gamma   = 29/30
# Accuracy treshold 
epsilon = 0.0001
V, policy = mz.value_iteration(env, gamma, epsilon)

num_simulations = 10000
method = 'ValIter'
start  = (0,0,6,5)
exit = 0
for i in range(num_simulations):
    path = env.simulate(start, policy, method)
    if[path[-1][0:2] == start[2:]]:
        exit +=1
print("Probability of exiting the maze =", exit/num_simulations)