import numpy as np
import maze as mz
import matplotlib.pyplot as plt

'''--------Part A--------'''

# Description of the maze as a numpy array
# with the convention 
# 0 = empty cell
# 1 = bank
maze = np.array([
    [1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
])
mz.draw_maze(maze)


# Create an environment maze
env = mz.Maze(maze)
# env.show()

'''--------Part B--------'''

start  = (0,0,1,2)
# Discount Factors
GAMMA = np.arange(0.1, 1, 0.01)
# Accuracy treshold 
epsilon = 0.0001
value = []
for gamma in GAMMA:
    V, policy = mz.value_iteration(env, gamma, epsilon)
    value.append(V[env.map[start]])

plt.plot(GAMMA, value)
plt.ylabel("Value")
plt.xlabel("λ")
plt.show()


# Discount Factor 
gamma   = 0.99 #0.5
# Accuracy treshold 
epsilon = 0.0001
V, policy = mz.value_iteration(env, gamma, epsilon)
method = 'ValIter'
start  = (0, 0, 1, 2)
path = env.simulate(start, policy, method)
# Show the shortest path 
#mz.animate_solution(maze, path, start)