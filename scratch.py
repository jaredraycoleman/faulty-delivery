import random

from matplotlib import pyplot as plt
import numpy as np

def evenly_sample_points_in_triangle(x, y, num_points):
    points = []
    num_steps = int(np.ceil(np.sqrt(num_points * 2)))  # Calculate number of steps to generate more than num_points

    # Generate all possible points
    for i in range(num_steps + 1):
        for j in range(num_steps - i + 1):
            u1 = i / num_steps
            u2 = j / num_steps
            
            px = u1 + u2 * x
            py = u2 * y
            
            points.append((px, py))
    
    return points

# Generate exactly 100 evenly spaced points for x = 1/4, y = 1/2
evenly_sampled_points_100 = evenly_sample_points_in_triangle(1/4, 1/2, num_points=1000)

# Convert to array for plotting
evenly_sampled_points_100 = np.array(evenly_sampled_points_100)

# Plot the triangle
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 0], 'k-')
plt.plot([0, 1/4], [0, 1/2], 'k-')
plt.plot([1/4, 1], [1/2, 0], 'k-')
plt.scatter(evenly_sampled_points_100[:, 0], evenly_sampled_points_100[:, 1], c='r', s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Evenly Sampled Points in Triangle')
plt.show()