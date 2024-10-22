from functools import partial
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def new_position(h: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    """Returns the new position of the helper after moving towards the target.
    
    Args:
        h: The current position of the helper.
        target: The target position.
        n: The number of potential failure points.
    """
    if np.linalg.norm(target - h) <= 1/n:
        return target
    return h + (1/n) * (target - h) / np.linalg.norm(target - h)

def distance(h: np.ndarray, target: np.ndarray) -> float:
    """Returns the distance between the helper and the target.

    Args:
        h: The current position of the helper.
        target: The target position.
    """
    return np.linalg.norm(target - h)

def _get_solution(h: np.ndarray, i: int, n: int) -> Tuple[float, np.ndarray]:
    """Returns the expected delivery time and the path of the helper, 
    supposing the helper starts at position h and the start is going to
    fail at time i/n, (i+1)/n, ..., 1 with equal probability.

    Args:
        h: The current position of the helper.
        i: The current step.
        n: The number of potential failure points.

    Returns:
        The expected delivery time and the path of the helper
    """
    if i == n: # Base case - starter fails at 1
        hprime = new_position(h, np.array([1, 0]), n) # helper moves towards (1, 0) while starter is moving towards (1, 0)
        delay = distance(hprime, np.array([1, 0])) # extra time taken for helper to reach (1, 0)
        return 1 + delay, [hprime] # return the expected delivery time and the path
    
    Fi = np.linspace(i/n, 1, n - i + 1) # potential fail
    min_value = float('inf')
    points = []
    for f in Fi: # minimize over which point to move towards
        h_prime = new_position(h, np.array([f, 0]), n) # helper moves towards (f, 0)
        t1 = 1 + distance(h_prime, np.array([i/n, 0])) # failure occurs at end of this round
        t2, sub_points = _get_solution(h_prime, i + 1, n)
        
        current_value = (1/len(Fi)) * t1 + (1 - 1/len(Fi)) * t2
        
        if current_value < min_value:
            min_value = current_value
            points = [h_prime] + sub_points
    
    return min_value, points

def get_solution(h: np.ndarray, n: int) -> Tuple[float, np.ndarray]:
    val, points = _get_solution(h, 1, n)
    return val, [h] + points

def get_pursuit_points(helper_start: np.ndarray,
                       num_points: int,
                       ratio: float = 0.0) -> Tuple[float, np.ndarray]:
    start_pos = np.array([ratio, 0.0])
    pursuer_pos = np.copy(helper_start)
    dt_pursued = (1-ratio) / num_points
    dt_purser = 1 / num_points
    path: List[np.ndarray] = [np.copy(pursuer_pos)]
    for _ in range(num_points):
        pursuit_point = np.array([start_pos[0] + dt_pursued, 0.0])
        # print(pursuit_point)
        if np.linalg.norm(pursuit_point - pursuer_pos) < 1 / num_points:
            pursuer_pos = pursuit_point
        else:
            pursuer_pos += dt_purser * (pursuit_point - pursuer_pos) / np.linalg.norm(pursuit_point - pursuer_pos)
        start_pos += np.array([dt_pursued, 0.0])
        path.append(np.copy(pursuer_pos))

    expected_delivery = np.mean([
        1+np.linalg.norm(point - np.array([i / (num_points - 1), 0.0]))
        for i, point in enumerate(path[1:])
    ])
    return expected_delivery, np.array(path)

# The main function with interactivity
def interactive_plot(n: int = 8):
    agents = {
        'opt': partial(get_solution, n=n),
        'pursuit': partial(get_pursuit_points, num_points=n),
        'pursuit-half': partial(get_pursuit_points, num_points=n, ratio=0.5)
    }
    colors = ['b', 'r', 'g']

    def onclick(event):
        # Check if the click is within the plot bounds
        if event.xdata is not None and event.ydata is not None:
            # Get the clicked x and y coordinates
            initial_position = np.array([event.xdata, event.ydata])

            ax.cla()  # Use cla() to clear the plot without affecting axis limits
            solutions = {}
            for i, (name, agent) in enumerate(agents.items()):
                exp_del, all_points = agent(initial_position)
                solutions[name] = (exp_del, all_points)

                # Convert points to format for plotting
                x_values = [p[0] for p in all_points]
                y_values = [p[1] for p in all_points]

                # Plot the points
                ax.plot(
                    x_values, y_values, marker='o',
                    linestyle='-', color=colors[i],
                    markersize=5, alpha=0.5
                )
                
            ax.scatter([0, 1], [0, 0])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Recursive Positions and Extra Points')
            ax.grid(True)
            ax.legend(agents.keys())

            # Restore the fixed axis limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Update the text output
            del_times = " | ".join([
                f"{name}: {exp_del:0.4f}"
                for name, (exp_del, _) in solutions.items()
            ])
            min_val_text.set_text(f"Expected delivery times\n{del_times}")

            # Redraw the plot
            plt.draw()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)  # Adjust layout to make room for text
    ax.set_aspect('equal')  # Set aspect ratio to be equal

    # Create text areas for displaying results, positioning them in figure space
    min_val_text = fig.text(0.5, 0.15, "", ha="center", fontsize=10)  # Moved to figure space
    path_text = fig.text(0.5, 0.15, "", ha="center", fontsize=8)     # Moved to figure space

    # Set initial axis limits (you can adjust these as needed)
    xlim = [0, 2]
    ylim = [0, 2]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Initial plot setup
    ax.set_title("Click on the plot to select an initial position")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    # Connect the click event to the onclick function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the plot
    plt.show()

def noninteractive_plot():
    # Call the function
    initial_position = np.array([0.0, 1.0])
    min_value, points = get_solution(initial_position, 8)

    all_points = [initial_position] + points

    # Convert points to format for plotting
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]

    # Plot the points
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5)
    plt.scatter([0, 1], [0, 0]) # Highlight the extra points

    # plot pursuit points
    _, pursuit_points = get_pursuit_points(initial_position, 9)
    print(pursuit_points)
    plt.plot(pursuit_points[:, 0], pursuit_points[:, 1], 'ro-', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Recursive Positions and Extra Points')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig('expected-recursive-2.png')

    # Print the results
    print(f"Expected delivery time: {min_value:0.4f}")
    path_str = ' -> '.join([f'({p[0]:0.2f}, {p[1]:0.2f})' for p in all_points])
    print(f"Path: {path_str}")

    # get total length of path
    total_length = 0
    for i in range(len(all_points) - 1):
        total_length += np.linalg.norm(all_points[i + 1] - all_points[i])
    print(f"Total length of path: {total_length}")

# plot expected delivery times over evenly spaced helper starting points
# on the interval [0, 1] x [0, 1]
def plot_expected_delivery_times(n: int = 8, num_points=100):
    # sample helper starting points
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)

    # compute expected delivery times
    Z = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            print(f"{i*num_points + j + 1}/{num_points**2} ({(i*num_points + j + 1)/num_points**2 * 100:.2f}%)", end='\r')
            initial_position = np.array([X[i, j], Y[i, j]])
            min_value, _ = get_solution(initial_position, n)
            Z[i, j] = min_value

    print(" " * 100, end='\r')

    # plot expected delivery times
    plt.figure()
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Expected Delivery Times')
    plt.savefig('expected-delivery-times.png')

def main():
    # noninteractive_plot()
    interactive_plot(n=8)
    # plot_expected_delivery_times(num_points=10)


if __name__ == '__main__':
    main()
