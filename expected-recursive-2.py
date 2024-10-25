from functools import partial
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
        # hprime = new_position(h, np.array([1, 0]), n) # helper moves towards (1, 0) while starter is moving towards (1, 0)
        delay = distance(h, np.array([1, 0])) # extra time taken for helper to reach (1, 0)
        return 1 + delay, [h] # return the expected delivery time and the path
    
    Fi = np.linspace(i/n, 1, n - i + 1) # potential fail
    min_value = float('inf')
    points = []
    for f in Fi: # minimize over which point to move towards
        t1 = 1 + distance(h, np.array([i/n, 0])) # failure occurs at beginning of this round
        h_prime = new_position(h, np.array([f, 0]), n) # helper moves towards (f, 0)
        t2, sub_points = _get_solution(h_prime, i + 1, n)
        
        current_value = (1/len(Fi)) * t1 + (1 - 1/len(Fi)) * t2
        
        if current_value < min_value:
            min_value = current_value
            points = [h] + sub_points
    
    return min_value, points

def get_solution(h: np.ndarray, n: int) -> Tuple[float, np.ndarray]:
    val, points = _get_solution(h, 0, n)
    return val, points

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
        if np.linalg.norm(pursuit_point - pursuer_pos) < dt_purser:
            pursuer_pos = pursuit_point
        else:
            pursuer_pos += dt_purser * (pursuit_point - pursuer_pos) / np.linalg.norm(pursuit_point - pursuer_pos)
        start_pos += np.array([dt_pursued, 0.0])
        path.append(np.copy(pursuer_pos))

    expected_delivery = np.mean([
        1 + distance(point, np.array([i / num_points, 0.0]))
        for i, point in enumerate(path)
    ])
    return expected_delivery, np.array(path)

def tangent_x_intercepts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Compute the slopes using central differences
    slopes = np.zeros_like(y)
    slopes[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    slopes[0] = (y[1] - y[0]) / (x[1] - x[0])  # Forward difference for the first point
    slopes[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])  # Backward difference for the last point

    # Compute the x-intercepts of the tangent lines
    x_intercepts = x - y / slopes

    # Handle cases where the slope is zero (to avoid division by zero)
    x_intercepts[slopes == 0] = np.nan  # No x-intercept for horizontal tangents

    return x_intercepts

def solve_minimization(helper_start: np.ndarray, n=100):
    """
    Solves the optimization problem numerically.

    Args:
        helper_start (np.ndarray): Initial position of the helper.
        N (int): Number of discretization points.

    Returns:
        Tuple[float, np.ndarray]: The expected delivery time and the path of the helper.
    """
    x0, y0 = helper_start
    t = np.linspace(0, 1, n + 1) # fail points
    dt = t[1] - t[0] # time step

    # Initial guess for x and y (straight line to destination (1, 0))
    x_init = np.linspace(x0, x0 + 1, n + 1)
    y_init = np.linspace(y0, y0, n + 1)

    z_init = np.concatenate([x_init, y_init]) # Flatten x and y for optimization
    def objective(z):
        x = z[:n + 1]
        y = z[n + 1:]
        expected_delivery = np.mean([
            1 + distance(np.array([x[i], y[i]]), np.array([t[i], 0.0]))
            for i in range(n + 1)
        ])
        return expected_delivery

    # Constraints
    constraints = []
    def speed_constraint(z):
        x = z[:n + 1]
        y = z[n + 1:]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        return ds - dt  # Should be zero

    # Initial conditions
    initial_x = lambda z: z[0] - x0
    initial_y = lambda z: z[n + 1] - y0

    # Add constraints
    for i in range(n):
        constraints.append({'type': 'eq', 'fun': lambda z, i=i: speed_constraint(z)[i]})
    constraints.append({'type': 'eq', 'fun': initial_x})
    constraints.append({'type': 'eq', 'fun': initial_y})

    # Bounds (optional, can be adjusted)
    bounds = [(None, None)] * len(z_init)

    # Optimization
    result = minimize(objective, z_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 1000})

    if result.success:
        z_opt = result.x
        x_opt = z_opt[:n + 1]
        y_opt = z_opt[n + 1:]
        expected_delivery = objective(z_opt)
        tangent_intercepts = tangent_x_intercepts(x_opt, y_opt)
        print(f"Diff: {np.diff(tangent_intercepts)}")
        print(f"Double-Diff: {np.diff(np.diff(tangent_intercepts))*1000}")
        ratios = []
        for i, xintercept in enumerate(tangent_intercepts):
            ratio = (xintercept - i / n) / (1 - i / n)
            print(f"{xintercept:0.2f}, {ratio:0.2f}")
            ratios.append(ratio)

        print(f"Ratio Diff: {np.diff(ratios)}")

        path = np.array([x_opt, y_opt]).T
        return expected_delivery, path
    else:
        raise ValueError("Optimization failed: " + result.message)

# The main function with interactivity
def interactive_plot(n: int = 8):
    agents = {
        'opt': partial(get_solution, n=n),
        # 'pursuit-half': partial(get_pursuit_points, num_points=n, ratio=0.5),
        'pursuit-0.53': partial(get_pursuit_points, num_points=n*5, ratio=0.53),
        'opt-2': partial(solve_minimization, n=n*5)
    }
    colors = ['b', 'r', 'g', 'y']

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
    xlim = [0, 1]
    ylim = [0, 1]
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
    min_value, points = get_solution(initial_position, 7)

    all_points = [initial_position] + points

    # Convert points to format for plotting
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]

    # Plot the points
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5)
    plt.scatter([0, 1], [0, 0]) # Highlight the extra points

    # plot pursuit points
    _, pursuit_points = get_pursuit_points(initial_position, 9)
    plt.plot(pursuit_points[:, 0], pursuit_points[:, 1], 'ro-', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Recursive Positions and Extra Points')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig('expected-recursive-2.png')

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
    interactive_plot(n=7)
    # plot_expected_delivery_times(num_points=10)


if __name__ == '__main__':
    main()
