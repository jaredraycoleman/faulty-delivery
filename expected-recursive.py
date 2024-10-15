from typing import List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

# A helper function to convert numpy arrays to tuples for caching
def array_to_tuple(array: np.ndarray) -> Tuple:
    return tuple(array.tolist())

@lru_cache(maxsize=None)
def get_solution(start: float,
                 helper_pos: Tuple[float],
                 fail_points: Tuple[Tuple[float, float], ...]) -> Tuple[List[np.ndarray], float]:
    """The starter is at position (start, 0) and the helper is at helper_pos.
    
    Args:
        start: The x-coordinate of the starter.
        helper_pos: The position of the helper as a tuple.
        fail_points: The x-coordinates of the potential failure points.
    Returns:
        A tuple containing the path of the helper and the expected delivery time.
    """
    prob = fail_points[0] - start
    dt = fail_points[0] - start
    if len(fail_points) == 1:
        # move towards the end
        new_pos = np.array(helper_pos) + dt * (np.array([1.0,0.0]) - np.array(helper_pos)) / np.linalg.norm(np.array([1.0,0.0]) - np.array(helper_pos))

        return [new_pos], 1 + np.linalg.norm(new_pos - np.array([1.0, 0.0]))


    min_delivery_time = float('inf')
    min_delivery_path = None
    for fail_point in fail_points:
        fail_point_array = np.array([fail_point, 0.0])
        new_pos = np.array(helper_pos) + dt * (fail_point_array - np.array(helper_pos)) / np.linalg.norm(fail_point_array - np.array(helper_pos))
        # If failure occurs in this round
        delivery_time_case_1 = 1 + np.linalg.norm(new_pos - fail_points[0])

        # If failure occurs in a later round
        points, delivery_time_case_2 = get_solution(fail_points[0], array_to_tuple(new_pos), fail_points[1:])

        # Expected delivery time
        delivery_time = prob * delivery_time_case_1 + (1 - prob) * delivery_time_case_2

        if delivery_time < min_delivery_time:
            min_delivery_time = delivery_time
            min_delivery_path = [np.array(new_pos)] + points

    return min_delivery_path, min_delivery_time

def get_pursuit_point(helper_start: np.ndarray, num_points: int) -> np.ndarray:
    start_pos = np.array([0.0, 0.0])
    pursuer_pos = np.copy(helper_start)
    dt = 1 / (num_points - 1)
    path: List[np.ndarray] = [np.copy(pursuer_pos)]
    for _ in range(num_points - 1):
        pursuit_point = np.array([start_pos[0] + dt, 0.0])
        # print(pursuit_point)
        pursuer_pos += dt * (pursuit_point - pursuer_pos) / np.linalg.norm(pursuit_point - pursuer_pos)
        start_pos += np.array([dt, 0.0])
        path.append(np.copy(pursuer_pos))

    return np.array(path)

def main():
    NPOINTS = 3
    helper_start = np.array([1.0, 1.0])
    fail_points = np.array([x for x in np.linspace(0, 1, NPOINTS)])[1:]
    print(fail_points)
    path, delivery_time = get_solution(0, array_to_tuple(helper_start), array_to_tuple(fail_points))
    path = np.array([helper_start] + path)

    pursuit_path = get_pursuit_point(helper_start, NPOINTS)


    # Plot Line
    plt.plot([0, 1], [0, 0], 'k--')

    # Plot the paths
    plt.plot(path[:, 0], path[:, 1], 'ro-', alpha=0.5)
    plt.plot(pursuit_path[:, 0], pursuit_path[:, 1], 'bo-', alpha=0.5)

    # Legend
    plt.legend(['Line', 'Helper', 'Pursuit Path'])

    # Save the plot
    plt.savefig('expected-recursive.png')

    # Print the results
    print(f"Expected delivery time: {delivery_time:0.4f}")
    nice_path = " -> ".join([f"({x:0.2f}, {y:0.2f})" for x, y in path])
    print(f"Path: {nice_path}")

if __name__ == '__main__':
    main()
