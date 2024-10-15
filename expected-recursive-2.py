import numpy as np
import matplotlib.pyplot as plt

def new_position(h: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    return h + (1/n) * (target - h) / np.linalg.norm(target - h)

def distance(h: np.ndarray, target: np.ndarray) -> float:
    return np.linalg.norm(target - h)

def get_solution(h: np.ndarray, i: int, n: int):
    if i == n:
        hprime = new_position(h, np.array([1, 0]), n)
        delay = distance(hprime, np.array([1, 0]))
        return 1 + delay, []
    Fi = np.linspace(i/n, 1, n - i + 1)
    
    min_value = float('inf')
    points = []
    
    # Minimize over f in F_i
    for f in Fi:
        h_prime = new_position(h, np.array([f, 0]), n)
        t1 = 1 + distance(h_prime, np.array([i/n, 0]))
        t2, sub_points = get_solution(h_prime, i + 1, n)
        
        current_value = (1/len(Fi)) * t1 + (1 - 1/len(Fi)) * t2
        
        if current_value < min_value:
            min_value = current_value
            points = [h_prime] + sub_points
    
    return min_value, points

def main():
    # Call the function
    initial_position = np.array([1, 1])
    min_value, points = get_solution(initial_position, 1, 8)

    all_points = [initial_position] + points

    # Convert points to format for plotting
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]

    # Plot the points
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=5)
    plt.scatter([0, 1], [0, 0]) # Highlight the extra points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Recursive Positions and Extra Points')
    plt.grid(True)
    plt.legend()
    plt.savefig('expected-recursive-2.png')

    # Print the results
    print(f"Expected delivery time: {min_value:0.4f}")
    path_str = ' -> '.join([f'({p[0]:0.2f}, {p[1]:0.2f})' for p in all_points])
    print(f"Path: {path_str}")

if __name__ == '__main__':
    main()
