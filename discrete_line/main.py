from functools import lru_cache
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def main():
    starting_pos = np.random.rand(2)
    points = np.sort(np.random.rand(10))
    probs = np.random.rand(10)
    probs /= np.sum(probs)

    starting_pos = np.array([0.5, 1])
    points = np.array([0, 1/2, 1])
    probs = np.array([0, 1/2, 1/2])


    @lru_cache(maxsize=None)
    def opt(lidx: int, ridx: int, on_left: bool) -> Tuple[float, List[str]]:
        cur_pos = points[lidx] if on_left else points[ridx]
        if lidx == 0 and ridx == len(points) - 1:
            return 0.0, []
        elif lidx == 0: # must move to ridx + 1
            prob_at_next = probs[ridx + 1] # / (1 - np.sum(probs[lidx:ridx+1]))
            sub_cost, sub_moves = opt(lidx, ridx + 1, False)
            dist = abs(cur_pos - points[ridx + 1])
            return prob_at_next * (dist + (1 - points[ridx + 1])) + (1 - prob_at_next) * (dist + sub_cost), ['r'] + sub_moves
        elif ridx == len(points) - 1: # must move to lidx - 1
            prob_at_next = probs[lidx - 1] # / (1 - np.sum(probs[lidx:ridx+1]))
            sub_cost, sub_moves = opt(lidx - 1, ridx, True)
            dist = abs(cur_pos - points[lidx - 1])
            return prob_at_next * (dist + (1 - points[lidx - 1])) + (1 - prob_at_next) * (dist + sub_cost), ['l'] + sub_moves
        else:
            prob_at_next_left = probs[lidx - 1] # / (1 - np.sum(probs[lidx:ridx+1]))
            prob_at_next_right = probs[ridx + 1] # / (1 - np.sum(probs[lidx:ridx+1]))
            sub_cost_left, sub_moves_left = opt(lidx - 1, ridx, True)
            sub_cost_right, sub_moves_right = opt(lidx, ridx + 1, False)
            dist_left = abs(cur_pos - points[lidx - 1]) 
            dist_right = abs(cur_pos - points[ridx + 1]) 
            left_cost = prob_at_next_left * (dist_left + (1 - points[lidx - 1])) + (1 - prob_at_next_left) * (dist_left + sub_cost_left)
            right_cost = prob_at_next_right * (dist_right + (1 - points[lidx - 1])) + (1 - prob_at_next_right) * (dist_right + sub_cost_right)
            if left_cost < right_cost:
                return left_cost, ['l'] + sub_moves_left
            else:
                return right_cost, ['r'] + sub_moves_right

    min_cost = float('inf')
    best_moves = []
    best_start = None
    for idx in range(len(points)):
        initial_cost = np.linalg.norm(starting_pos - np.array([points[idx], 0]))
        sub_cost, moves = opt(idx, idx, True)

        cost = probs[idx] * (initial_cost + (1 - points[idx])) + (1 - probs[idx]) * (initial_cost + sub_cost)
        print(f"Cost to start at {points[idx]}: {round(cost, 4)}")
        if cost < min_cost:
            min_cost = cost
            best_moves = moves
            best_start = idx

    print(f"Starting position: {np.round(starting_pos, 2)}")
    print(np.round(points, 2))
    print(np.round(probs, 2))
    print([best_start] + best_moves)
    print(round(min_cost, 4))

    # Plotting the results
    plt.figure(figsize=(8, 6))
    
    # Draw the line [0, 1]
    plt.plot([0, 1], [0, 0], 'k-', linewidth=1, label="Line [0, 1]")

    # Draw the points on the line
    plt.scatter(points, np.zeros_like(points), color='blue', label="Points on line")

    # Add probabilities as labels to the points
    for i, prob in enumerate(probs):
        plt.text(points[i], 0.02, f"{prob:.2f}", ha='center', color='blue', fontsize=9)

    # Highlight the starting position
    plt.scatter(starting_pos[0], starting_pos[1], color='green', label="Starting position")

    # Highlight the initial point in red
    plt.scatter(points[best_start], 0, color='red', label="Initial point")

    # Add moves to a text box
    moves_text = f"Moves: {['start'] + best_moves}"
    plt.text(0.05, -0.1, moves_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Add the cost to a text box
    cost_text = f"Cost: {round(min_cost, 4)}"
    plt.text(0.05, -0.15, cost_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


    # Add labels and legend
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Visualization of Points and Starting Position")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("results.png")

if __name__ == "__main__":
    main()