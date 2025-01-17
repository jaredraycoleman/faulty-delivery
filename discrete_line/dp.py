from functools import lru_cache
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pathlib

thisdir = pathlib.Path(__file__).parent

S = (0,0)
T = (1,0)

def solve(H: Tuple[float,float], points: np.ndarray, probs: np.ndarray) -> Tuple[List[Union[int,str]], float]:
    """Solves the discrete line problem using dynamic programming.

    Args:
        H (Tuple[float,float]): Initial position of the finisher.
        points (np.ndarray): Array of points where the object can be found.

    Returns:
        Tuple[List[Union[int,str]], float]: Path and minimum expected delivery time.
    """
    n = len(points)
    points = np.array([0, *points, 1], dtype=float)
    probs = np.array([0, *probs, 0], dtype=float)

    @lru_cache(maxsize=None)
    def d(x: Union[int, Tuple[float,float]], 
          y: Union[int, Tuple[float,float]]) -> float:
        """Distance between points x and y.

        If x or y are integers, they are treated as points on the line
        with coordinates (points[x], 0) or (points[y], 0) respectively.

        Args:
            x (Union[int, Tuple[float,float]]): First point.
            y (Union[int, Tuple[float,float]]): Second point.

        Returns:
            float: Distance between points x and y.
        """
        if isinstance(x, int):
            x = (points[x], 0)
        if isinstance(y, int):
            y = (points[y], 0)
        return np.linalg.norm(np.array(x) - np.array(y))

    @lru_cache(maxsize=None)
    def p(k: int, i: int, j: int):
        """Probability of finding the object at position k, given that the points [i+1, j-1] have been visited.
        
        Args:
            k (int): Position where the object is found.
            i (int): Leftmost point.
            j (int): Rightmost point.

        Returns:
            float: Probability of finding the object at position k.
        """
        prob_visited = np.sum(probs[i+1:j]) # in the interval [i+1, j-1]
        return probs[k] / (1 - prob_visited)

    @lru_cache(maxsize=None)
    def C(i: int, j: int, k: int) -> Tuple[List[int], float]:
        """Computes the minimum expected delivery time assuming that points [i+1,j-1] have been visited and the 
           agent is currently at position k.

        Args:
            i (int): Leftmost point.
            j (int): Rightmost point.
            k (int): Current position.

        Returns:
            Tuple[List[int], float]: Path and minimum expected delivery time.
        """
        assert k == (i+1) or k == (j-1)
        if i == 0: # All points to the left have been visited
            return [p for p in range(j, n+1)] + ["T"], d(k,T)
        if i == 1 and j == n + 1: # All points to the right have been visited and only the first point hasn't been visited
            return [i] + ["T"], d(k,1) + d(1,T)
        if j == n + 1: # All points to the right have been visited
            _path, _cost = C(i-1,n+1,i)
            return [i] + _path, d(k,i) + p(i,i,n+1) * d(i,T) + (1-p(i,i,n+1)) * _cost
        
        _path_left, _cost_left = C(i-1,j,i)
        _path_right, _cost_right = C(i,j+1,j)
        cost_left = d(k,i) + p(i,i,j) * d(i,T) + (1-p(i,i,j)) * _cost_left
        cost_right = d(k,j) + p(j,i,j) * d(j,T) + (1-p(j,i,j)) * _cost_right
        if cost_left < cost_right:
            return [i] + _path_left, cost_left
        else:
            return [j] + _path_right, cost_right

    min_cost = np.inf
    path = []
    for i in range(1, n+1):
        _path, _cost = C(i-1,i+1,i)
        _cost += d(H,i)
        _path = ["H", i] + _path
        print(f"[START] i={i}, _cost={_cost}")
        if _cost < min_cost:
            min_cost = _cost
            path = _path

    return path, min_cost

def plot_solution(
        H: Tuple[float,float],
        points: np.ndarray,
        probs: np.ndarray,
        path: List[Union[int,str]],
        savepath: str = None):
    """Plots the solution to the discrete line problem.

    Args:
        H (Tuple[float,float]): Initial position of the finisher.
        points (np.ndarray): Array of points where the object can be found.
        probs (np.ndarray): Array of probabilities of finding the object at each point.
        path (List[Union[int,str]]): Path to the solution.
        savepath (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([S[0], T[0]], [S[1], T[1]], 'k-', linewidth=1, label="Line [0, 1]")
    plt.scatter(points, np.zeros_like(points), color='black', label="Points")
    plt.scatter(H[0], H[1], color='blue', label="Starting position")
    plt.scatter(T[0], T[1], color='green', label="Finishing position")
    for i, prob in enumerate(probs):
        plt.text(points[i], 0.01, f"{prob:.2f}", fontsize=12, ha='center')
    for i, p in enumerate(path):
        if isinstance(p, int):
            plt.text(points[p-1], -0.04, f"{i}", fontsize=12, ha='center')
    plt.legend()

    # set x and y axis limits
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, H[1] + 0.1)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def main():
    H = (1/2, 1/2)
    points = np.array([0, 1/2, 1])
    probs = np.array([2/100, 49/100, 49/100])
    
    print(f"Points: {np.round(points, 2)}")
    print(f"Probs : {np.round(probs, 2)}")
    
    path, min_cost = solve(H, points, probs)
    plot_solution(H, points, probs, path, savepath=thisdir / "solution.png")
    print(path)
    print(min_cost)

if __name__ == "__main__":
    main()
        