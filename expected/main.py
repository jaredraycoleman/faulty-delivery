from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, OptimizeResult


def main():
    # Given parameters
    Hx, Hy = 0.2, 3/4  # Center of the constraint circle
    a, b = 1/4, 3/4  # Given points on the x-axis
    p, q = 1/2, 1 # failure probabilities
    radius = a  # Circle radius

    # Compute (x, y) from theta
    def get_xy(theta: float) -> Tuple[float, float]:
        x = Hx + radius * np.cos(theta)
        y = Hy + radius * np.sin(theta)
        return x, y

    # Objective function to minimize
    def objective(theta: float) -> float:
        x, y = get_xy(theta)
        return (
            p * (radius + np.hypot(x - a, y) + 1 - radius) +
            q * (1 - p) * (radius + np.hypot(x - b, y) - (1 - b)) +
            (1 - p) * (1 - q) * 1
        )

    # Optimize over theta
    result: OptimizeResult = minimize_scalar(objective, bounds=(0, 2 * np.pi), method='bounded', tol=1e-9)
    if not result.success:
        print(f"Optimization failed: {result.message}")
        return
    # Extract results
    optimal_theta: float = result.x
    optimal_x, optimal_y = get_xy(optimal_theta)

    # Print results
    print(f"Optimal theta: {optimal_theta:.4f} radians" if result.success else "Optimization failed.")
    print(f"Optimal point: ({optimal_x:.4f}, {optimal_y:.4f})" if result.success else "No optimal point found.")

    # Visualization
    theta_vals = np.linspace(0, 2 * np.pi, 300)
    circle_x = Hx + radius * np.cos(theta_vals)
    circle_y = Hy + radius * np.sin(theta_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(circle_x, circle_y, 'b-', label=None)
    plt.plot([0, 1], [0, 0], 'k-', linewidth=2, label=None)

    # Plot optimal point
    if result.success:
        plt.plot(optimal_x, optimal_y, 'ro', markersize=8, label="Optimal Point")
    plt.plot(Hx, Hy, 'go', markersize=6, label="(Hx, Hy)")
    plt.plot(a, 0, 'mo', markersize=6, label="failure points")
    plt.plot(b, 0, 'mo', markersize=6, label=None)

    # dashed line going through (Hx, Hy) and (x, y) but extended to the line segment
    if result.success:
        # find intersection point of line going through (Hx, Hy) and (x, y) with x-axis
        m = (optimal_y - Hy) / (optimal_x - Hx)
        c = Hy - m * Hx
        x_intersection = -c / m
        plt.plot([Hx, x_intersection], [Hy, 0], 'k--', label=None)
        # plot intersection point
        plt.plot(x_intersection, 0, 'rx', markersize=8, label="Intersection Point")
        print(f"Intersection point: ({x_intersection:.4f}, 0) - Normalized: {(x_intersection-a)/(b-a):.4f}")    

    # Formatting
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Optimization over Theta")
    # place legend outside of plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.axis("equal")

    # add text under legend with values of (Hx, Hy), a, b, p, q and theta, (x, y)
    plt.text(1.01, 0.95 - 0.25, f"(Hx, Hy) = ({Hx}, {Hy})", transform=plt.gca().transAxes)
    plt.text(1.01, 0.90 - 0.25, f"a = {a}, b = {b}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.85 - 0.25, f"p = {p}, q = {q}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.80 - 0.25, f"theta = {optimal_theta:.4f}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.75 - 0.25, f"(x, y) = ({optimal_x:.2f}, {optimal_y:.2f})", transform=plt.gca().transAxes)
    plt.text(1.01, 0.70 - 0.25, f"x-intercept: {x_intersection:.2f} ({(x_intersection-a)/(b-a):.2f})", transform=plt.gca().transAxes)

    # Show plot
    plt.savefig("plot.png")

def test_intercept():
        # Given parameters
    Hx = 0.2  # Center of the constraint circle
    a, b = 1/4, 3/4  # Given points on the x-axis
    p, q = 1/2, 1 # failure probabilities
    radius = a  # Circle radius

    for Hy in np.linspace(0, 1, 100):
        # Compute (x, y) from theta
        def get_xy(theta: float) -> Tuple[float, float]:
            x = Hx + radius * np.cos(theta)
            y = Hy + radius * np.sin(theta)
            return x, y

        # Objective function to minimize
        def objective(theta: float) -> float:
            x, y = get_xy(theta)
            return (
                p * (radius + np.hypot(x - a, y) + 1 - radius) +
                q * (1 - p) * (radius + np.hypot(x - b, y) - (1 - b)) +
                (1 - p) * (1 - q) * 1
            )

        # Optimize over theta
        result: OptimizeResult = minimize_scalar(objective, bounds=(0, 2 * np.pi), method='bounded', tol=1e-9)
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return
        # Extract results
        optimal_theta: float = result.x
        optimal_x, optimal_y = get_xy(optimal_theta)

        m = (optimal_y - Hy) / (optimal_x - Hx)
        c = Hy - m * Hx
        x_intersection = -c / m
        # plt.plot([Hx, x_intersection], [Hy, 0], 'k--', label=None)





if __name__ == "__main__":
    main()