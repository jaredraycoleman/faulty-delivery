from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar, OptimizeResult, curve_fit
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent

def get_results(Hx, Hy, a, b, p, q, radius) -> Tuple[float, float, float]:
    def get_xy(theta: float) -> Tuple[float, float]:
        x = Hx + radius * np.cos(theta)
        y = Hy + radius * np.sin(theta)
        return x, y

    def objective(theta: float) -> float:
        x, y = get_xy(theta)
        return (
            p * (radius + np.hypot(x - a, y) + 1 - radius) +
            q * (1 - p) * (radius + np.hypot(x - b, y) - (1 - b)) +
            (1 - p) * (1 - q) * 1
        )

    result: OptimizeResult = minimize_scalar(objective, bounds=(0, 2 * np.pi), method='bounded', tol=1e-9)
    if not result.success:
        print(f"Optimization failed: {result.message}")
        raise Exception("Optimization failed")

    optimal_theta: float = result.x
    optimal_x, optimal_y = get_xy(optimal_theta)

    return optimal_theta, optimal_x, optimal_y

def get_fit(Hxs, x_intercepts):
    """Fits the Gaussian-derivative function and returns fitted values."""
    def symmetric_gaussian_derivative(x, A, B, x0):
        return 0.5 + A * (x - x0) * np.exp(-B * (x - x0) ** 2)

    # Fit the function
    popt, _ = curve_fit(symmetric_gaussian_derivative, Hxs, x_intercepts, p0=[0.03, 0.14, 0.5], maxfev=10000)

    # Generate fitted values
    fit_values = symmetric_gaussian_derivative(Hxs, *popt)
    return Hxs, fit_values

def plot_points():
    Hy = 1  
    a, b = 1/4, 3/4  
    p, q = 1/2, 1 
    radius = a  

    x_intercepts = []
    Hxs = np.linspace(-10, 10, 1000)
    for Hx in Hxs:
        try:
            optimal_theta, optimal_x, optimal_y = get_results(Hx, Hy, a, b, p, q, radius)
        except:
            continue
        m = (optimal_y - Hy) / (optimal_x - Hx)
        c = Hy - m * Hx
        x_intersection = -c / m
        x_intercepts.append(x_intersection)

    plt.plot(Hxs, x_intercepts, 'b-', label="Data")

    # Compute and plot the fit
    Hxs_fit, fitted_values = get_fit(Hxs, x_intercepts)
    plt.plot(Hxs_fit, fitted_values, 'r--', label="Gaussian-Derivative Fit")

    plt.xlabel("Hx")
    plt.ylabel("Intersection point")
    plt.title("Intersection point vs Hx")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("intersection_vs_Hx.png")

    df = pd.DataFrame({"Hx": Hxs, "x_intercept": x_intercepts})
    df.to_csv("intersection_vs_Hx.csv", index=False)

def main():
    Hx, Hy = 0.2, 3/4  
    a, b = 1/4, 3/4  
    p, q = 1/2, 1 
    radius = a  

    optimal_theta, optimal_x, optimal_y = get_results(Hx, Hy, a, b, p, q, radius)

    print(f"Optimal theta: {optimal_theta:.4f} radians")
    print(f"Optimal point: ({optimal_x:.4f}, {optimal_y:.4f})")

    theta_vals = np.linspace(0, 2 * np.pi, 300)
    circle_x = Hx + radius * np.cos(theta_vals)
    circle_y = Hy + radius * np.sin(theta_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(circle_x, circle_y, 'b-', label=None)
    plt.plot([0, 1], [0, 0], 'k-', linewidth=2, label=None)

    plt.plot(optimal_x, optimal_y, 'ro', markersize=8, label="Optimal Point")
    plt.plot(Hx, Hy, 'go', markersize=6, label="(Hx, Hy)")
    plt.plot(a, 0, 'mo', markersize=6, label="failure points")
    plt.plot(b, 0, 'mo', markersize=6, label=None)

    m = (optimal_y - Hy) / (optimal_x - Hx)
    c = Hy - m * Hx
    x_intersection = -c / m
    plt.plot([Hx, x_intersection], [Hy, 0], 'k--', label=None)
    plt.plot(x_intersection, 0, 'rx', markersize=8, label="Intersection Point")
    print(f"Intersection point: ({x_intersection:.4f}, 0) - Normalized: {(x_intersection-a)/(b-a):.4f}")    

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Optimization over Theta")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.axis("equal")

    plt.text(1.01, 0.95 - 0.25, f"(Hx, Hy) = ({Hx}, {Hy})", transform=plt.gca().transAxes)
    plt.text(1.01, 0.90 - 0.25, f"a = {a}, b = {b}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.85 - 0.25, f"p = {p}, q = {q}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.80 - 0.25, f"theta = {optimal_theta:.4f}", transform=plt.gca().transAxes)
    plt.text(1.01, 0.75 - 0.25, f"(x, y) = ({optimal_x:.2f}, {optimal_y:.2f})", transform=plt.gca().transAxes)
    plt.text(1.01, 0.70 - 0.25, f"x-intercept: {x_intersection:.2f} ({(x_intersection-a)/(b-a):.2f})", transform=plt.gca().transAxes)

    plt.savefig("plot.png")

if __name__ == "__main__":
    plot_points()
