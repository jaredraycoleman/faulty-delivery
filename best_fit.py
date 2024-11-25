import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Data
t = np.array([0., 0.02857143, 0.05714286, 0.08571429, 0.11428571, 0.14285714,
              0.17142857, 0.2, 0.22857143, 0.25714286, 0.28571429, 0.31428571,
              0.34285714, 0.37142857, 0.4, 0.42857143, 0.45714286, 0.48571429,
              0.51428571, 0.54285714, 0.57142857, 0.6, 0.62857143, 0.65714286,
              0.68571429, 0.71428571, 0.74285714, 0.77142857, 0.8, 0.82857143,
              0.85714286, 0.88571429, 0.91428571, 0.94285714, 0.97142857, 1.])
x = np.array([1.11022302e-16, 1.31868092e-02, 2.66667832e-02, 4.04422013e-02,
              5.45152831e-02, 6.88881821e-02, 8.35629799e-02, 9.85416787e-02,
              1.13826194e-01, 1.29418349e-01, 1.45319863e-01, 1.61532347e-01,
              1.78057295e-01, 1.94896070e-01, 2.12049903e-01, 2.29519876e-01,
              2.47306917e-01, 2.65411787e-01, 2.83835071e-01, 3.02577167e-01,
              3.21638274e-01, 3.41018382e-01, 3.60717260e-01, 3.80734444e-01,
              4.01069224e-01, 4.21720635e-01, 4.42687444e-01, 4.63968136e-01,
              4.85560903e-01, 5.07463634e-01, 5.29673901e-01, 5.52188949e-01,
              5.75005684e-01, 5.98120664e-01, 6.21530086e-01, 6.45229780e-01])
y = np.array([0.98850575, 0.96315945, 0.93796785, 0.91293658, 0.88807145, 0.86337841,
              0.83886358, 0.81453325, 0.79039386, 0.76645203, 0.74271454, 0.71918834,
              0.69588056, 0.67279849, 0.64994958, 0.62734147, 0.60498196, 0.58287901,
              0.56104076, 0.53947551, 0.51819169, 0.49719793, 0.47650299, 0.45611576,
              0.43604531, 0.4163008, 0.39689153, 0.37782694, 0.35911654, 0.34076997,
              0.32279692, 0.30520717, 0.28801057, 0.27121699, 0.25483635, 0.23887856])

# Define candidate function forms
def quadratic(t, a, b, c):
    return a * t**2 + b * t + c

def exponential(t, a, b, c):
    return a * np.exp(b * t) + c

# Fit x(t)
params_quad_x, _ = curve_fit(quadratic, t, x)
params_exp_x, _ = curve_fit(exponential, t, x)

# Fit y(t)
params_quad_y, _ = curve_fit(quadratic, t, y)
try:
    params_exp_y, _ = curve_fit(exponential, t, y, p0=(1, -1, 0), maxfev=10000)
except RuntimeError:
    params_exp_y = None

# Generate fitted curves
t_fit = np.linspace(0, 1, 500)
x_quad_fit = quadratic(t_fit, *params_quad_x)
x_exp_fit = exponential(t_fit, *params_exp_x)
y_quad_fit = quadratic(t_fit, *params_quad_y)
if params_exp_y is not None:
    y_exp_fit = exponential(t_fit, *params_exp_y)

# Plot the data and fits
plt.figure(figsize=(10, 6))

# Parametric plot for quadratic fit
plt.plot(x_quad_fit, y_quad_fit, label="Quadratic Fit", linestyle="--")

# Parametric plot for exponential fit (only if succeeded)
if params_exp_y is not None:
    plt.plot(x_exp_fit, y_exp_fit, label="Exponential Fit", linestyle=":")

# Original data
plt.scatter(x, y, color="red", label="Original Data", zorder=5)

plt.title("Parametric Fit of x(t) and y(t)")
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()

# Display the constants
constants = {
    "Quadratic x(t)": {
        "a": params_quad_x[0],
        "b": params_quad_x[1],
        "c": params_quad_x[2],
    },
    "Quadratic y(t)": {
        "a": params_quad_y[0],
        "b": params_quad_y[1],
        "c": params_quad_y[2],
    },
    "Exponential x(t)": {
        "a": params_exp_x[0],
        "b": params_exp_x[1],
        "c": params_exp_x[2],
    },
    "Exponential y(t)": (
        {"a": params_exp_y[0], "b": params_exp_y[1], "c": params_exp_y[2]}
        if params_exp_y is not None
        else "Fit failed"
    ),
}

constants_df = pd.DataFrame(constants).T
print("Fitted Constants:")
print(constants_df)
