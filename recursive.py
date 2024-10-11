import numpy as np
from typing import Callable, Tuple, List

def line_cr(x: float,
            y: float,
            focus: float,
            fail: float,
            get_opt: Callable[[float], float]) -> float:
    dist_to_focus = np.sqrt((x-focus)**2 + y**2)
    if fail < dist_to_focus:
        a = fail*(focus-x)/dist_to_focus
        b = fail*y/dist_to_focus
        return (1 + np.sqrt((x+a)**2 + (y-b)**2) - fail) / get_opt(fail)
    else:
        return (1 + np.abs(fail-focus)) / get_opt(fail)
    
def best_line_cr(x: float,
                 y: float,
                 get_opt: Callable[[float], float],
                 num_samples: int = 100) -> Tuple[float, float]:
    best_focus, best_focus_worst_fail, best_focus_cr = None, None, np.inf
    for focus in np.linspace(0, 1, num_samples):
        worst_fail, worst_fail_cr = None, 0
        for fail in np.linspace(0, 1, num_samples):
            cr = line_cr(x, y, focus, fail, get_opt)
            if cr > worst_fail_cr:
                worst_fail, worst_fail_cr = fail, cr

        if worst_fail_cr < best_focus_cr:
            best_focus, best_focus_worst_fail, best_focus_cr = focus, worst_fail, worst_fail_cr

    return best_focus, best_focus_worst_fail, best_focus_cr

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

def scale_instance(start, x, y) -> Tuple[float, float]:
    """Scale the instance so that (start, 0) -> (0, 0), (x,y) -> (x^\prime, y^\prime), (1,0) -> (1,0)"""
    return (x-start)/(1-start), y

def line_cr_time_limited(start_x: float,
                         x: float,
                         y: float,
                         num_turns: int,
                         time_passed: float,
                         num_samples: int = 100) -> Tuple[List[Tuple[float, float]], float]:
    if num_turns == 0:
        best_focus, best_focus_worst_fail, best_focus_cr = best_line_cr(x, y, get_opt)
        return [(best_focus, 0)], best_focus_cr
    else:
        new_y = y - (y / num_turns)
        small_x = x-(1-1/(num_turns*y))
        big_x = x+(1-x)/(num_turns*y)
        best_x, best_x_worst_fail, best_x_cr = None, None, np.inf
        for new_x in np.linspace(small_x, big_x, num_samples):
            dist_to_point = np.sqrt((x-new_x)**2 + (y-new_y)**2)
            for fail in np.linspace(start_x, 1, num_samples):
                if fail < dist_to_point: # failure occurs before reaching the point
                    time_passed +
                else: # failure occurs after reaching the point (recurse)
                    pass






def main_line():
    num_samples = 100
    x, y = 1/2, 1/2

    def get_opt(fail: float) -> float:
        return max(fail, np.sqrt((x-fail)**2 + y**2)) + 1 - fail

    # print(line_cr(x, y, 0.6363636363636365, 0.04))
    # return

    best_focus, best_focus_worst_fail, best_focus_cr = best_line_cr(x, y, get_opt, num_samples)

    print(f"Best focus: {best_focus}")
    print(f"Worst fail: {best_focus_worst_fail}")
    print(f"CR: {best_focus_cr}")

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    a = best_focus_worst_fail*(best_focus-x)/np.sqrt((x-best_focus)**2 + y**2)
    b = best_focus_worst_fail*y/np.sqrt((x-best_focus)**2 + y**2)
    
    # plot points (0,0), (1,0), (x,y)
    ax.scatter([0, 1, x], [0, 0, y], color='black')
    # plot line from (x,y) to (best_focus, 0)
    ax.plot([x, best_focus], [y, 0], color='blue')
    # plot point at (best_focus_worst_fail, 0)
    ax.scatter([best_focus_worst_fail], [0], color='red')
    # plot point at (x+a, y-b)
    ax.scatter([x+a], [y-b], color='green')

    plt.show()

    
def main():
    main_line()

if __name__ == '__main__':
    main()
