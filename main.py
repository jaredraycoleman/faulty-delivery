import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List
import plotly.express as px

def main():
    robot_start = np.array([np.sqrt(3), -1])
    target_pos = np.array([robot_start[0], 1])
    robot = np.copy(robot_start)
    dt = 1e-4

    def hybrid_update(point: np.ndarray) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        def update(helper: np.ndarray, robot: np.ndarray, dt: float) -> np.ndarray:
            if robot[1] >= point[1]: 
                return helper + (robot - helper) / np.linalg.norm(robot - helper) * dt
            else: # move towards point
                return helper + (point - helper) / np.linalg.norm(point - helper) * dt
        return update
    
    def bangbang_update(points: List[np.ndarray]) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        def update(helper: np.ndarray, robot: np.ndarray, dt: float) -> np.ndarray:
            for point in points:
                if robot[1] > point[1]:
                    continue

                # move toward point
                return helper + (point - helper) / np.linalg.norm(point - helper) * dt
            
            # move toward target
            return helper + (target_pos - helper) / np.linalg.norm(target_pos - helper) * dt

        return update
    
    def pursue_x_update(x: float) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        def update(helper: np.ndarray, robot: np.ndarray, dt: float) -> np.ndarray:
            point_x = robot + ((target_pos - robot) * x)
            return helper + (point_x - helper) / np.linalg.norm(point_x - helper) * dt
        
        return update
    
    def bangbang_x_update(x: float) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        point_x = robot_start + ((target_pos - robot_start) * x)
        return bangbang_update([point_x])

    helpers = {
        "pursuit": {
            "color": "blue",
            "update": lambda helper, robot, dt: helper + (robot - helper) / np.linalg.norm(robot - helper) * dt,
            "description": "Move towards robot"
        },
        "go-to-end": {
            "color": "red",
            "update": lambda helper, robot, dt: helper + (target_pos - helper) / np.linalg.norm(target_pos - helper) * dt,
            "description": "Move towards the destination"
        },
        "pursue-x": {
            "color": "orange",
            "update": pursue_x_update(x=2/3),
            "description": "Move towards the midpoint of the robot and target"
        },
        "bangbang-x": {
            "color": "green",
            "update": bangbang_x_update(x=0.83),
            "description": "Move towards the robot until it reaches a certain point, then move towards the target"
        }
    }

    for helper in helpers:
        helpers[helper]["pos"] = np.array([0.0, 0.0])

    t = 0
    rows = []
    while robot[1] <= 1.0:
        optimal_delivery_time = max(np.linalg.norm(robot), t) + 2 - t
        for helper in helpers:
            delivery_time = 2 + np.linalg.norm(helpers[helper]["pos"] - robot)
            competitive_ratio = delivery_time / optimal_delivery_time
            if competitive_ratio > helpers[helper].get("CR", 0):
                helpers[helper]["CR"] = competitive_ratio
                helpers[helper]["CR_t"] = t
                helpers[helper]["CR_pos"] = np.copy(helpers[helper]["pos"]).round(2)
                helpers[helper]["CR_robot"] = np.copy(robot).round(2)

            # update helper position
            helpers[helper]["pos"] = helpers[helper]["update"](
                helpers[helper]["pos"], robot, dt
            )

            rows.append([t, helper, helpers[helper]["pos"][0], helpers[helper]["pos"][1]])

        # robot moves in y direction
        robot[1] += dt
        # update time
        t += dt

    df = pd.DataFrame(rows, columns=["t", "helper", "x", "y"])
    
    fig, ax = plt.subplots(figsize=(10, 10))

    for helper in helpers:
        ax.plot(df[df["helper"] == helper]["x"], df[df["helper"] == helper]["y"], label=helper, color=helpers[helper]["color"])
    ax.plot(robot_start[0], robot_start[1], "ko", label="robot start")
    ax.plot(target_pos[0], target_pos[1], "ko", label="target")
    ax.plot(0, 0, "ko", label="helper start")

    # plot worst-case positions for each helper
    for helper in helpers:
        ax.plot(
            helpers[helper]["CR_pos"][0], helpers[helper]["CR_pos"][1],
            color=helpers[helper]["color"],
            marker="x",
            linestyle="None",
            label=f"{helper} worst-case",
            markersize=20,
        )
        ax.plot(
            helpers[helper]["CR_robot"][0], helpers[helper]["CR_robot"][1],
            color=helpers[helper]["color"],
            marker="x",
            linestyle="None",
            label=f"{helper} worst-case",
            markersize=20
        )
        
    # put legend outside of plot
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # increase right margin to fit legend
    plt.subplots_adjust(right=0.7)

    # keep aspect ratio
    ax.set_aspect("equal")
    plt.savefig("plot.png")

    # table
    df_table = pd.DataFrame(
        [
            [helper, helpers[helper]["CR"], helpers[helper]["CR_t"], helpers[helper]["CR_pos"], helpers[helper]["CR_robot"]]
            for helper in helpers
        ],
        columns=["Helper", "CR", "t", "Position", "Robot"]
    ).sort_values("CR", ascending=False)
    print(df_table)


    # evaluate different x values for pursue-x
    rows = []
    helpers_x = {
        "pursue-x": pursue_x_update,
        "bangbang-x": bangbang_x_update,
    }
    for x in np.linspace(0.5, 1, 10):
        for alg, update_func in helpers_x.items():
            update = update_func(x)
            cr = 0
            robot = np.copy(robot_start)
            helper = np.array([0.0, 0.0])
            t = 0
            while robot[1] <= 1.0:
                optimal_delivery_time = max(np.linalg.norm(robot), t) + 2 - t
                delivery_time = 2 + np.linalg.norm(helper - robot)
                competitive_ratio = delivery_time / optimal_delivery_time
                cr = max(cr, competitive_ratio)

                helper = update(helper, robot, dt)
                robot[1] += dt
                t += dt

            rows.append([alg, x, cr])

    df = pd.DataFrame(rows, columns=["alg", "x", "CR"])
    print(df)
    fig, ax = plt.subplots()
    for alg in df["alg"].unique():
        ax.plot(df[df["alg"] == alg]["x"], df[df["alg"] == alg]["CR"], label=alg)
    ax.legend()
    fig.savefig("plot_x.png")

    # get best x for each algorithm
    df_best_x = df.groupby("alg")["CR"].idxmin()
    print(df.loc[df_best_x])



if __name__ == "__main__":
    main()


