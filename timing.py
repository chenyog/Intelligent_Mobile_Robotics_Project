# 统计计算时间
import time

import numpy as np

from flight_environment import FlightEnvironment
from path_planner import plan_path_a_star
from trajectory_generator import generate_minco_trajectory


def main():
    env = FlightEnvironment(50)
    start = (1, 2, 0)
    goal = (18, 18, 2)
    resolution = 0.5
    max_iterations = 200000

    print("Running A* planning...", flush=True)
    t0 = time.perf_counter()
    path = plan_path_a_star(env, start, goal, resolution=resolution, max_iterations=max_iterations)
    t1 = time.perf_counter()
    print(f"A* planning time: {(t1 - t0) * 1000.0:.3f} ms", flush=True)

    print("Running MINCO trajectory generation...", flush=True)
    generate_minco_trajectory(
        np.asarray(path),
        max_speed=1.5,
        dt=0.01,
        max_segment_length=1.5,
        env=env,
        is_plotting=False,
        show=False,
    )
    t2 = time.perf_counter()

    print(f"MINCO trajectory time: {(t2 - t1) * 1000.0:.3f} ms")
    print(f"Total: {(t2 - t0) * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
