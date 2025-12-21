import numpy as np
import matplotlib.pyplot as plt

from flight_environment import FlightEnvironment
from path_planner import plan_path_a_star
from trajectory_generator import generate_minco_trajectory

env = FlightEnvironment(50)

# Tunable parameters
START = (1, 2, 0)
GOAL = (18, 18, 3)
ASTAR_RESOLUTION = 0.5
ASTAR_MAX_ITER = 200000
MAX_SPEED = 2.0
DT = 0.01
MAX_SEGMENT_LENGTH = 1.5

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

path = plan_path_a_star(
    env,
    START,
    GOAL,
    resolution=ASTAR_RESOLUTION,
    max_iterations=ASTAR_MAX_ITER,
)

# --------------------------------------------------------------------------------------------------- #


env.plot_cylinders(path, show=False)
# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

generate_minco_trajectory(
    np.asarray(path),
    max_speed=MAX_SPEED,
    dt=DT,
    max_segment_length=MAX_SEGMENT_LENGTH,
    env=env,
    is_plotting=True,
    show=False,
)

plt.show()

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
