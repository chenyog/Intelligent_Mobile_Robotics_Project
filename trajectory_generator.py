"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from minco import MinJerkOpt


def _densify_path(path: np.ndarray, max_segment_length: float) -> np.ndarray:
    dense = [path[0]]
    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        if seg_len <= max_segment_length:
            dense.append(p1)
            continue
        steps = int(np.ceil(seg_len / max_segment_length))
        for k in range(1, steps + 1):
            alpha = k / steps
            dense.append(p0 + alpha * (p1 - p0))
    return np.array(dense, dtype=np.float32)


def generate_minco_trajectory(
    path: np.ndarray,
    max_speed: float = 1.5,
    dt: float = 0.0666667,
    max_segment_length: float = 1.5,
    env=None,
    is_plotting: bool = True,
    show: bool = True,
):
    if len(path) < 2:
        raise ValueError("Path must contain at least two points.")

    path = np.asarray(path, dtype=np.float32)
    dense_path = _densify_path(path, max_segment_length=max_segment_length)

    positions = np.asarray(dense_path, dtype=np.float32)[None, ...]
    num_pieces = positions.shape[1] - 1

    head_pva = np.zeros((1, 3, 3), dtype=np.float32)
    tail_pva = np.zeros((1, 3, 3), dtype=np.float32)
    head_pva[:, :, 0] = positions[:, 0, :]
    tail_pva[:, :, 0] = positions[:, -1, :]

    segment_vecs = positions[:, 1:, :] - positions[:, :-1, :]
    segment_lengths = np.linalg.norm(segment_vecs, axis=2).squeeze(0)
    durations = np.clip(segment_lengths / max_speed, a_min=0.2, a_max=None)[None, :]

    inner_pts = positions[:, 1:-1, :].transpose(0, 2, 1)
    MJO = MinJerkOpt(head_pva, tail_pva, num_pieces)
    MJO.generate(inner_pts, durations)
    traj = MJO.get_traj()

    total_time = float(traj.get_total_duration()[0])
    times = np.arange(0.0, total_time + 1e-6, dt)
    pos_samples = []
    for t in times:
        pos = traj.get_pos(np.array([t], dtype=np.float32))[0]
        pos_samples.append(pos)
    pos_samples = np.array(pos_samples, dtype=np.float32)

    if env is not None:
        for p in pos_samples:
            if env.is_outside(p) or env.is_collide(p):
                break

    fig = None
    if is_plotting:
        fig = _plot_trajectory(times, pos_samples, path, max_speed=max_speed, show=show)

    return times, pos_samples, traj, fig


def _plot_trajectory(times: np.ndarray, positions: np.ndarray, path: np.ndarray, max_speed: float, show: bool):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(times, positions[:, i], label=f"{labels[i]}(t)")
        ax.scatter(_path_times(path, max_speed), path[:, i], s=15, label="path points")
        ax.set_ylabel(labels[i])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _path_times(path: np.ndarray, max_speed: float) -> np.ndarray:
    times = [0.0]
    for i in range(len(path) - 1):
        seg = np.linalg.norm(path[i + 1] - path[i])
        times.append(times[-1] + max(seg / max_speed, 0.2))
    return np.array(times, dtype=np.float32)
