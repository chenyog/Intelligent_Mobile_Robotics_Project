"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np


GridIndex = Tuple[int, int, int]


def _world_to_grid(point: np.ndarray, resolution: float) -> GridIndex:
    return (int(round(point[0] / resolution)),
            int(round(point[1] / resolution)),
            int(round(point[2] / resolution)))


def _grid_to_world(index: GridIndex, resolution: float) -> np.ndarray:
    return np.array(index, dtype=np.float32) * resolution


def _build_neighbor_offsets() -> List[Tuple[int, int, int, float]]:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cost = (dx * dx + dy * dy + dz * dz) ** 0.5
                offsets.append((dx, dy, dz, cost))
    return offsets


_NEIGHBOR_OFFSETS = _build_neighbor_offsets()


def _line_is_free(env, p0: np.ndarray, p1: np.ndarray, step: float) -> bool:
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return True
    num = max(2, int(dist / step) + 1)
    for alpha in np.linspace(0.0, 1.0, num=num):
        p = p0 + alpha * vec
        if env.is_outside(p) or env.is_collide(p):
            return False
    return True


def _shortcut_path(env, path: np.ndarray, step: float) -> np.ndarray:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _line_is_free(env, path[i], path[j], step):
                break
            j -= 1
        simplified.append(path[j])
        i = j
    return np.array(simplified, dtype=np.float32)


def plan_path_a_star(env, start: Tuple[float, float, float], goal: Tuple[float, float, float],
                     resolution: float = 0.5, max_iterations: int = 200000) -> np.ndarray:
    start_np = np.array(start, dtype=np.float32)
    goal_np = np.array(goal, dtype=np.float32)

    if env.is_outside(start_np) or env.is_collide(start_np):
        raise ValueError("Start point is outside the environment or in collision.")
    if env.is_outside(goal_np) or env.is_collide(goal_np):
        raise ValueError("Goal point is outside the environment or in collision.")

    start_idx = _world_to_grid(start_np, resolution)
    goal_idx = _world_to_grid(goal_np, resolution)

    open_heap: List[Tuple[float, GridIndex]] = []
    heapq.heappush(open_heap, (0.0, start_idx))

    came_from: Dict[GridIndex, Optional[GridIndex]] = {start_idx: None}
    g_score: Dict[GridIndex, float] = {start_idx: 0.0}

    def heuristic(a: GridIndex, b: GridIndex) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    iterations = 0
    while open_heap and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_heap)

        if current == goal_idx:
            break

        current_world = _grid_to_world(current, resolution)
        if env.is_outside(current_world) or env.is_collide(current_world):
            continue

        for dx, dy, dz, cost in _NEIGHBOR_OFFSETS:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            neighbor_world = _grid_to_world(neighbor, resolution)
            if env.is_outside(neighbor_world) or env.is_collide(neighbor_world):
                continue

            tentative = g_score[current] + cost * resolution
            if neighbor not in g_score or tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal_idx) * resolution
                heapq.heappush(open_heap, (f_score, neighbor))

    if goal_idx not in came_from:
        raise RuntimeError("A* failed to find a collision-free path.")

    path_indices = []
    node = goal_idx
    while node is not None:
        path_indices.append(node)
        node = came_from[node]
    path_indices.reverse()

    path = np.array([_grid_to_world(idx, resolution) for idx in path_indices], dtype=np.float32)
    path[0] = start_np
    path[-1] = goal_np

    return path

