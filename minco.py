import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Trajectory:
    def __init__(self, durations=None, coeff_mats=None):
        self.durations = np.asarray(durations, dtype=np.float32)
        self.coeff_mats = np.asarray(coeff_mats, dtype=np.float32)
        self.num_env, self.N = self.durations.shape

    def __getitem__(self, index):
        new_durations = self.durations[index]
        new_coeff_mats = self.coeff_mats[index]
        return Trajectory(durations=new_durations, coeff_mats=new_coeff_mats)

    def __setitem__(self, index, traj_new):
        self.durations[index] = traj_new.durations
        self.coeff_mats[index] = traj_new.coeff_mats

    def get_total_duration(self):
        return self.durations.sum(axis=1)

    def locate_piece_idx(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=np.float32)
        if t.ndim == 0:
            t = np.full(self.num_env, t, dtype=np.float32)
        if t.shape[0] != self.num_env:
            raise ValueError("Time vector length must match num_env.")

        cum_durations = np.concatenate(
            [np.zeros((self.num_env, 1), dtype=np.float32), self.durations.cumsum(axis=1)],
            axis=1,
        )
        idx = np.zeros(self.num_env, dtype=np.int64)
        t_local = np.zeros(self.num_env, dtype=np.float32)
        for env in range(self.num_env):
            idx_env = np.searchsorted(cum_durations[env], t[env], side="right") - 1
            idx_env = max(0, min(self.N - 1, idx_env))
            idx[env] = idx_env
            t_local[env] = t[env] - cum_durations[env, idx_env]

        mask = t_local > self.durations[np.arange(self.num_env), idx] + 1e-6
        if np.any(mask):
            for env in np.where(mask)[0]:
                logger.error("Retrieved timestamp out of trajectory duration in environment %d #^#", int(env))

        return idx, t_local

    def get_pos(self, t: np.ndarray) -> np.ndarray:
        idx, t_local = self.locate_piece_idx(t)
        pos = np.zeros((self.num_env, 3), dtype=np.float32)
        tn = np.ones(self.num_env, dtype=np.float32)

        coeff = self.coeff_mats[np.arange(self.num_env), idx]
        for i in range(5, -1, -1):
            pos += tn[:, None] * coeff[:, :, i]
            tn *= t_local
        return pos

    def get_vel(self, t: np.ndarray) -> np.ndarray:
        idx, t_local = self.locate_piece_idx(t)
        vel = np.zeros((self.num_env, 3), dtype=np.float32)
        tn = np.ones(self.num_env, dtype=np.float32)
        n = 1

        coeff = self.coeff_mats[np.arange(self.num_env), idx]
        for i in range(4, -1, -1):
            vel += n * tn[:, None] * coeff[:, :, i]
            tn *= t_local
            n += 1
        return vel

    def get_acc(self, t: np.ndarray) -> np.ndarray:
        idx, t_local = self.locate_piece_idx(t)
        acc = np.zeros((self.num_env, 3), dtype=np.float32)
        tn = np.ones(self.num_env, dtype=np.float32)
        m = 1
        n = 2

        coeff = self.coeff_mats[np.arange(self.num_env), idx]
        for i in range(3, -1, -1):
            acc += m * n * tn[:, None] * coeff[:, :, i]
            tn *= t_local
            m += 1
            n += 1
        return acc

    def get_jer(self, t: np.ndarray) -> np.ndarray:
        idx, t_local = self.locate_piece_idx(t)
        jer = np.zeros((self.num_env, 3), dtype=np.float32)
        tn = np.ones(self.num_env, dtype=np.float32)
        l = 1
        m = 2
        n = 3

        coeff = self.coeff_mats[np.arange(self.num_env), idx]
        for i in range(2, -1, -1):
            jer += l * m * n * tn[:, None] * coeff[:, :, i]
            tn *= t_local
            l += 1
            m += 1
            n += 1
        return jer


class BandedSystem:
    def __init__(self, num_env, n, p, q):
        self.N = n
        self.lower_bw = p
        self.upper_bw = q
        self.mat_data = np.zeros((num_env, self.N * (self.lower_bw + self.upper_bw + 1)), dtype=np.float32)

    def __call__(self, i, j):
        return self.mat_data[:, (i - j + self.upper_bw) * self.N + j]

    def factorizeLU(self):
        eps = 1e-6
        for k in range(self.N - 1):
            iM = min(k + self.lower_bw, self.N - 1)
            cVl = self(k, k)
            cVl[np.abs(cVl) < eps] += eps
            for i in range(k + 1, iM + 1):
                mask = self(i, k) != 0.0
                if np.any(mask):
                    self(i, k)[mask] /= cVl[mask]

            jM = min(k + self.upper_bw, self.N - 1)
            for j in range(k + 1, jM + 1):
                cVl = self(k, j)
                mask = cVl != 0.0
                if np.any(mask):
                    for i in range(k + 1, iM + 1):
                        mask_ = (self(i, k) != 0.0) & mask
                        if np.any(mask_):
                            self(i, j)[mask_] -= self(i, k)[mask_] * cVl[mask_]

    def solve(self, b: np.ndarray):
        for j in range(self.N):
            iM = min(j + self.lower_bw, self.N - 1)
            for i in range(j + 1, iM + 1):
                mask = self(i, j) != 0.0
                if np.any(mask):
                    b[mask, i] -= self(i, j)[mask][:, None] * b[mask, j]

        eps = 1e-6
        for j in range(self.N - 1, -1, -1):
            self(j, j)[np.abs(self(j, j)) < eps] += eps
            b[:, j] /= self(j, j)[:, None]
            iM = max(0, j - self.upper_bw)
            for i in range(iM, j):
                mask = self(i, j) != 0.0
                if np.any(mask):
                    b[mask, i] -= self(i, j)[mask][:, None] * b[mask, j]


class MinJerkOpt:
    def __init__(self, head_pva, tail_pva, num_pieces):
        self.num_env = head_pva.shape[0]
        self.N = num_pieces
        self.head_pva = np.asarray(head_pva, dtype=np.float32).copy()
        self.tail_pva = np.asarray(tail_pva, dtype=np.float32).copy()

    def generate(self, inner_pts_, durations):
        inner_pts = np.asarray(inner_pts_, dtype=np.float32).copy()
        self.t1 = np.asarray(durations, dtype=np.float32).copy()

        if inner_pts.size == 0 or inner_pts.shape[2] == 0:
            t1_inv = 1.0 / self.t1
            t2_inv = t1_inv * t1_inv
            t3_inv = t2_inv * t1_inv
            t4_inv = t2_inv * t2_inv
            t5_inv = t4_inv * t1_inv
            coeff_mat_reversed = np.zeros((self.num_env, 3, 6), dtype=np.float32)
            coeff_mat_reversed[:, :, 5] = (
                0.5 * (self.tail_pva[:, :, 2] - self.head_pva[:, :, 2]) * t3_inv
                - 3.0 * (self.head_pva[:, :, 1] + self.tail_pva[:, :, 1]) * t4_inv
                + 6.0 * (self.tail_pva[:, :, 0] - self.head_pva[:, :, 0]) * t5_inv
            )
            coeff_mat_reversed[:, :, 4] = (
                (-self.tail_pva[:, :, 2] + 1.5 * self.head_pva[:, :, 2]) * t2_inv
                + (8.0 * self.head_pva[:, :, 1] + 7.0 * self.tail_pva[:, :, 1]) * t3_inv
                + 15.0 * (-self.tail_pva[:, :, 0] + self.head_pva[:, :, 0]) * t4_inv
            )
            coeff_mat_reversed[:, :, 3] = (
                (0.5 * self.tail_pva[:, :, 2] - 1.5 * self.head_pva[:, :, 2]) * t1_inv
                - (6.0 * self.head_pva[:, :, 1] + 4.0 * self.tail_pva[:, :, 1]) * t2_inv
                + 10.0 * (self.tail_pva[:, :, 0] - self.head_pva[:, :, 0]) * t3_inv
            )
            coeff_mat_reversed[:, :, 2] = 0.5 * self.head_pva[:, :, 2]
            coeff_mat_reversed[:, :, 1] = self.head_pva[:, :, 1]
            coeff_mat_reversed[:, :, 0] = self.head_pva[:, :, 0]
            self.b = coeff_mat_reversed.transpose(0, 2, 1)
        else:
            t2 = self.t1 * self.t1
            t3 = t2 * self.t1
            t4 = t2 * t2
            t5 = t4 * self.t1

            A = BandedSystem(self.num_env, 6 * self.N, 6, 6)
            self.b = np.zeros((self.num_env, 6 * self.N, 3), dtype=np.float32)

            A(0, 0)[:] = 1.0
            A(1, 1)[:] = 1.0
            A(2, 2)[:] = 2.0
            self.b[:, 0] = self.head_pva[:, :, 0]
            self.b[:, 1] = self.head_pva[:, :, 1]
            self.b[:, 2] = self.head_pva[:, :, 2]

            for i in range(self.N - 1):
                A(6 * i + 3, 6 * i + 3)[:] = 6.0
                A(6 * i + 3, 6 * i + 4)[:] = 24.0 * self.t1[:, i]
                A(6 * i + 3, 6 * i + 5)[:] = 60.0 * t2[:, i]
                A(6 * i + 3, 6 * i + 9)[:] = -6.0
                A(6 * i + 4, 6 * i + 4)[:] = 24.0
                A(6 * i + 4, 6 * i + 5)[:] = 120.0 * self.t1[:, i]
                A(6 * i + 4, 6 * i + 10)[:] = -24.0
                A(6 * i + 5, 6 * i)[:] = 1.0
                A(6 * i + 5, 6 * i + 1)[:] = self.t1[:, i]
                A(6 * i + 5, 6 * i + 2)[:] = t2[:, i]
                A(6 * i + 5, 6 * i + 3)[:] = t3[:, i]
                A(6 * i + 5, 6 * i + 4)[:] = t4[:, i]
                A(6 * i + 5, 6 * i + 5)[:] = t5[:, i]
                A(6 * i + 6, 6 * i)[:] = 1.0
                A(6 * i + 6, 6 * i + 1)[:] = self.t1[:, i]
                A(6 * i + 6, 6 * i + 2)[:] = t2[:, i]
                A(6 * i + 6, 6 * i + 3)[:] = t3[:, i]
                A(6 * i + 6, 6 * i + 4)[:] = t4[:, i]
                A(6 * i + 6, 6 * i + 5)[:] = t5[:, i]
                A(6 * i + 6, 6 * i + 6)[:] = -1.0
                A(6 * i + 7, 6 * i + 1)[:] = 1.0
                A(6 * i + 7, 6 * i + 2)[:] = 2 * self.t1[:, i]
                A(6 * i + 7, 6 * i + 3)[:] = 3 * t2[:, i]
                A(6 * i + 7, 6 * i + 4)[:] = 4 * t3[:, i]
                A(6 * i + 7, 6 * i + 5)[:] = 5 * t4[:, i]
                A(6 * i + 7, 6 * i + 7)[:] = -1.0
                A(6 * i + 8, 6 * i + 2)[:] = 2.0
                A(6 * i + 8, 6 * i + 3)[:] = 6 * self.t1[:, i]
                A(6 * i + 8, 6 * i + 4)[:] = 12 * t2[:, i]
                A(6 * i + 8, 6 * i + 5)[:] = 20 * t3[:, i]
                A(6 * i + 8, 6 * i + 8)[:] = -2.0
                self.b[:, 6 * i + 5] = inner_pts[:, :, i]

            A(6 * self.N - 3, 6 * self.N - 6)[:] = 1.0
            A(6 * self.N - 3, 6 * self.N - 5)[:] = self.t1[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 4)[:] = t2[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 3)[:] = t3[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 2)[:] = t4[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 1)[:] = t5[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 5)[:] = 1.0
            A(6 * self.N - 2, 6 * self.N - 4)[:] = 2 * self.t1[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 3)[:] = 3 * t2[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 2)[:] = 4 * t3[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 1)[:] = 5 * t4[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 4)[:] = 2.0
            A(6 * self.N - 1, 6 * self.N - 3)[:] = 6 * self.t1[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 2)[:] = 12 * t2[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 1)[:] = 20 * t3[:, self.N - 1]

            self.b[:, 6 * self.N - 3] = self.tail_pva[:, :, 0]
            self.b[:, 6 * self.N - 2] = self.tail_pva[:, :, 1]
            self.b[:, 6 * self.N - 1] = self.tail_pva[:, :, 2]

            A.factorizeLU()
            A.solve(self.b)

    def get_traj(self):
        blocks = self.b.reshape(self.num_env, self.N, 6, 3)
        coeff_mats = np.flip(blocks.transpose(0, 1, 3, 2), axis=3)
        return Trajectory(self.t1, coeff_mats)
