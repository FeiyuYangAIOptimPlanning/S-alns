"""
travel_times.py — 距离 / 旅行时间矩阵

对齐论文 §5.1:
- 小规模: Euclidean distance, UAV_v=1.0 km/min, UGV_v=0.1 km/min
- 大规模: Manhattan distance

论文 t_h^D 与 t_{j,j'}^K 均包含 handling (takeoff/landing/unload) 时间。
小规模实例中 handling 常被吸收进 uav_v / ugv_v 的等效速度里，不单独建模。
如需单独建模 handling_time,可通过 config 参数 hand_D / hand_K 叠加。
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
import math


class DistanceMode(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


def _dist(p: tuple[float, float], q: tuple[float, float], mode: DistanceMode) -> float:
    if mode == DistanceMode.EUCLIDEAN:
        return math.hypot(p[0] - q[0], p[1] - q[1])
    elif mode == DistanceMode.MANHATTAN:
        return abs(p[0] - q[0]) + abs(p[1] - q[1])
    raise ValueError(f"Unknown distance mode: {mode}")


@dataclass
class TravelTimes:
    """所有预计算的旅行时间。

    - t_hD[h]:  FC → hub 单程飞行时间 (含 handling)
    - t_hj_K[h][j]: hub → customer 时间 (UGV)
    - t_jh_K[j][h]: customer → hub 时间 (UGV)
    - t_jj_K[j][j']: customer → customer 时间 (UGV), j==j' 时为 ∞
    """
    t_hD: list[float]
    t_hj_K: list[list[float]]  # indexed [h][j]
    t_jh_K: list[list[float]]  # indexed [j][h]
    t_jj_K: list[list[float]]  # indexed [j1][j2]
    mode: DistanceMode

    @classmethod
    def build(
        cls,
        customer_coords: list[tuple[float, float]],
        hub_coords: list[tuple[float, float]],
        fc_coord: tuple[float, float],
        uav_v: float,
        ugv_v: float,
        mode: DistanceMode,
        hand_D: float = 0.0,
        hand_K: float = 0.0,
    ) -> "TravelTimes":
        n_j = len(customer_coords)
        n_h = len(hub_coords)
        t_hD = [_dist(fc_coord, hub_coords[h], mode) / uav_v + hand_D for h in range(n_h)]
        t_hj_K = [[_dist(hub_coords[h], customer_coords[j], mode) / ugv_v + hand_K
                   for j in range(n_j)] for h in range(n_h)]
        t_jh_K = [[_dist(customer_coords[j], hub_coords[h], mode) / ugv_v + hand_K
                   for h in range(n_h)] for j in range(n_j)]
        t_jj_K = [[(_dist(customer_coords[a], customer_coords[b], mode) / ugv_v + hand_K)
                   if a != b else float("inf")
                   for b in range(n_j)] for a in range(n_j)]
        return cls(t_hD=t_hD, t_hj_K=t_hj_K, t_jh_K=t_jh_K, t_jj_K=t_jj_K, mode=mode)

    # ---- 访问便利函数 ----
    def gtrip_eta(self, hub: int, seq: list[int]) -> float:
        """计算一条 Gtrip 的总 travel time η (含 hub→first, between, last→hub).

        对应论文 Eq. (20) 右侧的三部分之和.
        """
        if not seq:
            return 0.0
        eta = self.t_hj_K[hub][seq[0]]
        for i in range(len(seq) - 1):
            eta += self.t_jj_K[seq[i]][seq[i + 1]]
        eta += self.t_jh_K[seq[-1]][hub]
        return eta
