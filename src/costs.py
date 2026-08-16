"""
costs.py — 成本分解, 对齐论文 Eq. (1)

min Σ c^H·a_h + Σ c^{D,Inv}·x_d + Σ c^{K,Inv}·y_{kh}
  + Σ 2·c^{D,Trav}·t_h^D·p_{dwh}
  + Σ c^{K,Trav}·η_{kw}
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from .structures import Solution
from .data_loader import DataContainer
from .travel_times import TravelTimes


@dataclass
class CostBreakdown:
    hub_cost: float = 0.0
    uav_inv_cost: float = 0.0
    ugv_inv_cost: float = 0.0
    uav_trav_cost: float = 0.0
    ugv_trav_cost: float = 0.0

    @property
    def total(self) -> float:
        return (self.hub_cost + self.uav_inv_cost + self.ugv_inv_cost
                + self.uav_trav_cost + self.ugv_trav_cost)

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


def compute_cost(sol: Solution, data: DataContainer, tt: TravelTimes) -> CostBreakdown:
    cb = CostBreakdown()
    cb.hub_cost = len(sol.active_hubs) * data.cost_H
    cb.uav_inv_cost = len(sol.aroutes) * data.cost_D_inv
    cb.ugv_inv_cost = len(sol.groutes) * data.cost_K_inv
    for a in sol.all_atrips():
        cb.uav_trav_cost += 2 * data.cost_D_trav * tt.t_hD[a.hub]
    for g in sol.all_gtrips():
        cb.ugv_trav_cost += data.cost_K_trav * g.eta
    sol.cost_cache = cb.total
    return cb


def cost(sol: Solution, data: DataContainer, tt: TravelTimes) -> float:
    return compute_cost(sol, data, tt).total
