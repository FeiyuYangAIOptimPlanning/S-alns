"""
structures.py — 核心数据结构

对齐论文 Section 3：Atrip / Aroute / Gtrip / Groute / Solution 等启发式内部对象。
重要修正（review v2）：
- Atrip.arr_hub 为主变量（= 论文 φ_{dw}），dep_fc / return_fc 为派生
- Gtrip.dep 为主变量（= 论文 τ_{kw}^O）
- 时间桥 t_j^{Rel,H} = atrip.arr_hub（不加 t_h^D）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools


# ---------------------------------------------------------------------------
# 全局 ID 生成器（用于 Atrip / Gtrip / Aroute / Groute）
# ---------------------------------------------------------------------------
class _IDGen:
    """Thread-unsafe 但足够: ALNS 单线程运行."""
    def __init__(self) -> None:
        self._counters: dict[str, itertools.count] = {}

    def next(self, kind: str) -> int:
        if kind not in self._counters:
            self._counters[kind] = itertools.count(0)
        return next(self._counters[kind])

    def reset(self) -> None:
        self._counters = {}


ID_GEN = _IDGen()


# ---------------------------------------------------------------------------
# 节点对象
# ---------------------------------------------------------------------------
@dataclass
class Customer:
    """客户/包裹. 论文 §3.1 中的 j ∈ J."""
    id: int
    code: str                    # "C0", "C1", ...
    coord: tuple[float, float]
    t_rel: float                 # 论文 t_j^{Rel}
    t_due: float                 # 论文 t_j^{Due}
    weight: float                # 论文 g_j
    # —— 动态派生字段（由 timing_subproblem 填）——
    t_rel_H: Optional[float] = None   # 论文 t_j^{Rel,H} = arr_hub of the Atrip
    t_due_H: Optional[float] = None   # 论文 t_j^{Due,H} = τ^O of the Gtrip
    t_due_FC: Optional[float] = None  # = t_due_H − t_h^D  (派生, 导出用)


@dataclass
class Hub:
    """候选/激活 docking hub. 论文 §3.1 中的 h ∈ H."""
    id: int
    code: str                    # "H0", "H1", ...
    coord: tuple[float, float]


@dataclass
class FC:
    """Fulfillment Center, 固定坐标."""
    coord: tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Trip 对象
# ---------------------------------------------------------------------------
@dataclass
class Atrip:
    """UAV trip: FC → hub → FC. 论文 §4."""
    hub: int                      # 目标 hub id
    pkgs: list[int] = field(default_factory=list)  # 客户 id 列表
    id: int = field(default_factory=lambda: ID_GEN.next("Atrip"))
    # —— 由 timing 填 ——
    load: float = 0.0
    early: float = 0.0            # φ_v^{O,early}, arr_hub 视角
    late: float = float("inf")    # φ_v^{O,late},  arr_hub 视角
    arr_hub: float = 0.0          # 论文 φ_{dw} (主变量)
    dep_fc: float = 0.0           # = arr_hub - t_h^D
    return_fc: float = 0.0        # = arr_hub + t_h^D
    feasible: bool = True


@dataclass
class Gtrip:
    """UGV trip: hub → j1 → ... → jn → hub. 论文 §4."""
    hub: int                      # 起止 hub id
    seq: list[int] = field(default_factory=list)   # 客户访问顺序
    id: int = field(default_factory=lambda: ID_GEN.next("Gtrip"))
    # —— 由 timing 填 ——
    load: float = 0.0
    eta: float = 0.0              # 论文 η_{kw} = 总 travel time
    early: float = 0.0            # τ_w^{O,early}
    late: float = float("inf")    # τ_w^{O,late}
    dep: float = 0.0              # τ_{kw}^O (主变量, 从 hub 出发)
    ret: float = 0.0              # τ_{kw}^B (回到 hub)
    cust_arr: dict[int, float] = field(default_factory=dict)  # τ_{kwj}^F
    feasible: bool = True


# ---------------------------------------------------------------------------
# Route 对象
# ---------------------------------------------------------------------------
@dataclass
class Aroute:
    """同一 UAV 的所有 trips, 顺序执行. 论文 §4."""
    trips: list[Atrip] = field(default_factory=list)
    uav_id: int = field(default_factory=lambda: ID_GEN.next("Aroute"))
    feasible: bool = True


@dataclass
class Groute:
    """同一 UGV 的所有 trips, 顺序执行. 单一 hub 绑定."""
    hub: int                       # y_{kh}=1 的 hub
    trips: list[Gtrip] = field(default_factory=list)
    ugv_id: int = field(default_factory=lambda: ID_GEN.next("Groute"))
    feasible: bool = True


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------
@dataclass
class Solution:
    """完整解. 论文 ALNS 中 S_{initial}, S_{current}, S_{best} 等."""
    active_hubs: set[int] = field(default_factory=set)
    aroutes: list[Aroute] = field(default_factory=list)
    groutes: list[Groute] = field(default_factory=list)
    unserved: set[int] = field(default_factory=set)
    # —— 快速索引（变更时同步维护）——
    cust_to_atrip: dict[int, Atrip] = field(default_factory=dict)
    cust_to_gtrip: dict[int, Gtrip] = field(default_factory=dict)
    cust_to_hub: dict[int, int] = field(default_factory=dict)
    # —— 缓存 ——
    cost_cache: Optional[float] = None
    feasible: bool = True

    # ========================= 辅助 =========================
    def all_atrips(self) -> list[Atrip]:
        return [a for r in self.aroutes for a in r.trips]

    def all_gtrips(self) -> list[Gtrip]:
        return [g for r in self.groutes for g in r.trips]

    def atrips_of_hub(self, h: int) -> list[Atrip]:
        return [a for a in self.all_atrips() if a.hub == h]

    def gtrips_of_hub(self, h: int) -> list[Gtrip]:
        return [g for g in self.all_gtrips() if g.hub == h]

    def groutes_of_hub(self, h: int) -> list[Groute]:
        return [r for r in self.groutes if r.hub == h]

    def aroute_of_atrip(self, atrip: Atrip) -> Optional[Aroute]:
        for r in self.aroutes:
            if atrip in r.trips:
                return r
        return None

    def groute_of_gtrip(self, gtrip: Gtrip) -> Optional[Groute]:
        for r in self.groutes:
            if gtrip in r.trips:
                return r
        return None

    def rebuild_cust_indexes(self) -> None:
        """Re-sync cust_to_* 索引.

        出现于 destroy/repair 大幅修改后的 sanity step.
        """
        self.cust_to_atrip.clear()
        self.cust_to_gtrip.clear()
        self.cust_to_hub.clear()
        for a in self.all_atrips():
            for j in a.pkgs:
                self.cust_to_atrip[j] = a
                self.cust_to_hub[j] = a.hub
        for g in self.all_gtrips():
            for j in g.seq:
                self.cust_to_gtrip[j] = g

    def is_fully_served(self, J: list[int]) -> bool:
        return len(self.unserved) == 0 and all(j in self.cust_to_atrip for j in J) \
               and all(j in self.cust_to_gtrip for j in J)

    def clean_empty(self) -> None:
        """清理空 trip / route, 并级联清空 hub."""
        for r in list(self.aroutes):
            r.trips = [t for t in r.trips if len(t.pkgs) > 0]
            if not r.trips:
                self.aroutes.remove(r)
        for r in list(self.groutes):
            r.trips = [t for t in r.trips if len(t.seq) > 0]
            if not r.trips:
                self.groutes.remove(r)
        # 若某个 hub 不再有任何 trip/route, 停用
        referenced_hubs = {a.hub for a in self.all_atrips()} | {g.hub for g in self.all_gtrips()}
        self.active_hubs &= referenced_hubs
