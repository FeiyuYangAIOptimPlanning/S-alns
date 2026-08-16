"""
data_loader.py — CSV 实例加载 + 候选 hub 预检查

对齐论文 §3.1 (问题描述) 与 §5.1 (参数设置)。
支持的 CSV 格式见 IMPLEMENTATION_PLAN §0 / README。

默认值策略: 若 params CSV 缺项, 按论文小规模 §5.1 默认值兜底, 并 log.warning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import logging

from .structures import Customer, Hub, FC, ID_GEN
from .travel_times import TravelTimes, DistanceMode

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认参数表（论文 §5.1 / 计划 §0）
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict[str, object] = {
    "cost_H": 1000.0,
    "cost_D_inv": 500.0,
    "cost_K_inv": 1000.0,
    "cost_D_trav": 5.0,
    "cost_K_trav": 2.0,
    "t_D_set": 10.0,
    "t_K_set": 20.0,
    "l_D": 10.0,
    "l_K": 20.0,
    "UAV_v": 1.0,
    "UGV_v": 0.1,
    "e_D": 30.0,
    "e_K": 60.0,
    "Distance_Mode": "euclidean",
    "hand_D": 0.0,
    "hand_K": 0.0,
    # UAV/UGV/W 池大小的 soft cap（启发式不限制, 仅做上限 sanity）:
    "max_uav": 100,
    "max_ugv": 100,
    "max_trips_per_vehicle": 50,
}


@dataclass
class DataContainer:
    """完整算例数据。"""
    customers: list[Customer]
    hubs: list[Hub]
    fc: FC
    # —— 索引集合 ——
    J: list[int] = field(default_factory=list)      # 客户 id 集合
    H: list[int] = field(default_factory=list)      # hub id 集合
    D: list[int] = field(default_factory=list)      # UAV id 池 (逻辑上)
    K: list[int] = field(default_factory=list)      # UGV id 池
    W: list[int] = field(default_factory=list)      # trip index 池
    # —— 参数（来自 CSV 或默认）——
    cost_H: float = 1000.0
    cost_D_inv: float = 500.0
    cost_K_inv: float = 1000.0
    cost_D_trav: float = 5.0
    cost_K_trav: float = 2.0
    t_D_set: float = 10.0
    t_K_set: float = 20.0
    l_D: float = 10.0
    l_K: float = 20.0
    UAV_v: float = 1.0
    UGV_v: float = 0.1
    e_D: float = 30.0
    e_K: float = 60.0
    distance_mode: DistanceMode = DistanceMode.EUCLIDEAN
    hand_D: float = 0.0
    hand_K: float = 0.0
    instance_id: str = "unknown"

    def cust_by_id(self, j: int) -> Customer:
        return self.customers[j]


# ---------------------------------------------------------------------------
# 加载器
# ---------------------------------------------------------------------------
def load_instance(
    nodes_csv: str,
    params_csv: str,
    instance_id: str = "unknown",
) -> DataContainer:
    """从两个 CSV 加载实例.

    Args:
        nodes_csv: NodeID, Type, X, Y, ReleaseTime, DueTime, Weight
        params_csv: Parameter, Value
    Returns:
        DataContainer
    Raises:
        ValueError: 格式错误 / 必填字段缺失
    """
    ID_GEN.reset()  # 每次加载重置全局 ID 生成器

    # --- parse nodes ---
    df_nodes = pd.read_csv(nodes_csv)
    required = {"NodeID", "Type", "X", "Y", "ReleaseTime", "DueTime", "Weight"}
    missing = required - set(df_nodes.columns)
    if missing:
        raise ValueError(f"nodes CSV 缺必填列: {missing}")

    customers: list[Customer] = []
    hubs: list[Hub] = []
    for _, row in df_nodes.iterrows():
        nid = str(row["NodeID"]).strip()
        typ = str(row["Type"]).strip().lower()
        x, y = float(row["X"]), float(row["Y"])
        if typ == "customer":
            customers.append(Customer(
                id=len(customers),  # 0-based 顺序 id
                code=nid,
                coord=(x, y),
                t_rel=float(row["ReleaseTime"]),
                t_due=float(row["DueTime"]),
                weight=float(row["Weight"]),
            ))
        elif typ == "hub":
            hubs.append(Hub(
                id=len(hubs),
                code=nid,
                coord=(x, y),
            ))
        else:
            raise ValueError(f"未知 Type: {typ} (NodeID={nid})")

    if not customers:
        raise ValueError("No customers found in nodes CSV")
    if not hubs:
        raise ValueError("No hubs found in nodes CSV")

    # --- parse params ---
    df_params = pd.read_csv(params_csv)
    required_p = {"Parameter", "Value"}
    if not required_p.issubset(df_params.columns):
        raise ValueError(f"params CSV 缺必填列: {required_p}")
    param_map: dict[str, object] = {}
    for _, row in df_params.iterrows():
        key = str(row["Parameter"]).strip()
        val = row["Value"]
        # 保持字符串 / 数值两种情形
        param_map[key] = val

    # apply defaults
    merged: dict[str, object] = dict(DEFAULT_PARAMS)
    for k, v in param_map.items():
        if k in merged:
            merged[k] = v
        else:
            log.warning(f"Unknown param '{k}' ignored.")

    for k in DEFAULT_PARAMS:
        if k not in param_map:
            log.warning(f"param '{k}' not found, using default {merged[k]}")

    # distance mode
    dm_str = str(merged["Distance_Mode"]).strip().lower()
    distance_mode = DistanceMode(dm_str)

    data = DataContainer(
        customers=customers,
        hubs=hubs,
        fc=FC(coord=(0.0, 0.0)),
        J=[c.id for c in customers],
        H=[h.id for h in hubs],
        D=list(range(int(merged["max_uav"]))),
        K=list(range(int(merged["max_ugv"]))),
        W=list(range(int(merged["max_trips_per_vehicle"]))),
        cost_H=float(merged["cost_H"]),
        cost_D_inv=float(merged["cost_D_inv"]),
        cost_K_inv=float(merged["cost_K_inv"]),
        cost_D_trav=float(merged["cost_D_trav"]),
        cost_K_trav=float(merged["cost_K_trav"]),
        t_D_set=float(merged["t_D_set"]),
        t_K_set=float(merged["t_K_set"]),
        l_D=float(merged["l_D"]),
        l_K=float(merged["l_K"]),
        UAV_v=float(merged["UAV_v"]),
        UGV_v=float(merged["UGV_v"]),
        e_D=float(merged["e_D"]),
        e_K=float(merged["e_K"]),
        distance_mode=distance_mode,
        hand_D=float(merged["hand_D"]),
        hand_K=float(merged["hand_K"]),
        instance_id=instance_id,
    )
    return data


def build_travel_times(data: DataContainer) -> TravelTimes:
    return TravelTimes.build(
        customer_coords=[c.coord for c in data.customers],
        hub_coords=[h.coord for h in data.hubs],
        fc_coord=data.fc.coord,
        uav_v=data.UAV_v,
        ugv_v=data.UGV_v,
        mode=data.distance_mode,
        hand_D=data.hand_D,
        hand_K=data.hand_K,
    )


# ---------------------------------------------------------------------------
# 预检查：候选 hub 的 UAV 可达性（计划 §0.5, 论文 §5.1 约束）
# ---------------------------------------------------------------------------
def validate_candidate_hubs_for_uav(
    data: DataContainer,
    tt: TravelTimes,
    strict: bool = True,
) -> list[str]:
    """检查所有 hub 对 UAV 单次续航可达.

    Returns: 错误消息列表（若 strict=True 且非空则 raise）.
    """
    errors: list[str] = []
    for h in data.H:
        round_trip = 2 * tt.t_hD[h]
        if round_trip > data.e_D:
            errors.append(
                f"Hub {data.hubs[h].code}: round-trip {round_trip:.2f} > e_D {data.e_D}"
            )
        elif round_trip + data.t_D_set > data.e_D:
            log.warning(
                f"Hub {data.hubs[h].code}: 单次可达, 但 2nd-trip setup 超限 "
                f"(round + set = {round_trip + data.t_D_set:.2f} > e_D)"
            )
    if strict and errors:
        raise ValueError(
            "UAV 不可达的候选 hub 存在:\n  " + "\n  ".join(errors)
        )
    return errors
