"""
SFC 末端 Y 向抗振力 —— 纯控制律。

    F_SFC = −K·e − B0·v − μ·|v|^(n−1)·v

    K   : 固定恢复刚度              [N/m]
    B0  : 基础线性阻尼              [N·s/m]
    μ   : 剪切增稠非线性阻尼系数     [N·(s/m)^n]
    n   : 非线性指数                [-]
    e   : 当前 Y 向偏差              [m]      e = y_sim − y_d
    v   : 当前 Y 向误差速度          [m/s]    v = de/dt

【参数纪律】n 与 μ 是两个彼此独立的显式参数，n > 1。
本模块【不做任何参数推导】——不根据设计点、不根据目标交点、不根据结果反算 n。
μ 的单位 N·(s/m)^n 随 n 变化，所以跨 n 比较 μ 的数值没有意义；
批量实验统一由 sweep.py 用「参考速度 v_ref 处的相对阻尼强度 r」标定：
        μ = r · B0 / v_ref^(n−1)
那只是标定基准，不属于本模块的控制律。

三组对照只差控制律，其余全部相同：
    A0      : F = 0
    Alinear : F = −K·e − B0·v
    ASFC    : F = −K·e − B0·v − μ·|v|^(n−1)·v

本模块是纯函数，不读仿真状态、不读扰动力、不做任何积分/前馈/自适应/额外补偿。
"""

from __future__ import annotations

import math


def sfc_force(e: float, v: float, K: float, B0: float, mu: float, n: float,
              mode: str) -> float:
    """返回末端 Y 向控制力 F_SFC [N]（正号 = 指向 +Y）。"""
    if mode == "A0":
        return 0.0
    if mode == "Alinear":
        return -K * e - B0 * v
    if mode == "ASFC":
        shear = mu * (abs(v) ** (n - 1.0)) * v
        return -K * e - B0 * v - shear
    raise ValueError(f"未知模式：{mode}（应为 A0 / Alinear / ASFC）")


def force_terms(e: float, v: float, K: float, B0: float, mu: float, n: float,
                mode: str) -> dict[str, float]:
    """把 F_SFC 拆成三项，用于验证“非线性项是否真的被激活”。

    返回 {"F_K", "F_B", "F_shear", "F_total"}，单位 N，均为已带符号的实际贡献。
    """
    F_K = -K * e
    F_B = -B0 * v
    F_shear = -mu * (abs(v) ** (n - 1.0)) * v
    if mode == "A0":
        F_K = F_B = F_shear = 0.0
    elif mode == "Alinear":
        F_shear = 0.0
    elif mode != "ASFC":
        raise ValueError(f"未知模式：{mode}")
    return {"F_K": F_K, "F_B": F_B, "F_shear": F_shear,
            "F_total": F_K + F_B + F_shear}


def shear_ratio(v: float, B0: float, mu: float, n: float) -> float:
    """剪切项 / 线性阻尼项 的幅值比（|v| 很小时返回 0）。用于标定校核与事后验证。"""
    lin = abs(B0 * v)
    shear = abs(mu * (abs(v) ** (n - 1.0)) * v)
    if lin < 1e-15:
        return 0.0
    return shear / lin


def mu_from_r(r: float, B0: float, n: float, v_ref: float) -> float:
    """由“参考速度处的相对阻尼强度 r”反解 μ：  μ = r·B0 / v_ref^(n−1)。

    约定：在 |v| = v_ref 处，μ·v_ref^(n−1) = r·B0，即剪切阻尼是线性阻尼的 r 倍。
    这只是为了让不同 n 之间可比，不是控制律的一部分。
    """
    if v_ref <= 0.0:
        raise ValueError("v_ref 必须为正")
    return float(r) * float(B0) / (float(v_ref) ** (float(n) - 1.0))


def power_terms(v: float, B0: float, mu: float, n: float) -> dict[str, float]:
    """瞬时功率（耗散正确性检查用）。单位 W。

        P_linear = F_linear_damping · v = −B0·v²            ≤ 0 恒成立
        P_shear  = F_shear          · v = −μ|v|^(n−1)·v²    ≤ 0 恒成立

    除数值舍入外不应出现正值；大量正值说明符号实现有误，该组不能算有效实验。
    """
    F_lin = -B0 * v
    F_shr = -mu * (abs(v) ** (n - 1.0)) * v
    return {"F_linear_damping": F_lin, "F_shear": F_shr,
            "P_linear": F_lin * v, "P_shear": F_shr * v}
