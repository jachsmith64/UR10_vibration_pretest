"""
SFC 控制器 —— 把「状态 → e,v」「F_SFC → τ_SFC」两步串起来。

本模块只做两件事，且两件事分别被隔离在可替换的适配器里：

  ① measurement_adapter :  UR10 状态 → (e, v)
       本阶段用 IdealStateMeasurement（直接读 MuJoCo 连续状态）。
       以后换 132 Hz 相机时，只需提供同样接口的新适配器，控制器本体不动。

  ② execution_adapter   :  F_SFC → τ_SFC
       本阶段用 JacobianTorqueExecution（τ = J^T·F）。
       以后换真实 UR10 硬件接口时，只需替换该适配器。

硬约束：本模块不得 import disturbance，不得读取 Fd。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

import sfc


# ============================================================ ① 测量适配器
class MeasurementAdapter(Protocol):
    """UR10 状态 → 误差 e [m] 与误差速度 v [m/s]。"""

    def measure(self, env: Any, y_d: float, y_d_dot: float = 0.0) -> tuple[float, float]:
        ...


class IdealStateMeasurement:
    """理想测量：直接用 MuJoCo 当前连续状态，不加采样率/噪声/掉帧/延迟/滤波。

        e = y_sim − y_d
        v = de/dt = J_y·q̇ − ẏ_d      （雅可比给出精确末端速度，不做数值差分）
    """

    name = "ideal_state"

    def measure(self, env: Any, y_d: float, y_d_dot: float = 0.0) -> tuple[float, float]:
        e = env.tcp_y() - float(y_d)
        v = env.tcp_y_velocity() - float(y_d_dot)
        return float(e), float(v)


# ============================================================ ② 执行适配器
class ExecutionAdapter(Protocol):
    """末端 Y 向力 → 关节力矩 (nv,)。"""

    def execute(self, env: Any, f_y: float) -> np.ndarray:
        ...


class JacobianTorqueExecution:
    """τ_SFC = J^T · F_task，F_task = [0, F_y, 0]。

    只用到雅可比第 1 行：τ = J_y^T · F_y。
    """

    name = "jacobian_torque"

    def execute(self, env: Any, f_y: float) -> np.ndarray:
        J = env.jacp()
        f_task = np.array([0.0, float(f_y), 0.0])
        return J.T @ f_task


# ============================================================ 控制器
class SFCController:
    def __init__(self, env, params: dict[str, float], mode: str,
                 measurement: MeasurementAdapter | None = None,
                 execution: ExecutionAdapter | None = None):
        if mode not in ("A0", "Alinear", "ASFC"):
            raise ValueError(f"未知模式：{mode}")
        if mode == "ASFC":
            # n 与 μ 是彼此独立的显式参数；这里只做合法性检查，不做任何推导。
            if not float(params["n"]) > 1.0:
                raise ValueError(f"SFC 要求 n > 1，收到 n={params['n']}")
            if float(params["mu"]) < 0.0:
                raise ValueError(f"μ 不能为负，收到 μ={params['mu']}")
        self.env = env
        self.params = dict(params)
        self.mode = mode
        self.measurement = measurement or IdealStateMeasurement()
        self.execution = execution or JacobianTorqueExecution()

    def step(self, y_d: float, y_d_dot: float = 0.0) -> dict[str, Any]:
        """采一次状态 → 算 F_SFC → 转成 τ 并施加。返回本步记录。"""
        e, v = self.measurement.measure(self.env, y_d, y_d_dot)
        terms = sfc.force_terms(e, v, self.params["K"], self.params["B0"],
                                self.params["mu"], self.params["n"], self.mode)
        f_sfc = terms["F_total"]
        tau = self.execution.execute(self.env, f_sfc)
        self.env.apply_joint_torque(tau)
        return {
            "e": e, "v": v, "F_sfc": f_sfc,
            "F_K": terms["F_K"], "F_B": terms["F_B"], "F_shear": terms["F_shear"],
            "tau": tau,
        }
