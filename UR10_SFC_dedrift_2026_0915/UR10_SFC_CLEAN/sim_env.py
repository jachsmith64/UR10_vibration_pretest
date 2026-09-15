"""
MuJoCo / UR10e 环境薄封装。

职责：
  - 加载模型、复位、按物理步长推进
  - 读取末端状态（TCP 位置 / 雅可比 / 由雅可比得到的精确末端速度）
  - 施加两类互不干扰的广义力：
        * 等效扰动力 Fd  → 作用在【TCP site 所在空间点】的世界系 Y 向 Cartesian 力，
                          经 mj_applyFT 折算成广义力，存入内部缓冲 _qfrc_dist
        * SFC 关节力矩 τ_SFC → 内部缓冲 _tau_sfc
    两者相加后写入 data.qfrc_applied（广义力可叠加），互不覆盖。
  - 名义运动：走模型内置关节位置伺服（data.ctrl = q_d），与上面的广义力相加而非替代

【关于 Fd 的作用点】
  旧实现直接写 xfrc_applied[wrist_3_link] = [0, Fd, 0]，那是把力作用在刚体质心上，
  与 SFC 使用的 TCP 雅可比不是同一个物理点（少了 TCP 相对质心偏置产生的 r×F 力矩）。
  现在改用 mujoco.mj_applyFT(force, torque=0, point=TCP, body=wrist_3_link, target)，
  它按 mj_jac(point) 折算： qfrc += J_point^T · force + J_rot^T · torque，
  等价于“力 + 偏置力矩”，与 J_tcp^T @ [0,F,0] 数值一致（见 check_tcp_force_generalized）。

本模块不含任何控制律、不含扰动重建逻辑。
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

import config


class SimEnv:
    def __init__(self, xml_path=None):
        self.model = mujoco.MjModel.from_xml_path(
            str(xml_path) if xml_path is not None else str(config.MODEL_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.physics_dt()

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, config.TCP_SITE_NAME)
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, config.FORCE_BODY_NAME)
        if self.site_id < 0:
            raise ValueError(f"site 不存在：{config.TCP_SITE_NAME}")
        if self.body_id < 0:
            raise ValueError(f"body 不存在：{config.FORCE_BODY_NAME}")

        self.nv = int(self.model.nv)
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

        # 两条广义力支路的独立缓冲：互不覆盖，最后相加
        self._tau_sfc = np.zeros(self.nv)
        self._qfrc_dist = np.zeros(self.nv)
        self._force = np.zeros(3)
        self._torque = np.zeros(3)

    # ---------------------------------------------------------------- 生命周期
    def reset(self, qpos: list[float] | np.ndarray) -> None:
        """置位到指定关节角并静止；ctrl 同步为目标，让伺服一开始就托住重力。"""
        d = self.data
        d.qpos[:] = np.asarray(qpos, dtype=float)
        d.qvel[:] = 0.0
        d.ctrl[:] = np.asarray(qpos, dtype=float)
        d.xfrc_applied[:] = 0.0
        self._tau_sfc[:] = 0.0
        self._qfrc_dist[:] = 0.0
        self._sync()
        mujoco.mj_forward(self.model, d)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """只刷新派生量（位置/雅可比），不推进时间。"""
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- 状态读取
    def tcp_pos(self) -> np.ndarray:
        """TCP 世界坐标 (3,) m。"""
        return self.data.site_xpos[self.site_id].copy()

    def tcp_y(self) -> float:
        return float(self.data.site_xpos[self.site_id][1])

    def jacp(self) -> np.ndarray:
        """TCP 平动雅可比 (3, nv)，世界系。"""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return self._jacp.copy()

    def tcp_y_velocity(self) -> float:
        """TCP Y 向精确速度 (m/s)：v_y = J_y · q̇。用连续状态直接得到，不做差分。"""
        return float(self.jacp()[1] @ self.data.qvel)

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    # ---------------------------------------------------------------- 作用量
    def _sync(self) -> None:
        """两条支路的广义力相加后写入 qfrc_applied（叠加，互不覆盖）。"""
        np.add(self._tau_sfc, self._qfrc_dist, out=self.data.qfrc_applied)

    def set_ctrl(self, q_cmd: np.ndarray) -> None:
        """名义运动控制：交给模型内置关节位置伺服。"""
        self.data.ctrl[:] = np.asarray(q_cmd, dtype=float)

    def apply_disturbance_force_y(self, f_y: float) -> None:
        """冻结等效扰动力 Fd(t)：世界系 Y 向力，作用在【TCP site 所在空间点】。

        用 mj_applyFT 折算成广义力（torque=0，但作用点取 TCP，因此仍会带上
        TCP 相对刚体质心偏置产生的 r×F 等效力矩）。这是相对于旧实现的关键修正。
        """
        self._qfrc_dist[:] = 0.0
        self._force[0] = 0.0
        self._force[1] = float(f_y)
        self._force[2] = 0.0
        mujoco.mj_applyFT(self.model, self.data, self._force, self._torque,
                          self.tcp_pos(), self.body_id, self._qfrc_dist)
        self._sync()

    def apply_joint_torque(self, tau: np.ndarray) -> None:
        """SFC 附加关节力矩 τ_SFC：广义力注入，与位置伺服的输出相加。"""
        self._tau_sfc[:] = np.asarray(tau, dtype=float)
        self._sync()

    def clear_external(self) -> None:
        self._tau_sfc[:] = 0.0
        self._qfrc_dist[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self._sync()

    # ---------------------------------------------------------------- 数值自检
    def check_tcp_force_generalized(self, f_test: float = 1.0,
                                    tol_rel: float = 1e-9) -> dict[str, Any]:
        """验证“TCP 点施力”折算出的广义力 == J_tcp^T · [0, F, 0]。

        这是本轮修正 Fd 作用点后的强制数值检查：若两者不一致，说明力并没有真正
        作用在 TCP 上（例如仍落在刚体质心）。
        """
        self.forward()
        J = self.jacp()
        q_jac = J.T @ np.array([0.0, float(f_test), 0.0])
        self.apply_disturbance_force_y(f_test)
        q_ft = self._qfrc_dist.copy()
        err = float(np.max(np.abs(q_ft - q_jac)))
        ref = float(np.max(np.abs(q_jac)))
        return {
            "f_test_N": float(f_test),
            "max_abs_err": err,
            "ref_max_abs": ref,
            "rel_err": err / ref if ref > 1e-15 else float("inf"),
            "passed": bool(err <= tol_rel * max(ref, 1.0)),
            "qfrc_from_applyFT": [float(x) for x in q_ft],
            "qfrc_from_Jtcp": [float(x) for x in q_jac],
        }


# -------------------------------------------------------------------- 名义运动 IK
def _ik_dls(env: SimEnv, target_xyz: np.ndarray, q_seed: np.ndarray,
            lam: float = 1e-4, iters: int = 60, tol: float = 1e-8) -> np.ndarray:
    """阻尼最小二乘 IK：只约束 TCP 三维位置（姿态留自由），种子取上一时刻解以保证连续。

    tol 取 10 nm：远小于本工程关心的 µm 级误差，同时让热启动能少迭代几次（60k 点连续求解）。
    """
    q = np.asarray(q_seed, dtype=float).copy()
    for _ in range(iters):
        env.data.qpos[:] = q
        env.forward()
        err = np.asarray(target_xyz, dtype=float) - env.tcp_pos()
        if float(np.linalg.norm(err)) < tol:
            break
        J = env.jacp()
        A = J @ J.T + (lam ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        q = q + dq
    env.data.qpos[:] = q
    env.forward()
    return q


def build_nominal_joint_trajectory(along_mm: np.ndarray, y_target: float,
                                   z_hold: float, q_init=None) -> np.ndarray:
    """把沿程 schedule 转成名义关节轨迹 q_d(t)，形状 (n, nv)。

    x_d(t) = x_init + along_mm(t)/1000，y/z 保持标称值。
    逐点用 DLS IK 求解，并用上一点作种子保证关节连续、不跳支。
    """
    env = SimEnv()
    env.reset(config.INIT_Q if q_init is None else q_init)
    x0 = float(env.tcp_pos()[0])

    n = len(along_mm)
    q_traj = np.zeros((n, env.nv))
    q_seed = np.asarray(config.INIT_Q if q_init is None else q_init, dtype=float)
    for i in range(n):
        target = np.array([x0 + float(along_mm[i]) / 1000.0, y_target, z_hold])
        q_seed = _ik_dls(env, target, q_seed)
        q_traj[i] = q_seed
    return q_traj
