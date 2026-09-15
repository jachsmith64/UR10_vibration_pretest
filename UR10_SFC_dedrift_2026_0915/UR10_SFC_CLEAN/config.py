"""
UR10 末端 Y 向抗振仿真 —— 全局参数与路径（唯一参数来源）。

本工程只研究一件事：
    冻结等效扰动力 Fd(t) 作用于 UR10 末端 TCP 的 Y 方向时，
    SFC 本身能否降低末端 Y 向振动；非线性剪切增稠项相比普通线性阻抗是否有额外贡献。

三条支路严格独立（见 README §架构隔离）：
    ① 扰动： 实验数据 → Fd(t) → TCP 点 Y 向外力 → UR10   （disturbance.py）
    ② SFC：  UR10 状态 → e,v → F_SFC                       （sfc.py / controller.py）
    ③ 执行：  F_SFC → J_tcp^T → τ_SFC → UR10               （controller.py）

任何模块不得跨层读取：SFC 不读 Fd；disturbance 不读 SFC 输出；
正式运行中 Fd 不得根据当前 e_sim 重算。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

# ============================================================ 路径
PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
ASSET_DIR: Final[Path] = PROJECT_DIR / "assets_ur10e"
MODEL_XML_PATH: Final[Path] = ASSET_DIR / "scene.xml"
DATA_CSV: Final[Path] = PROJECT_DIR / "data" / "P05R01" / "D1_时序.csv"
OUTPUT_ROOT: Final[Path] = PROJECT_DIR / "outputs"
REFERENCE_DIR: Final[Path] = PROJECT_DIR / "reference"   # 旧工程对照资料，仅参考，不被代码依赖

# 冻结数据包目录。本轮把「去慢偏移后的净振动」版本作为正式包；
# 上一轮「未去慢偏移」的包另存一份，仅用于对照与画辅图，正式运行不读它。
FD_PACKET_DIR: Final[Path] = OUTPUT_ROOT / "fd_packet"
FD_PACKET_RAW_DIR: Final[Path] = OUTPUT_ROOT / "fd_packet_raw"
PLOT_DIR: Final[Path] = OUTPUT_ROOT / "figs"

# ============================================================ 机器人
# 初始构型：肩部回转=0 使臂沿 +X 展开；此时 TCP ≈ (-0.691, -0.174, 0.694) m
INIT_Q: Final[list[float]] = [0.0, -math.pi / 2.0, math.pi / 2.0,
                              -math.pi / 2.0, -math.pi / 2.0, 0.0]
TCP_SITE_NAME: Final[str] = "attachment_site"     # 末端 TCP site（Fd 的作用点）
FORCE_BODY_NAME: Final[str] = "wrist_3_link"      # TCP site 所属 body（mj_applyFT 需要）
PHYSICS_HZ: Final[float] = 1000.0                 # 物理步长 1 ms
# 模型内置关节位置伺服：τ = 5000·(ctrl − q) − 500·q̇（见 ur10e.xml 的 gainprm/biasprm）
SERVO_KP: Final[float] = 5000.0
SERVO_KV: Final[float] = 500.0
# Y 轴标称目标（= 初始构型下 TCP 的 Y），仿真中保持恒定
Y_NOMINAL: Final[float] = -0.174
Z_HOLD: Final[float] = 0.694                      # 标称运动保持的 Z 高度

# ============================================================ 理想慢偏移 b(t)（本轮新增）
# e_raw = b(t) + e_vib：先把方向性慢偏移扣掉，只对净振动做扰动重建与 SFC 验证。
# 构造方式：稳态段「沿程位置分箱中位数 + LOWESS 鲁棒平滑」→ b±(x)；
#           启动/转向/停止段用三次 Hermite 一阶连续桥接，不追随瞬态尖峰。
# 分箱宽度的选取依据（实测谱分离，不是拍脑袋）：
#   正向稳态段 慢瓣(0–1 Hz, 波长≈7.7 mm) 占 33.7%，振动簇(2.5–10 Hz, 波长 0.70–1.28 mm) 占 53.5%。
#   若 bin 取得比振动波长还小（曾用 0.5 mm），箱内不足一个振动周期，
#   中位数会被振动相位拉动，出现 ±20 µm 的假慢偏移。
#   实测 bin=1.0 / span=2.5 mm 时：慢带残留 36–40%，振动带残留 99.9%（两个方向都成立），
#   是候选里分离得最干净的一组。
SLOW_BIN_MM: Final[float] = 1.0           # 分箱宽度（沿程 mm）
SLOW_BIN_MIN_SAMPLES: Final[int] = 8      # 箱内样本数低于此则该箱弃用
SLOW_SPAN_MM: Final[float] = 2.5          # LOWESS 带宽（沿程 mm）：宽度大于它的慢变化才被跟随
SLOW_SPAN_GROWTH: Final[float] = 1.5      # 验收未过时逐次加宽的倍率
SLOW_REFINE_PASSES: Final[int] = 2        # 最多加宽几次（总尝试 = 该值 + 1）
# 人工验收阈值（见 slowoffset.acceptance）
SLOW_CENTER_ABS_UM: Final[float] = 2.0    # 稳态段 |中位数| 的绝对上限
SLOW_CENTER_REL: Final[float] = 0.15      # 稳态段 |中位数| / 该段 e_vib RMS 的相对上限
# 瞬态保留判据用【数学硬下界】而非拍的峰值比例：
#   ptp(e_vib) = ptp(e_raw − b) ≥ ptp(e_raw) − ptp(b)   （max(a−b) ≥ max a − max b，min 同理）
# 即「b 至多吃掉自己那么大的峰谷」。这是恒成立的等式约束，不引入任何主观容差，
# 只要它被违反就说明 b 的构造本身不自洽。SLOW_PTP_TOL_UM 仅留给浮点与端点重合。
SLOW_PTP_TOL_UM: Final[float] = 1.0
# 过渡段内 b 的变化 ≤ 该段两端点差 |Δb| 的多少倍（超出即认为桥接/外推在乱动）
SLOW_B_OVERSHOOT_MAX: Final[float] = 1.5

# 重组来源：vib = 只对去慢偏移后的净振动做 wfit（本轮正式采用）；raw = 上一轮口径
RECON_SOURCE_DEFAULT: Final[str] = "vib"

# ============================================================ 扰动重建
GRID_DT_S: Final[float] = 0.001          # 重建/反演统一 1 ms 网格
TEMPLATE_DURATION_S: Final[float] = 60.0  # 目标模板时长（稳态窗复制到覆盖该时长）
SEAM_SMOOTH_S: Final[float] = 0.10        # 拼接缝 sin² 平滑半窗
# 切窗阈值（P05R01 数据集专用；任一条件不命中即报错，不静默兜底）
CUT_IA_ALONG_MM: Final[float] = 3.2
CUT_IB_ALONG_MM: Final[float] = 11.0
CUT_IC_SEARCH_S: Final[float] = 1.3
CUT_IC_CROSS_UM: Final[float] = -45.0
CUT_ID_SPEED_MM_S: Final[float] = -3.5

# Plant 频响标定：sum-of-sines 探针（在 INIT_Q 构型下测 “TCP 点 Y 外力 → TCP Y 位移”）
# fmax 必须【不低于反演上限】，否则 plant.at() 会在 35 Hz 处被 np.interp 夹住，
# 等于拿一个错的增益去反演高频段。
PROBE_DURATION_S: Final[float] = 60.0
PROBE_FMAX_HZ: Final[float] = 50.0
PROBE_AMP_N: Final[float] = 0.05
PROBE_SETTLE_STEPS: Final[int] = 300      # 丢弃初始伺服瞬态
PROBE_DC_SETTLE_STEPS: Final[int] = 500

# 带限反演：30 Hz 以下完整保留，30→45 Hz 余弦渐减，45 Hz 以上截断。
#
# 【本轮为何把带宽从 11/16 Hz 拓宽】上一轮的口径是 e_raw（慢漂移占主导），
# 11/16 Hz 够用。本轮目标换成去慢偏移后的净振动后，实测其谱能量分布为
#     0–2.5 Hz  8.82 µm (55%)   2.5–5 Hz  8.81 µm   5–11 Hz  5.70 µm
#     11–16 Hz  3.40 µm         16–45 Hz  3.14 µm   ← 占目标总能量约 20%
# 且 16–45 Hz 段并非插值噪声：谱形在 24–30 Hz 有明确凹陷、在 34–36 Hz 有明显
# 共振峰（原始记录中位采样 7.56 ms ≈ 132 Hz，Nyquist ≈ 66 Hz，35 Hz 远在其下，
# 是真实振动而非混叠）。旧带宽把这 20% 整体切掉，残差被硬性钉在 ~19.6%，
# 20% 的验收门槛在结构上无法达到。故拓宽到 plant 可标定的范围。
INV_F_PASS_HZ: Final[float] = 30.0
INV_F_STOP_HZ: Final[float] = 45.0

# 残差迭代修正（吸收 Plant 单构型标定 vs 沿程构型漂移的误差）
REFINE_ITERS: Final[int] = 5              # 上限 5 轮
REFINE_LR: Final[float] = 0.7
# 收敛判据用【相对】形式：残差 RMS / 目标 RMS ≤ 20%（与验收门槛同一把尺子）
REFINE_STOP_RESIDUAL_FRAC: Final[float] = 0.20

# 扰动环境验收门槛（重建完成后判定“能否进入后续控制测试”）
GATE_CORR_MIN: Final[float] = 0.98
GATE_RESIDUAL_FRAC_MAX: Final[float] = 0.20

# ============================================================ SFC 参数
# 公式： F_SFC = −K·e − B0·v − μ·|v|^(n−1)·v
#
# K, B0 : 线性阻抗基础，本轮固定，不参与标定。
# n, μ  : 剪切增稠项的【两个独立参数】，显式配置，n > 1。
#         本轮【不再由任何公式自动推导 n】。
#
# μ 的单位是 N·(s/m)^n，随 n 变化 —— 所以“不同 n 之间比较同一个 μ 的数值”没有意义。
# 本轮改用「参考速度 v_ref 处的相对阻尼强度 r」来标定 μ（见 sweep.py）：
#         μ = r · B0 / v_ref^(n−1)
#   r = 1 表示“在 v_ref 处，剪切阻尼与线性阻尼等强”。这样不同 n 之间才可比。
#   r 只是标定用的强度基准，不属于控制算法本身。
SFC_K: Final[float] = 5000.0              # N/m
SFC_B0: Final[float] = 400.0              # N·s/m
SFC_N_DEFAULT: Final[float] = 3.0         # 默认 ASFC 的 n（显式指定）
SFC_R_DEFAULT: Final[float] = 1.0         # 默认 ASFC 的强度基准 r

# ---- 本轮（V1.2）正式对照：不再做几十组全扫描，只跑有代表性的几组 ----
# 每组 (n, r)；μ 由 μ = r·B0 / v_ref^(n−1) 标定，v_ref 由本轮 A0 重新定出。
#   (3.0, 1.0) 默认组、 (2.0, 2.0) 中等强度、 (1.5, 4.0) 上一轮效果最好的强代表
ROUND_ASFC: Final[tuple[tuple[float, float], ...]] = ((3.0, 1.0), (2.0, 2.0), (1.5, 4.0))

# ---- 「回正阻碍」局部复查（§9.2）----
# 上一轮肉眼觉得「约 X≈2.5 mm 处 ASFC 妨碍回正」，本轮在去慢偏移后专门复查这一段。
RETURN_LOCAL_ALONG_MM: Final[float] = 2.5    # 复查中心（沿程 mm）
RETURN_LOCAL_HALF_MM: Final[float] = 1.5     # 复查半窗（沿程 mm）

# ---- 第一轮主参数扫描 ----
SWEEP_N: Final[tuple[float, ...]] = (1.5, 2.0, 2.5, 3.0, 4.0)
SWEEP_R: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)

# ---- 条件分支：若第一轮非线性仍未激活，补做少量极端诊断 ----
EXTREME_N: Final[tuple[float, ...]] = (2.0, 3.0)
EXTREME_R: Final[tuple[float, ...]] = (8.0, 16.0)
# 触发条件：第一轮 ASFC 与 Alinear 几乎完全重合，且实测
#           F_shear_rms / F_linear_rms < 该阈值
INACTIVE_RATIO_THRESHOLD: Final[float] = 0.05

# ---- 第二轮：3 个代表组的线性基础参数局部扩展 ----
K_SCALES: Final[tuple[float, ...]] = (0.5, 2.0)     # 外加基准 K×1.0
B0_SCALES: Final[tuple[float, ...]] = (0.5, 2.0)    # 外加基准 B0×1.0

# ---- v_ref 定义（用于把 r 换算成 μ）----
V_REF_TRIM_S: Final[float] = 3.0          # 去掉首尾各 3 s 的启动/停止瞬态
V_REF_PERCENTILE: Final[float] = 75.0     # v_ref = percentile(|v_A0|, 75%)
V_REF_MIN_M_S: Final[float] = 1e-6        # 低于此认为“异常接近 0”，改用 RMS 并记录原因

# ============================================================ 数值安全（只做检测与中止，不改控制律）
# 本工程【不施加任何力限幅】，因此 “是否限幅” 恒为否；
# 用 F_SFC_peak / τ 范数 / 越限样本比例作为等价的安全指标。
ABORT_QVEL_MAX: Final[float] = 100.0      # rad/s，任一关节速度超此值判定发散
ABORT_E_ABS_MAX: Final[float] = 0.05      # m，末端偏差超此值判定跑飞
F_DIAG_REF_N: Final[float] = 10.0         # 仅诊断用参考力，用于统计越限样本比例
P_TOL_W: Final[float] = 1e-12             # 耗散功率正值的容差（W）

# ============================================================ 运行 / 分析
MODES: Final[tuple[str, ...]] = ("A0", "Alinear", "ASFC")   # 三组对照，仅控制律不同
MODE_LABEL: Final[dict[str, str]] = {
    "A0": "无抗振控制 F = 0",
    "Alinear": "普通线性阻抗 F = −K·e − B0·v",
    "ASFC": "完整剪切增稠 F = −K·e − B0·v − μ|v|^(n−1)v",
}
# 报告用频段划分（源自预实验谱 profile：慢瓣主峰 ~0.5 Hz，振动带主簇 2.5–14 Hz）
SLOW_BAND_HZ: Final[tuple[float, float]] = (0.0, 2.5)
VIB_BAND_HZ: Final[tuple[float, float]] = (2.5, 45.0)
WELCH_WIN_S: Final[float] = 2.0

# 完整时序落盘的抽稀步长。物理步长 1 ms，取 5 → 200 Hz 采样，
# 远高于关心的 45 Hz 振动带，同时把单组 CSV 压到 MB 量级。
TRACE_STRIDE: Final[int] = 5

# 文件名
FD_FILENAME: Final[str] = "fd_frozen.csv"
FD_META_FILENAME: Final[str] = "fd_meta.json"
SWEEP_CSV: Final[str] = "sweep_summary.csv"


def physics_dt() -> float:
    return 1.0 / PHYSICS_HZ


def snapshot() -> dict:
    """把本模块里所有 Final 常量导出成可序列化字典，供 config_snapshot.json 存档。"""
    out: dict = {}
    for key, val in sorted(globals().items()):
        if key.startswith("_") or key in ("math", "Path", "Final", "annotations"):
            continue
        if isinstance(val, Path):
            out[key] = str(val)
        elif isinstance(val, (int, float, str, bool)) or val is None:
            out[key] = val
        elif isinstance(val, (list, tuple, dict)):
            out[key] = list(val) if isinstance(val, tuple) else val
    return out
