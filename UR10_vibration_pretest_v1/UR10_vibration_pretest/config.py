"""
UR10 末端振动预实验：统一配置文件。

你平时最常修改的就是这个文件。其余文件负责“怎么做”，这里负责“这次做什么、用什么参数做”。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


# =============================================================================
# 0. 项目路径
# =============================================================================

# 所有相对路径都以 config.py 所在文件夹为基准，而不是以终端当前目录为基准。
# 这样即使你从 VS Code 的其他目录启动 main.py，输入和输出位置也不会突然改变。
PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent

# 输入图片、输入视频和运行结果分别放在不同文件夹，避免原始数据与处理结果混在一起。
# 程序会自动创建输出文件夹，但不会替你伪造真实的输入图片或视频。
IMAGE_FOLDER = PROJECT_DIR / "input_images"
VIDEO_PATH = PROJECT_DIR / "input_video" / "test.mp4"
OUTPUT_ROOT = PROJECT_DIR / "outputs"


# =============================================================================
# 1. 总运行模式：一次只允许选择一条路线
# =============================================================================

# 建议当前先用 vision_test。拿到 UR10 前不要把模式改成 robot_test 或 experiment。
RUN_MODE = "vision_test"

# 允许的五种模式会在 validate_config() 中统一检查，拼错一个字母也会在启动阶段报清楚。
VALID_RUN_MODES: Final[set[str]] = {
    "vision_test",    # 只测图像读取、标志识别和位移计算。
    "robot_dry_run",  # 只生成并检查轨迹，不导入 UR 库，也不连接真机。
    "robot_test",     # 连接 UR，默认只读取状态；必须再次开关才允许低速运动。
    "experiment",     # 相机、UR 和记录进程共同运行的正式预实验。
    "analyze",        # 读取已有 TXT/JSONL 记录并生成振动分析结果。
}


# =============================================================================
# 2. 视觉输入来源与算法选择
# =============================================================================

# image_folder 和 video 不需要海康 MVS SDK；hik_camera 只在真正选择它时才导入 SDK。
VISION_SOURCE = "image_folder"
VALID_VISION_SOURCES: Final[set[str]] = {"image_folder", "video", "hik_camera"}

# compare 会让圆点法和棋盘格法同时计算，适合前期比较稳定性；正式实验后再决定主方法。
VISION_METHOD = "compare"
VALID_VISION_METHODS: Final[set[str]] = {"circles", "checkerboard", "compare"}

# 图像文件按“自然文件名顺序”读取，例如 frame2 会排在 frame10 前面。
# 支持这些常见扩展名，其他格式不会被程序误当作图片打开。
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"
)

# 离线图片没有可靠的相机时间戳，因此用这个帧率把帧号换算成相对时间。
# 如果图片是从 60 fps 视频逐帧导出的，这里就填 60.0。
IMAGE_FOLDER_FPS = 60.0

# 是否弹出 OpenCV 预览窗口。服务器或无桌面环境应设为 False，Windows 本地调试可设为 True。
SHOW_PREVIEW = True

# 按 q 或 Esc 可提前结束预览；不显示预览时，程序会自动处理完全部输入。
PREVIEW_MAX_WIDTH = 1400

# 调试图会画出角点、圆心、位移和质量指标；每帧都保存会很占硬盘，所以允许间隔保存。
SAVE_DEBUG_IMAGE = True
DEBUG_IMAGE_EVERY_N_FRAMES = 10
MAX_DEBUG_IMAGES = 300

# 完整实验中不建议保存大量调试图，因为磁盘写入可能干扰实时取图。
# 只有明确确认电脑性能足够时，才把这个额外开关改成 True。
SAVE_DEBUG_IMAGE_DURING_EXPERIMENT = False


# =============================================================================
# 3. 图像预处理
# =============================================================================

# ROI 依次为 x, y, width, height；None 表示处理整幅图像。
# 正式固定相机后建议裁出测量纸附近区域，这会减少误识别并提高处理速度。
VISION_ROI: tuple[int, int, int, int] | None = None

# 轻微高斯滤波可减小传感器噪点；必须使用正奇数，1 表示不滤波。
# 振动图像可能已有运动模糊，因此默认只使用很小的 3×3 核。
GAUSSIAN_BLUR_KERNEL = 3

# 若已经完成相机标定，可填入 3×3 内参和 5/8 个畸变系数；None 表示暂不校正。
# 这两个参数必须同时填写或同时保持 None，配置检查会阻止只填一半。
CAMERA_MATRIX: list[list[float]] | None = None
DISTORTION_COEFFICIENTS: list[float] | None = None

# 清晰度使用拉普拉斯方差衡量。它只用于报警和质量记录，不会因为低于阈值就擅自删掉一帧。
BLUR_WARNING_THRESHOLD = 60.0

# 过暗或过曝比例用于区分“算法找不到图案”和“图像本身曝光错误”。
DARK_PIXEL_THRESHOLD = 15
BRIGHT_PIXEL_THRESHOLD = 245


# =============================================================================
# 4. 棋盘格参数
# =============================================================================

# OpenCV 要的是“内角点数量”而不是黑白方格数量，顺序为列数、行数。
# 例如这里的 (7, 5) 对应实际打印 8 列×6 行方格。
CHECKERBOARD_INNER_CORNERS = (7, 5)

# 单个黑白方格边长，单位 mm。它负责把像素位移换算成纸面内的毫米位移。
CHECKER_SQUARE_MM = 4.0

# 亚像素迭代次数和停止精度。数值更严不一定更准，图像清晰度仍是主要限制。
CHECKER_SUBPIX_MAX_ITER = 40
CHECKER_SUBPIX_EPS = 0.001

# 残差超过该值时质量分会明显下降，但仍保留原始结果供你检查。
CHECKER_RESIDUAL_WARNING_PX = 0.8


# =============================================================================
# 5. 圆点参数
# =============================================================================

# 预期圆点数量和最低有效数量。跟踪中只要仍有 MIN_VALID_CIRCLES 个对应点就能估计刚体运动。
CIRCLE_EXPECTED_COUNT = 7
MIN_VALID_CIRCLES = 5

# 圆点轮廓面积是像素面积，因此与工作距离和分辨率有关；首次实拍后要根据调试图调整。
CIRCLE_MIN_AREA_PX = 25.0
CIRCLE_MAX_AREA_PX = 12000.0

# 圆度越接近 1 越像圆；透视后圆会变椭圆，所以同时允许一定长短轴比例变化。
CIRCLE_MIN_CIRCULARITY = 0.68
CIRCLE_MIN_AXIS_RATIO = 0.52

# 当前帧圆心通过“距上一帧预测位置最近”保持身份，超过此距离的匹配会被拒绝。
# 该值应大于相邻两帧最大位移，但不能大到允许不同圆点互相串号。
CIRCLE_MAX_MATCH_DISTANCE_PX = 45.0

# 圆点纸的理论中心坐标，单位 mm。它同时用于生成 SVG 测量纸和估计圆点法的毫米/像素比例。
# 布局故意不对称，目的是减少旋转后“看起来仍完全一样”的身份歧义。
CIRCLE_LAYOUT_MM: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 0.0),
    (11.0, 2.0),
    (24.0, 0.5),
    (4.0, 12.0),
    (17.0, 15.0),
    (29.0, 10.5),
    (9.0, 26.0),
)

# 普通圆点直径和方向锚点直径。锚点略大，只用于人眼和后续扩展辨向，不参与单点位移判断。
CIRCLE_DIAMETER_MM = 3.0
CIRCLE_ANCHOR_DIAMETER_MM = 4.5

# 圆点整体刚体拟合的像素残差阈值。残差大通常意味着串点、局部翘曲或轮廓识别错误。
CIRCLE_RESIDUAL_WARNING_PX = 1.0


# =============================================================================
# 6. 可打印复合测量纸
# =============================================================================

# vision_test 启动时会生成一份矢量 SVG；按 100% 比例打印，不能选择“适应页面”。
GENERATE_MARKER_SHEET_ON_VISION_TEST = True
MARKER_SHEET_PATH = OUTPUT_ROOT / "marker_sheet.svg"

# 测量纸尺寸单位为 mm，适合先平面打印后再裁剪。
# 如果要贴在圆柱面上，应尽量减小弯曲并把被测圆点放在相机正对区域。
MARKER_SHEET_WIDTH_MM = 82.0
MARKER_SHEET_HEIGHT_MM = 42.0
MARKER_MARGIN_MM = 4.0


# =============================================================================
# 7. 海康相机参数
# =============================================================================

# 空字符串表示使用枚举到的第一台设备；正式实验建议填序列号，避免多相机时连错。
HIK_CAMERA_SERIAL = ""

# MVS 安装后若 Python 包不在系统路径，可把官方 Samples/Python/MvImport 路径填在这里。
# 程序只会把明确填写的目录加入 sys.path，不会扫描整块硬盘。
HIK_MVS_IMPORT_PATH: str | None = None

# 曝光单位通常为微秒，增益单位由相机节点定义；None 表示不主动改相机当前配置。
# 首次连接先在 MVS 客户端里确认可用范围，再把稳定参数抄到这里。
HIK_EXPOSURE_US: float | None = None
HIK_GAIN: float | None = None

# 取帧等待时间过短会把轻微网络抖动误报成故障，过长则会拖慢异常退出。
HIK_FRAME_TIMEOUT_MS = 1000


# =============================================================================
# 8. 机器人连接与“禁止误运动”开关
# =============================================================================

# 这里必须改成实验室 UR10 的真实 IP；示例地址不能证明与你的网络配置一致。
ROBOT_HOST = "192.168.0.10"
ROBOT_DASHBOARD_PORT = 29999
ROBOT_CONNECT_TIMEOUT_S = 5.0

# 第一次 robot_test 必须保持 False，此时只读取版本、关节和 TCP 状态，不发送运动命令。
ROBOT_TEST_ALLOW_MOTION = False

# 只有在示教器逐点确认 A/B/C 位姿、工作区和姿态都安全后，才允许改为 True。
# 任何会让真机运动的模式都会检查这个开关，防止示例坐标被误发给机器人。
ROBOT_POSES_CONFIRMED = False

# 是否在真机运动前要求操作者在终端再次输入指定文本。正式实验建议一直保持 True。
REQUIRE_OPERATOR_CONFIRMATION = True
OPERATOR_CONFIRM_TEXT = "I CONFIRM THE ROBOT AREA IS CLEAR"

# ur_rtde 的 -1.0 表示让库按控制器代际选择默认频率：CB 系列 125 Hz，e/UR 系列 500 Hz。
# 这里只是接口交换频率；本项目实际写日志仍可按下方 ROBOT_RECORD_HZ 降采样。
ROBOT_RTDE_FREQUENCY = -1.0

# 状态记录 125 Hz 已足以覆盖当前预计 5–40 Hz 振动，并减小文件体积。
# 它不是 moveL 的“指令发送频率”，因为本版轨迹是一次提交给控制器执行的。
ROBOT_RECORD_HZ = 125.0


# =============================================================================
# 9. 轨迹类型、示例位姿和运动参数
# =============================================================================

# static 只静止记录，line 是 A→B，l_shape 是 A→B→C。
TRAJECTORY_TYPE = "l_shape"
VALID_TRAJECTORY_TYPES: Final[set[str]] = {"static", "line", "l_shape"}

# 以下位姿格式固定为 [x, y, z, rx, ry, rz]，位置单位 m，姿态为旋转向量 rad。
# 这些只是方便 dry_run 检查代码流程的演示值，绝不能直接作为你的 UR10 实验坐标。
POINT_A = [-0.450, -0.250, 0.350, 2.220, -2.220, 0.000]
POINT_B = [-0.350, -0.250, 0.350, 2.220, -2.220, 0.000]
POINT_C = [-0.350, -0.150, 0.350, 2.220, -2.220, 0.000]

# 线速度单位 m/s，加速度单位 m/s²。初次真机运动应从更低速度开始。
LINEAR_SPEED_M_S = 0.05
LINEAR_ACCELERATION_M_S2 = 0.10

# B 点交融半径单位 m。L 形路径中它让机械臂不停下地圆滑经过 B 点。
BLEND_RADIUS_M = 0.005

# robot_test 若允许运动，只执行 A→B 的低速测试，并使用更保守的速度和加速度。
ROBOT_TEST_SPEED_M_S = 0.02
ROBOT_TEST_ACCELERATION_M_S2 = 0.05

# 软件工作区是额外保险，不等于 UR 控制器自身的安全平面，也无法识别桌面和夹具。
# 这六个范围分别限制 TCP 的 x、y、z，单位 m；实机前必须按实验台重新填写。
WORKSPACE_LIMITS_M: Final[dict[str, tuple[float, float]]] = {
    "x": (-0.80, 0.20),
    "y": (-0.80, 0.80),
    "z": (0.10, 1.20),
}

# 相邻点距离过长通常意味着单位写错或误抄示教坐标；超过阈值直接拒绝真机运动。
MAX_SEGMENT_LENGTH_M = 0.30

# 位姿的前三项与安全起点相比超过此值时，不允许假装已经位于起点。
START_POSE_TOLERANCE_M = 0.005

# 单段最长允许时间是兜底超时，不代表预计运动一定要持续这么久。
ROBOT_MOTION_TIMEOUT_S = 60.0


# =============================================================================
# 10. 控制器预留接口
# =============================================================================

# 第一版只做开环预设轨迹。disabled 表示没有在线 SFC 修正，名称故意写得直白以免误认。
CONTROL_MODE = "disabled"
EXECUTOR_MODE = "open_loop"

# 若误选 sfc，当前版本会在接触机器人前明确停止；以后真正实现后再扩展允许集合。
VALID_CONTROL_MODES: Final[set[str]] = {"disabled", "sfc"}
VALID_EXECUTOR_MODES: Final[set[str]] = {"open_loop", "servo"}


# =============================================================================
# 11. 完整实验的时间和进程参数
# =============================================================================

# start_event 发出后先静止记录 PRE_RECORD_SECONDS，再提交运动轨迹。
PRE_RECORD_SECONDS = 3.0

# 运动完成后继续记录，用于观察残余振动如何衰减。
POST_RECORD_SECONDS = 5.0

# 相机和机器人启动、报告 ready 的最长等待时间；超时后主程序会让全部子进程退出。
WORKER_READY_TIMEOUT_S = 20.0

# 每个子进程正常退出的等待时间，超时后才会作为最后手段终止该进程。
WORKER_JOIN_TIMEOUT_S = 8.0

# 记录队列不能无限增长，否则磁盘过慢时会逐渐吃完内存。
RECORD_QUEUE_MAXSIZE = 20000
ERROR_QUEUE_MAXSIZE = 100


# =============================================================================
# 12. 离线分析参数
# =============================================================================

# None 表示自动选择 outputs 下最新的 run_log.txt；也可填具体 Path 锁定一次实验。
ANALYSIS_FILE: Path | None = None

# 优先分析圆点法还是棋盘格法，可选 circles 或 checkerboard。
ANALYSIS_VISION_METHOD = "circles"

# 可选 x、y 或 magnitude；magnitude 会计算 sqrt(dx²+dy²)。
ANALYSIS_AXIS = "x"
VALID_ANALYSIS_AXES: Final[set[str]] = {"x", "y", "magnitude"}

# linear 适合直线恒速段；savgol 适合缓慢弯曲趋势；highpass 适合明确只关心某频率以上振动。
DETREND_METHOD = "linear"
VALID_DETREND_METHODS: Final[set[str]] = {"linear", "savgol", "highpass"}

# Savitzky-Golay 趋势窗口按秒定义，程序会根据实际帧率换成奇数点数。
SAVGOL_WINDOW_SECONDS = 0.50
SAVGOL_POLYORDER = 2

# highpass 模式的截止频率和阶数。截止频率不应高于你希望保留的最低振动频率。
HIGHPASS_CUTOFF_HZ = 2.0
HIGHPASS_ORDER = 4

# 频谱只在这个范围内寻找主频，避开零频漂移和超过奈奎斯特频率的无效区。
DOMINANT_FREQ_MIN_HZ = 1.0
DOMINANT_FREQ_MAX_HZ = 45.0

# 这些频带用于汇总能量，可按后续实验关注范围调整。
FREQUENCY_BANDS_HZ: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
)

# 恢复时间阈值取“静止基线 RMS×倍数”和绝对阈值中的较大者。
# 连续低于阈值达到 RECOVERY_HOLD_SECONDS 后，才算真正恢复而不是偶然穿过阈值。
RECOVERY_BASELINE_FACTOR = 3.0
RECOVERY_ABSOLUTE_THRESHOLD_MM = 0.02
RECOVERY_HOLD_SECONDS = 0.30


# =============================================================================
# 13. 配置检查
# =============================================================================

def _check_pose(name: str, pose: list[float]) -> None:
    """检查位姿必须恰好由 6 个有限数值组成，防止少抄或多抄一列。"""

    import math

    if len(pose) != 6:
        raise ValueError(f"{name} 必须包含 6 个数：[x, y, z, rx, ry, rz]。")

    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pose):
        raise ValueError(f"{name} 中存在非数值或无穷值。")


def validate_config(run_mode: str | None = None) -> None:
    """
    在导入硬件库或创建子进程前检查明显配置错误。

    run_mode 允许 main.py 的命令行参数临时覆盖 RUN_MODE；不传时就检查文件顶部选择的模式。
    """

    selected_mode = run_mode or RUN_MODE

    if selected_mode not in VALID_RUN_MODES:
        raise ValueError(
            f"未知 RUN_MODE={selected_mode!r}，可选值为 {sorted(VALID_RUN_MODES)}。"
        )

    if VISION_SOURCE not in VALID_VISION_SOURCES:
        raise ValueError(
            f"未知 VISION_SOURCE={VISION_SOURCE!r}，可选值为 {sorted(VALID_VISION_SOURCES)}。"
        )

    if VISION_METHOD not in VALID_VISION_METHODS:
        raise ValueError(
            f"未知 VISION_METHOD={VISION_METHOD!r}，可选值为 {sorted(VALID_VISION_METHODS)}。"
        )

    if TRAJECTORY_TYPE not in VALID_TRAJECTORY_TYPES:
        raise ValueError(
            f"未知 TRAJECTORY_TYPE={TRAJECTORY_TYPE!r}，"
            f"可选值为 {sorted(VALID_TRAJECTORY_TYPES)}。"
        )

    if CONTROL_MODE not in VALID_CONTROL_MODES:
        raise ValueError(f"未知 CONTROL_MODE={CONTROL_MODE!r}。")

    if EXECUTOR_MODE not in VALID_EXECUTOR_MODES:
        raise ValueError(f"未知 EXECUTOR_MODE={EXECUTOR_MODE!r}。")

    if selected_mode in {"robot_test", "experiment"} and CONTROL_MODE == "sfc":
        raise ValueError(
            "当前版本只预留了 SFC 接口，尚未实现在线控制。"
            "为了避免把开环运动误当成 SFC，程序拒绝连接机器人。"
        )

    if selected_mode == "experiment" and EXECUTOR_MODE != "open_loop":
        raise ValueError("当前正式实验只实现 open_loop 轨迹执行器。")

    if selected_mode in {"robot_test", "experiment"} and not ROBOT_HOST.strip():
        raise ValueError("真机模式必须填写非空 ROBOT_HOST。")

    if CHECKERBOARD_INNER_CORNERS[0] < 2 or CHECKERBOARD_INNER_CORNERS[1] < 2:
        raise ValueError("棋盘格横向和纵向内角点数都必须至少为 2。")

    if CHECKER_SQUARE_MM <= 0:
        raise ValueError("CHECKER_SQUARE_MM 必须为正数。")

    if not 3 <= CIRCLE_EXPECTED_COUNT <= len(CIRCLE_LAYOUT_MM):
        raise ValueError("CIRCLE_EXPECTED_COUNT 必须在 3 到圆点理论坐标数量之间。")

    if not 3 <= MIN_VALID_CIRCLES <= CIRCLE_EXPECTED_COUNT:
        raise ValueError("MIN_VALID_CIRCLES 必须介于 3 和预期圆点总数之间。")

    if GAUSSIAN_BLUR_KERNEL < 1 or GAUSSIAN_BLUR_KERNEL % 2 == 0:
        raise ValueError("GAUSSIAN_BLUR_KERNEL 必须为正奇数；1 表示不滤波。")

    if (CAMERA_MATRIX is None) != (DISTORTION_COEFFICIENTS is None):
        raise ValueError("CAMERA_MATRIX 与 DISTORTION_COEFFICIENTS 必须同时填写。")

    if VISION_ROI is not None:
        if len(VISION_ROI) != 4 or any(value < 0 for value in VISION_ROI):
            raise ValueError("VISION_ROI 必须是非负的 (x, y, width, height)。")
        if VISION_ROI[2] == 0 or VISION_ROI[3] == 0:
            raise ValueError("VISION_ROI 的 width 和 height 必须大于 0。")

    for name, pose in (("POINT_A", POINT_A), ("POINT_B", POINT_B), ("POINT_C", POINT_C)):
        _check_pose(name, pose)

    if LINEAR_SPEED_M_S <= 0 or LINEAR_ACCELERATION_M_S2 <= 0:
        raise ValueError("线速度和线加速度必须为正数。")

    if BLEND_RADIUS_M < 0:
        raise ValueError("BLEND_RADIUS_M 不能为负数。")

    if ROBOT_RECORD_HZ <= 0:
        raise ValueError("ROBOT_RECORD_HZ 必须为正数。")

    if PRE_RECORD_SECONDS < 0 or POST_RECORD_SECONDS < 0:
        raise ValueError("运动前后记录时间不能为负数。")

    if ANALYSIS_VISION_METHOD not in {"circles", "checkerboard"}:
        raise ValueError("ANALYSIS_VISION_METHOD 只能是 circles 或 checkerboard。")

    if ANALYSIS_AXIS not in VALID_ANALYSIS_AXES:
        raise ValueError(f"ANALYSIS_AXIS 可选值为 {sorted(VALID_ANALYSIS_AXES)}。")

    if DETREND_METHOD not in VALID_DETREND_METHODS:
        raise ValueError(f"DETREND_METHOD 可选值为 {sorted(VALID_DETREND_METHODS)}。")

    # 这里只创建程序自己的输出目录，不会创建假的输入数据。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
