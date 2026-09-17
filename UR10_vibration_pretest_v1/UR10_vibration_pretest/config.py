"""
UR10 末端振动预实验：统一配置文件。

这个文件不直接“做实验”，而是回答三个问题：
1. 这次运行哪种实验流程；
2. 视觉、机器人、分析各自使用哪些参数；
3. 启动前哪些配置必须先被拦住，避免后面产生危险动作或误导性结果。

阅读方式建议：
- 先看第 1、2 节，确认程序会跑“离线视觉、机器人干跑、真机测试、完整实验、离线分析”中的哪一条路。
- 再按实验阶段阅读：视觉看第 2-7 节，机器人看第 8-11 节，结果分析看第 12 节。
- 最后看 validate_config()。它相当于启动前检查表，只检查配置是否明显错误，不会连接相机或机器人。

安全原则：
- ROBOT、POINT、WORKSPACE 相关值必须来自真实示教和现场确认，不能把示例值当实验值。
- HIK 开头的值只服务海康相机 SDK；普通图片文件夹和视频分析不依赖这些设置。
- .venv、依赖安装、VS Code 解释器设置不在这里配置；这些看 ENVIRONMENT_SETUP.md。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final


# =============================================================================
# 0. 项目路径
# =============================================================================

# 本段确定整个项目的“文件坐标系”。
# 输入：config.py 自己所在的位置；输出：后续图片、视频、结果目录都从 PROJECT_DIR 派生。
# 实验作用：无论从 VS Code、终端还是脚本启动，程序都能找到同一批输入和输出位置。
PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent

# 本段只声明数据应该放在哪里。
# 输入：你人工放入的图片序列或视频；输出：程序生成的 outputs 结果目录。
# 实验作用：把原始数据和计算结果分开，便于回看某次实验到底用了哪些输入。
IMAGE_FOLDER = PROJECT_DIR / "input_images"
VIDEO_PATH = PROJECT_DIR / "input_video" / "test.mp4"
OUTPUT_ROOT = PROJECT_DIR / "outputs"


# =============================================================================
# 1. 总运行模式：一次只允许选择一条路线
# =============================================================================

# 本段选择本次启动的主流程。
# 输入：RUN_MODE 字符串；输出：main.py 会进入对应函数分支。
# 实验作用：预实验理解代码时通常先用 vision_test；真机相关模式必须等坐标和安全区确认后再启用。
RUN_MODE = "vision_test"

# 本段定义“合法流程清单”，供 validate_config() 检查。
# 每个模式对应一类实验任务；写错模式名时，程序会在启动阶段停止，而不是跑到一半才失败。
VALID_RUN_MODES: Final[set[str]] = {
    "vision_test",    # 只测图像读取、标志识别和位移计算。
    "vision_capture", # 只高速采集相机原始帧，保存为 RAW，尽量少做额外处理。
    "vision_offline", # 读取 vision_capture 的 RAW 结果，再离线逐帧完整识别。
    "offline_static_test", # 机械臂可完全断电；仅用工业相机重复测静止视觉/环境底噪。
    "robot_dry_run",  # 只生成并检查轨迹，不导入 UR 库，也不连接真机。
    "robot_connection_test", # 只读检测 UR 通信，不创建控制接口、不发送运动命令。
    "robot_test",     # 连接 UR，默认只读取状态；必须再次开关才允许低速运动。
    "experiment",     # 相机、UR 和记录进程共同运行的正式预实验。
    "x_line_experiment",  # 以当前 TCP 为 A 点，执行 X 方向往返相对运动并采 RAW。
    "xy_line_experiment", # 以当前 TCP 为 A 点，执行 X-Y 倾斜直线往返并采 RAW。
    "xy_l_experiment",    # 以当前 TCP 为 A 点，执行 X-Y 平面 L 折线往返并采 RAW。
    "batch_experiment",   # 相机和 UR 各初始化一次，按 UI 计划连续执行分段批量实验。
    "micro_closed_loop",  # 一键XY多目标视觉闭环逼近测试。
    "batch_vision_offline", # 批次采集结束后，另行读取 AVI 和侧车时间戳做视觉识别。
    "boundary_check",     # 不等待手电筒，只显示实时画面并低速检查当前计划最大包络。
    "analyze",        # 读取已有 TXT/JSONL 记录并生成振动分析结果。
}


# =============================================================================
# 2. 视觉输入来源与算法选择
# =============================================================================

# 本段选择图像从哪里来。
# 输入：图片文件夹、视频文件或海康相机；输出：camera.py 会把不同来源统一包装成 FramePacket。
# 实验作用：离线预实验通常用 image_folder/video；正式在线实验才会使用 hik_camera。
VISION_SOURCE = "hik_camera"
VALID_VISION_SOURCES: Final[set[str]] = {"image_folder", "video", "hik_camera"}

# 本段选择每帧图像用哪套识别方法。
# 输入：同一张 FramePacket；输出：圆点法、棋盘格法或两者并行的位移/坐标结果。
# 实验作用：compare 适合前期判断哪种标志更稳定；正式实验可固定为更可靠的一种，减少计算负担。
VISION_METHOD = "checkerboard"
VALID_VISION_METHODS: Final[set[str]] = {"circles", "checkerboard", "compare"}

# 本段限定离线图片序列的文件类型和读取顺序。
# 输入：input_images 文件夹中的文件名；输出：按自然顺序进入处理循环的图片帧。
# 实验作用：避免 frame10 排在 frame2 前面，也避免把非图片文件误当作实验帧。
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"
)

# 本段控制“离线图片序列的时间轴从哪里来”。
# 输入：可选 timestamps.csv，格式可为 filename,time_s 或 frame_id,time_s。
# 输出：每张图片进入 vision_results.txt 时使用的 analysis_time_s。
# 实验作用：真实时间戳优先；没有时间戳时才用 IMAGE_FOLDER_FPS 兜底，避免误把电脑解码速度当采样速度。
IMAGE_TIMESTAMPS_CSV: Path | None = IMAGE_FOLDER / "timestamps.csv"
IMAGE_TIMESTAMPS_REQUIRED = False

# 本段是离线图片没有真实时间戳时的兜底时间模型。
# 输入：frame_id 和 IMAGE_FOLDER_FPS；输出：analysis_time_s = frame_id / fps。
# 实验作用：让纯图片序列也能做频谱分析；若相机实际帧率偏离 132 fps，分析阶段会再给出提示。
IMAGE_FOLDER_FPS = 132.0

# 本段控制“采样率偏差是否提示”。
# 输入：分析阶段从 analysis_time_s 估计出的真实 fps。
# 输出：正常接近 132 fps 时不提示；偏差超过容差或出现大间隔时写入摘要并在终端提醒。
EXPECTED_VISION_FPS: float | None = 132.23
FPS_WARNING_RELATIVE_TOLERANCE = 0.03
FPS_WARNING_ABSOLUTE_TOLERANCE_HZ = 1.0
FRAME_GAP_WARNING_FACTOR = 2.5

# 本段只控制人眼预览，不改变识别结果。
# 输入：VisionProcessor 画好标注后的 debug 图；输出：OpenCV 窗口显示。
# 实验作用：现场调试时可以边跑边看是否识别错点；无桌面环境或批处理时应关闭。
SHOW_PREVIEW = True

# 本段限制预览窗口宽度，避免高分辨率相机画面超过屏幕。
# 若 SHOW_PREVIEW=True，操作者还可以按 q 或 Esc 提前结束当前视觉测试。
PREVIEW_MAX_WIDTH = 1400

# 本段控制人工质检图的保存节奏。
# 输入：VisionProcessor 生成的 debug 图；输出：debug_images 文件夹中的抽样 jpg。
# 实验作用：调试图只帮助人检查识别点、编号和画质，不参与后续分析；132 fps 下每 66 帧约半秒一张。
SAVE_DEBUG_IMAGE = True
DEBUG_IMAGE_EVERY_N_FRAMES = 66
MAX_DEBUG_IMAGES = 300

# 本段控制视觉链路性能计时。
# 输入：每帧相机取流、预处理、棋盘格识别、绘图、写文件等阶段。
# 输出：vision_results.txt 中的 timing_ms 字段，以及 vision_timing_summary.txt 汇总报告。
# 实验作用：只观察耗时瓶颈，不改变识别算法、不改变 ROI、不优化处理流程。
ENABLE_VISION_TIMING = True
VISION_TIMING_WARMUP_FRAMES = 3
SAVE_PER_FRAME_TIMING = True

# 本段控制“双阶段视觉预实验”的高速采集阶段。
# 输入：海康相机实时 Mono8 图像；输出：outputs/vision_capture_时间戳 中的 frames.raw 和时间戳文件。
# 实验作用：先用尽量轻的流程保留相机原始帧，之后再用 vision_offline 慢慢逐帧完整识别。
CAPTURE_DURATION_S: float | None = None
CAPTURE_START_COUNTDOWN_S = 3.0
# stream_raw 会边采集边追加写 frames.raw.tmp；
# memmap_raw 会复制到内存映射文件；
# ram_then_raw 会先放 RAM，结束后再整块写 RAW，避免采集循环被写盘拖慢。
CAPTURE_STORAGE_MODE = "stream_raw"
CAPTURE_RAW_WRITE_CHUNK_FRAMES = 64
CAPTURE_REQUIRE_MONO8 = True
CAPTURE_SAVE_SAMPLE_IMAGES = True
CAPTURE_SHOW_PREVIEW = True
CAPTURE_PREVIEW_FPS = 20.0
RAW_CAPTURE_DIR: Path | None = None
OFFLINE_SHOW_PREVIEW = False
OFFLINE_PROGRESS_EVERY_N_FRAMES = 25

# 本段服务“手机手电筒同步标记”门控，只在三个相对运动实验中使用。
# 输入：RAW 采集循环中 frame[::16, ::16] 的稀疏亮度统计；输出：是否允许机器人开始运动。
# 实验作用：索尼相机由人手动录像，工业相机和索尼画面里的同一次亮度峰用于离线对齐。
BRIGHTNESS_BASELINE_SECONDS = 0.5
FLASH_WAIT_TIMEOUT_SECONDS = 10.0
FLASH_MIN_DURATION_SECONDS = 0.15
VISION_RECOVERY_STABLE_SECONDS = 1.0
FLASH_RECOVERY_TIMEOUT_SECONDS = 5.0
POST_MOTION_RECORD_SECONDS = 1.0
# 2026-09-01 现场 RAW 回放：基线约 95.92、正常波动小于 2，手机手电筒峰值约 114.81。
# 下面三项取 max 后要求相对基线出现约 +8 的明显增亮，并连续保持 FLASH_MIN_DURATION_SECONDS。
FLASH_MEAN_RELATIVE_INCREASE = 0.08
FLASH_MEAN_ABSOLUTE_INCREASE = 8.0
FLASH_MEAN_MAD_MULTIPLIER = 6.0
FLASH_SATURATION_THRESHOLD = 245
FLASH_SATURATION_RELATIVE_INCREASE = 0.005
FLASH_RECOVERY_MEAN_TOLERANCE = 8.0
FLASH_RECOVERY_SATURATION_TOLERANCE = 0.01

# 本段是正式实验专用的额外保护。
# 输入：完整实验中的实时 debug 图；输出：是否允许边实验边落盘保存质检图。
# 实验作用：默认关闭，避免磁盘写入拖慢在线取图；需要现场验证电脑性能后再打开。
SAVE_DEBUG_IMAGE_DURING_EXPERIMENT = False


# =============================================================================
# 3. 图像预处理
# =============================================================================

# 本段决定每帧图像中真正送入识别算法的区域。
# 输入：原始相机画面；输出：整幅图或裁剪后的 ROI 小图。
# 实验作用：相机位置固定后，裁剪到测量纸附近可以减少背景干扰并提高处理速度。
# ROI 格式为 x, y, width, height；None 表示暂时处理整幅图。
VISION_ROI: tuple[int, int, int, int] | None = None

# 本段决定识别前是否先做轻微平滑。
# 输入：灰度图；输出：降噪后的灰度图。
# 实验作用：小核可以压低传感器噪声；核过大则会抹掉圆点边缘或棋盘格角点，所以默认保守。
# 数值必须是正奇数；1 表示不滤波。
GAUSSIAN_BLUR_KERNEL = 3

# 本段接收相机标定结果。
# 输入：3×3 内参矩阵和畸变系数；输出：像素点投影、去畸变和空间坐标计算所需的相机模型。
# 实验作用：像素坐标可以先不依赖标定；一旦需要空间坐标，这里就必须填真实标定值。
# 两个参数必须同时填写或同时保持 None，配置检查会阻止只填一半。
CAMERA_MATRIX: list[list[float]] | None = None
DISTORTION_COEFFICIENTS: list[float] | None = None

# 本段控制是否保存识别点的原始像素坐标。
# 输入：圆点法/棋盘格法识别出的点；输出：每帧 JSON 中的 circle_points_px / checker_corners_px。
# 实验作用：先检查像素坐标是否可靠，再判断二维位移或空间坐标是否可信。
SAVE_POINT_COORDINATES = True

# 本段控制是否把像素点进一步投影为空间点。
# 输入：像素坐标、相机内参，以及下面所选模式需要的平面/外参参数。
# 输出：每帧 JSON 中的 *_spatial 字段。
# 实验作用：空间坐标作为第二层结果和像素坐标并列保存；如果空间结果异常，可以回头先看像素坐标是否已经异常。
SPATIAL_COORDINATES_ENABLED = False
SPATIAL_COORDINATE_MODE = "camera_plane"
VALID_SPATIAL_COORDINATE_MODES: Final[set[str]] = {"camera_plane", "robot_plane"}

# camera_plane 模式输入：目标点所在平面到相机的固定 Z 深度，单位 m。
# 输出：相机坐标系下的三维点；不输出机器人坐标。
SPATIAL_CAMERA_PLANE_Z_M: float | None = None

# robot_plane 模式输入：相机到机器人外参，以及测量平面在机器人坐标系中的点和法向。
# 输出：同一批点的相机坐标和机器人坐标。
CAMERA_TO_ROBOT_ROTATION: list[list[float]] | None = None
CAMERA_TO_ROBOT_TRANSLATION_M: list[float] | None = None
MEASUREMENT_PLANE_POINT_ROBOT_M: list[float] | None = None
MEASUREMENT_PLANE_NORMAL_ROBOT: list[float] | None = None

# 本段定义图像质量记录的阈值。
# 输入：每帧的拉普拉斯方差、过暗像素比例、过亮像素比例；输出：质量分和警告信息。
# 实验作用：帮助区分“真实振动导致点动了”和“图像太糊/曝光错误导致识别不可靠”。
# 这些阈值只影响报警和记录，不会擅自删除一帧。
BLUR_WARNING_THRESHOLD = 60.0

DARK_PIXEL_THRESHOLD = 15
BRIGHT_PIXEL_THRESHOLD = 245


# =============================================================================
# 4. 棋盘格参数
# =============================================================================

# 本段定义棋盘格法要寻找的几何模板。
# 输入：实际打印纸上的黑白棋盘格；输出：OpenCV 在图像中应找到的内角点网格。
# 实验作用：只有模板尺寸和实物一致，棋盘格法才能把角点位移解释成纸面位移。
# 注意 OpenCV 要的是“内角点数量”，不是黑白方格数量。
# 当前购买棋盘格为 12×9 个 3 mm 方格，图案尺寸 36×27 mm，
# 因此可检测的内角点数量是 11×8。
CHECKERBOARD_INNER_CORNERS = (11, 8)

# 本段提供棋盘格法的物理尺度。
# 输入：单个黑白方格边长，单位 mm；输出：像素位移到纸面毫米位移的比例。
# 实验作用：这个值填错时，位移趋势可能还像对的，但毫米量级会整体错。
CHECKER_SQUARE_MM = 3.0

# 本段控制棋盘格角点的亚像素精修。
# 输入：初步检测到的角点；输出：更精细的角点坐标。
# 实验作用：提高静止帧中的角点稳定性；若图像模糊，继续加严迭代参数通常也救不回来。
CHECKER_SUBPIX_MAX_ITER = 40
CHECKER_SUBPIX_EPS = 0.001

# 本段给棋盘格拟合质量设定报警线。
# 输入：角点与理想网格之间的像素残差；输出：质量分中的 warning。
# 实验作用：提示棋盘格可能识别歪了、局部遮挡了，或测量纸本身不够平整。
CHECKER_RESIDUAL_WARNING_PX = 0.8


# =============================================================================
# 5. 圆点参数
# =============================================================================

# 本段定义圆点法需要多少个点才算可用。
# 输入：每帧识别出的候选圆点；输出：是否允许继续做圆点跟踪和刚体拟合。
# 实验作用：允许少量圆点临时丢失，但低于 MIN_VALID_CIRCLES 时不再相信该帧位移。
CIRCLE_EXPECTED_COUNT = 7
MIN_VALID_CIRCLES = 5

# 本段用轮廓面积筛掉明显不是标志点的图案。
# 输入：二值化后每个候选轮廓的像素面积；输出：保留下来的圆点候选。
# 实验作用：首次实拍后应根据 debug 图调整，保证真实圆点被保留、背景杂点被排除。
CIRCLE_MIN_AREA_PX = 25.0
CIRCLE_MAX_AREA_PX = 12000.0

# 本段从“形状像不像圆”继续筛选候选点。
# 输入：轮廓圆度和椭圆长短轴比例；输出：更可靠的圆点中心。
# 实验作用：透视会让圆变成椭圆，所以阈值不能过严；过松又容易把污点、文字或边缘当圆点。
CIRCLE_MIN_CIRCULARITY = 0.68
CIRCLE_MIN_AXIS_RATIO = 0.52

# 本段控制相邻帧之间如何维持“同一个圆点”的身份。
# 输入：上一帧圆点位置和当前帧候选圆点；输出：编号稳定的圆点序列。
# 实验作用：它应大于相邻两帧最大真实位移，但不能大到允许不同圆点互相串号。
CIRCLE_MAX_MATCH_DISTANCE_PX = 45.0

# 本段定义圆点测量纸的理论布局。
# 输入：纸面坐标系下每个圆点中心，单位 mm；输出：SVG 测量纸和圆点法的物理尺度参考。
# 实验作用：不对称布局可以减少旋转后的身份歧义，让程序更容易知道“哪个点是哪个点”。
CIRCLE_LAYOUT_MM: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 0.0),
    (11.0, 2.0),
    (24.0, 0.5),
    (4.0, 12.0),
    (17.0, 15.0),
    (29.0, 10.5),
    (9.0, 26.0),
)

# 本段只影响生成的测量纸外观。
# 输入：圆点直径和方向锚点直径，单位 mm；输出：marker_sheet.svg 中的圆形标志。
# 实验作用：锚点略大，方便人眼辨认方向；当前位移计算仍主要使用圆点中心。
CIRCLE_DIAMETER_MM = 3.0
CIRCLE_ANCHOR_DIAMETER_MM = 4.5

# 本段给圆点整体刚体拟合设定报警线。
# 输入：多个圆点从参考帧到当前帧的刚体变换残差；输出：质量分中的 warning。
# 实验作用：残差大通常说明串点、局部翘曲、纸面不平或轮廓识别错误。
CIRCLE_RESIDUAL_WARNING_PX = 1.0


# =============================================================================
# 6. 可打印复合测量纸
# =============================================================================

# 本段控制是否生成实验用测量纸。
# 输入：棋盘格和圆点布局参数；输出：一份可打印的 marker_sheet.svg。
# 实验作用：vision_test 时自动生成参考纸，方便保持“代码里的几何尺寸”和“打印出来的实物”一致。
# 打印时必须使用 100% 比例，不能选择“适应页面”。
GENERATE_MARKER_SHEET_ON_VISION_TEST = False
MARKER_SHEET_PATH = OUTPUT_ROOT / "marker_sheet.svg"

# 本段定义测量纸的外框和留白。
# 输入：纸张宽高和边距，单位 mm；输出：SVG 画布尺寸和图案摆放区域。
# 实验作用：平面打印后裁剪使用；若贴在圆柱面上，应尽量减小弯曲并把被测区域放在相机正对位置。
MARKER_SHEET_WIDTH_MM = 82.0
MARKER_SHEET_HEIGHT_MM = 42.0
MARKER_MARGIN_MM = 4.0


# =============================================================================
# 7. 海康相机参数
# =============================================================================

# 本段指定在线取图时连接哪台海康相机。
# 输入：相机序列号；输出：hik_camera 模式实际打开的设备。
# 实验作用：单相机调试可留空；正式实验建议填写序列号，避免多相机环境连错设备。
HIK_CAMERA_SERIAL = ""

# 本段告诉 Python 去哪里找海康 MVS SDK 的导入文件。
# 输入：官方 Samples/Python/MvImport 路径；输出：运行时追加到 sys.path 的目录。
# 实验作用：只在 hik_camera 模式需要；离线图片和视频流程完全不依赖它。
HIK_MVS_IMPORT_PATH: str | None = (
    r"D:\Software\MVScamera\MVS\Development\Samples\Python\MvImport"
)

# 本段控制相机成像亮度。
# 输入：曝光时间和增益；输出：相机节点参数或保留相机当前设置。
# 实验作用：稳定曝光能减少识别点明暗波动；首次连接应先在 MVS 客户端确认可用范围。
# None 表示程序不主动修改相机当前配置。
# 当前 132 fps 单帧周期约为 7576 us，曝光应低于这个值以免压低帧率。
HIK_EXPOSURE_US: float | None = 6500.0
HIK_GAIN: float | None = None

# 本段控制在线取帧时“等一帧最多等多久”。
# 输入：相机 SDK 取帧调用；输出：成功返回 FramePacket 或超时错误。
# 实验作用：过短会把轻微网络抖动误报成故障，过长则会拖慢异常退出。
HIK_FRAME_TIMEOUT_MS = 1000


# =============================================================================
# 8. 机器人连接与“禁止误运动”开关
# =============================================================================

# 本段定义程序如何找到 UR10 控制器。
# 输入：实验室网络中的机器人 IP 和 dashboard 端口；输出：robot.py 建立状态读取或运动控制连接。
# 实验作用：只有 robot_test/experiment 会真正使用；示例地址不能证明与你的网络配置一致。
ROBOT_HOST = "192.168.125.12"
ROBOT_DASHBOARD_PORT = 29999
ROBOT_CONNECT_TIMEOUT_S = 5.0
# CB3 在上一次RTDE控制脚本刚释放或网络瞬时抖动时，可能主动关闭新会话。
# 重试仅发生在任何轨迹下发之前；每次失败都会先停止/断开已创建的接口。
ROBOT_RTDE_CONNECT_ATTEMPTS = 3
ROBOT_RTDE_CONNECT_RETRY_DELAY_S = 1.0

# 本段把 robot_test 拆成“只连机读取”和“允许低速动一下”两级。
# 输入：ROBOT_TEST_ALLOW_MOTION；输出：robot_test 是否会发送 A→B 测试运动。
# 实验作用：第一次接触实机必须保持 False，先确认连接、坐标读取和日志流程正常。
ROBOT_TEST_ALLOW_MOTION = False

# 本段专门保护“以当前 TCP 为 A 点”的三个相对运动实验。
# 输入：实验室现场确认后的人工开关；输出：是否允许 x_line/xy_line/xy_l 三个新模式发送运动命令。
# 实验作用：这些模式不依赖 POINT_A/B/C，但仍必须有独立的真机运动许可，默认绝不运动。
ROBOT_RELATIVE_MOTION_ENABLED = True

# 本段是所有真机运动前的总安全闸。
# 输入：人工示教并确认后的 A/B/C 位姿和工作区；输出：是否允许 robot.py 发送运动命令。
# 实验作用：防止示例坐标或未确认坐标被误发给机器人；只有现场逐点确认后才可改 True。
ROBOT_POSES_CONFIRMED = False

# 本段要求操作者在真机运动前做最后一次人工确认。
# 输入：终端中输入的确认文本；输出：继续执行或立即停止。
# 实验作用：给人留一个“手离键盘前再想一次”的关口，正式实验建议一直保持 True。
REQUIRE_OPERATOR_CONFIRMATION = True
OPERATOR_CONFIRM_TEXT = "I CONFIRM THE ROBOT AREA IS CLEAR"

# 本段设置 RTDE 接口和机器人控制器交换数据的频率。
# 输入：ur_rtde 连接参数；输出：底层状态刷新频率。
# 实验作用：这不是 moveL 指令发送频率，也不是最终日志频率；日志仍按 ROBOT_RECORD_HZ 记录。
# -1.0 表示让 ur_rtde 按控制器代际选择默认频率：CB 系列 125 Hz，e/UR 系列 500 Hz。
ROBOT_RTDE_FREQUENCY = -1.0

# 本段决定机器人状态日志的采样密度。
# 输入：RTDE 持续读取到的机器人状态；输出：run_log.txt 中 robot_state 记录的时间序列。
# 实验作用：125 Hz 足以覆盖当前预计 5-40 Hz 振动，同时控制文件体积。
ROBOT_RECORD_HZ = 125.0

# 本段服务只读机械臂通信检测。
# 输入：检测持续时间；输出：robot_connection_test 按 ROBOT_RECORD_HZ 连续读取状态。
# 实验作用：确认 RTDE Receive 能稳定读状态，且全程不创建控制接口、不发送任何运动命令。
ROBOT_CONNECTION_TEST_SECONDS = 3.0


# =============================================================================
# 9. 轨迹类型、示例位姿和运动参数
# =============================================================================

# 本段选择机器人在完整实验中执行哪种预设轨迹。
# 输入：TRAJECTORY_TYPE；输出：robot.py 生成 static、A→B 或 A→B→C 路径。
# 实验作用：对应你的静止、直线匀速、L 形转弯等预实验工况。
TRAJECTORY_TYPE = "l_shape"
VALID_TRAJECTORY_TYPES: Final[set[str]] = {"static", "line", "l_shape"}

# 本段提供轨迹端点位姿。
# 输入：示教器确认后的 TCP 位姿 [x, y, z, rx, ry, rz]；输出：轨迹生成器使用的 A/B/C 点。
# 实验作用：这些点决定机械臂实际走哪里。当前值只是 dry_run 演示值，绝不能直接作为实机坐标。
# 位置单位 m，姿态为旋转向量 rad。
POINT_A = [-0.450, -0.250, 0.350, 2.220, -2.220, 0.000]
POINT_B = [-0.350, -0.250, 0.350, 2.220, -2.220, 0.000]
POINT_C = [-0.350, -0.150, 0.350, 2.220, -2.220, 0.000]

# 本段定义完整实验轨迹的速度级别。
# 输入：目标线速度和线加速度；输出：发送给 UR 控制器的 moveL/moveJ 相关运动参数。
# 实验作用：速度越高越容易激发振动，但风险也更高；初次真机运动应从更低速度开始。
LINEAR_SPEED_M_S = 0.05
LINEAR_ACCELERATION_M_S2 = 0.10

# 本段控制 L 形轨迹在 B 点是否“停顿转弯”。
# 输入：B 点交融半径，单位 m；输出：控制器在 A→B→C 中对 B 点的圆滑过渡。
# 实验作用：交融半径越大，越接近连续转弯；为 0 时更接近到点停下再走下一段。
BLEND_RADIUS_M = 0.005

# 本段只服务 robot_test 的低速验证。
# 输入：A/B 点和保守速度参数；输出：一段短距离 A→B 测试运动。
# 实验作用：在正式 experiment 前验证机器人可连接、可运动、可记录，但不直接跑完整实验速度。
ROBOT_TEST_SPEED_M_S = 0.02
ROBOT_TEST_ACCELERATION_M_S2 = 0.05

# 本段服务三个相对运动实验的默认 UI 参数和安全范围。
# 输入：启动器或命令行传入的 mm/s、秒、角度、方向；输出：main.py/robot.py 二次校验后的相对轨迹。
# 实验作用：每次从当前实际 TCP 作为 A 点，只改 X/Y，不复用示例 POINT_A/B/C。
ROBOT_EXPERIMENT_DEFAULT_SPEED_MM_S = 1.0
ROBOT_EXPERIMENT_DEFAULT_ONE_WAY_TIME_S = 9.0
# P01 的 L 时间保持 9 s；X/D 每轴缩为同预设 L 单轴位移的 2/3。
# 现场输入只保留 0.1 s，并向下取整以保证实际包络不超过 2/3 目标。
ROBOT_EXPERIMENT_DEFAULT_X_LINE_ONE_WAY_TIME_S = 3.0
ROBOT_EXPERIMENT_DEFAULT_XY_LINE_ONE_WAY_TIME_S = 4.2
ROBOT_EXPERIMENT_DEFAULT_L_ONE_WAY_TIME_S = ROBOT_EXPERIMENT_DEFAULT_ONE_WAY_TIME_S
# 旧命令行参数只作为兼容入口；新 UI 和批量计划统一使用 ONE_WAY_TIME。
ROBOT_EXPERIMENT_DEFAULT_TOTAL_TIME_S = 2.0 * ROBOT_EXPERIMENT_DEFAULT_ONE_WAY_TIME_S
ROBOT_EXPERIMENT_DEFAULT_X_ONE_WAY_TIME_S = ROBOT_EXPERIMENT_DEFAULT_ONE_WAY_TIME_S / 2.0
ROBOT_EXPERIMENT_DEFAULT_Y_ONE_WAY_TIME_S = ROBOT_EXPERIMENT_DEFAULT_ONE_WAY_TIME_S / 2.0
ROBOT_EXPERIMENT_DEFAULT_ANGLE_DEG = 45.0
ROBOT_EXPERIMENT_ACCELERATION_M_S2 = 0.10
ROBOT_EXPERIMENT_MIN_SPEED_MM_S = 0.5
ROBOT_EXPERIMENT_MAX_SPEED_MM_S = 20.0
ROBOT_EXPERIMENT_MAX_SEGMENT_MM = 100.0
ROBOT_EXPERIMENT_MAX_TOTAL_TIME_S = 120.0
ROBOT_EXPERIMENT_MAX_ANGLE_DEG = 90.0
ROBOT_EXPERIMENT_STOP_SPEED_MM_S = 0.2
ROBOT_RETURN_WARNING_MM = 0.5
ROBOT_RELATIVE_BLEND_MM = 1.0

# 批量状态机：每段运动前记录 1 s，回到 A 后记录 1 s。
# 相邻运动之间不再额外等待；这两段画面共同提供约 2 s 的分析窗口。
BATCH_STATIC_BASELINE_SECONDS = 5.0
BATCH_PRE_MOTION_SECONDS = 1.0
BATCH_POST_MOTION_SECONDS = 1.0
BATCH_VIDEO_CODEC = "MJPG"
# 批量相机刚启动时，Windows 创建预览窗口和并行子进程加载可能造成一次短暂停顿。
# 预热期间持续取帧并初始化预览，但不做正式缺帧统计，也不进入手电筒亮度基线。
# 预热结束后重新建立 frame_id 基准；正式批量期间的缺帧保护仍保持不变。
BATCH_CAMERA_WARMUP_SECONDS = 1.0
# 手电筒门禁和段间空闲时保留正常预览。
BATCH_PREVIEW_FPS = 10.0
# 整个批量流程始终使用相同的小画幅预览，避免静止/运动切换时窗口忽大忽小；
# 正式录像仍使用相机完整分辨率和真实帧率，不会因此裁剪或降低保存分辨率。
BATCH_RECORDING_PREVIEW_FPS = 10.0
BATCH_PREVIEW_MAX_WIDTH = 700
# 批量累计缺帧率只有严格超过 15% 才中止；零星单帧缺失保留在时间戳中，
# 后续离线分析按真实 frame_id/time 基准处理。连续大段断流仍由下一项独立拦截。
BATCH_MAX_MISSING_RATIO = 0.15
# 132 fps 下允许单次连续缺失最多 30 帧（约 0.23 s）；只有超过 30 帧
# 才立即判定为严重断流。累计零星缺帧另由上面的 15% 阈值约束。
BATCH_MAX_CONSECUTIVE_MISSING_FRAMES = 30

# -----------------------------------------------------------------------------
# 9b. 一键 XY 微动闭环能力测试（micro_closed_loop）
# -----------------------------------------------------------------------------
# 本段服务一个独立的一键流程：设备检查 -> 相机预热 -> 轴向/符号探针
# -> 5 s 静止基线 -> 单一视觉原点 -> X/Y 多目标闭环逼近 -> 自动收尾。
#
# 与前一轮被舍弃的开环微动实验（micro_motion_experiment）的区别：
# 那一版按固定档位表开环走步、只回答“走一步走多少”；这一版是**闭环**，
# 用"走一步—停稳—测量—再走一步"逼近一个视觉目标位置，因此会自然产生
# 大量不同幅值的小命令，并回答“能否逼近 5 μm”与“是否形成持续极限环”。
#
# 全程只做最慢的离散闭环，不引入轨迹规划、MPC、PID、APF、SFC。
# 唯一的控制律就是操作者给出的 command = Kp · error。

# 目标幅值（μm）与两个自由度。方向固定为 +1 / −1 两个。
# 旧常量仅供历史离线分析函数/旧结果读取；当前 micro_closed_loop 主流程不再使用。
MICRO_LOOP_AMPLITUDES_UM: Final[tuple[int, ...]] = (5, 20, 50)
MICRO_LOOP_AXES: Final[tuple[str, ...]] = ("X", "Y")

# 当前多目标闭环实验。相对步长累加后得到 100, 250, 460, 360, 210, 0 μm。
# X/Y 各执行一次相同序列；另一轴名义目标始终保持在本次实验视觉原点。
MICRO_TARGET_RELATIVE_STEPS_UM: Final[tuple[int, ...]] = (
    100,
    150,
    210,
    -100,
    -150,
    -210,
)
MICRO_TARGET_ABSOLUTE_UM: Final[tuple[int, ...]] = (100, 250, 460, 360, 210, 0)
MICRO_TARGET_MAX_ITER = 12
MICRO_TARGET_STABLE_TOL_UM = 3.0
MICRO_TARGET_STABLE_COUNT = 3
MICRO_TARGET_SMALL_COMMAND_UM = 10.0
# 新序列相邻目标最大差为 210 μm。这个限值只用于本模式的单条命令防错，
# 不改变机器人速度/加速度；固定 20 cm 安全包络仍是最高优先级联锁。
MICRO_TARGET_MAX_COMMAND_UM = 250.0
MICRO_TARGET_LOCAL_RANGE_UM = 1000.0
MICRO_TARGET_MATRIX_MAX_CONDITION = 10.0
MICRO_TARGET_LIMIT_WINDOW = 5
MICRO_TARGET_LIMIT_MIN_SIGN_CHANGES = 3
MICRO_TARGET_LIMIT_IMPROVEMENT_RATIO = 0.20
MICRO_TARGET_LARGE_REVERSE_COMMAND_UM = 10.0
MICRO_TARGET_LARGE_REVERSE_COUNT = 2
MICRO_TARGET_VISION_LOSS_COUNT = 2

# 比例增益。操作者第一版要求固定 Kp = 1.0，这里做成可配置以便离线重调，
# 但运行时不启用任何自适应。
MICRO_LOOP_KP = 1.0
# 纯安全限幅：单步修正量的绝对值上限（μm）。100 μm 远小于 200 mm 相对工作区，
# 也在机械臂正常微动能力之内，只用于防止符号写反时一路狂奔。
MICRO_LOOP_MAX_CORRECTION_UM = 100.0
# 命令死区（μm）。**这不是调参，是必需的**：robot.validate_trajectory 拒绝
# length <= 1e-6 m（“两点位置重合”），而收敛时 |error| 可能只有 1 μm，
# Kp=1 时命令 1 μm，正好撞在那条硬线上——那会在一个目标即将成功的时刻
# 把整轮弄死。低于死区时不发运动，但仍照常录前段/后段、照常测量、照常判收敛。
MICRO_LOOP_MIN_COMMAND_UM = 2.0

# 迭代终止参数。
# 迭代上限 6：本轮定位是"10 分钟以内快速判断 5/20/50 μm 三个数量级"，
# 不是精确标定最小分辨率。配合下面的 STALL_PATIENCE，
# 走不动的目标会在 3 次内被点名，不会为了硬凑 5 μm 而磨满迭代。
MICRO_LOOP_MAX_ITER = 6
MICRO_LOOP_POSITION_TOL_UM = 3.0
# "连续几次没有明显改善就判 STALLED"。这条让 5 μm 档的目标早退：
# Kp=1.0 时正常闭环 1–2 步就到位；即使有效增益只有 0.3，
# 50 μm 目标的误差序列也是 50 → 35 → 24.5 → 17，每次改善远大于 1 μm，
# 所以这条门不会误伤"收敛得慢但确实在收敛"的目标。
MICRO_LOOP_STALL_PATIENCE = 3
MICRO_LOOP_STALL_MIN_IMPROVE_UM = 1.0
# absolute = 永远用 POSITION_TOL_UM；noise_relative = 放宽到
# max(POSITION_TOL_UM, NOISE_SIGMA_MULT · 本组实测测量噪声)。
# 若一次闭环测量的不确定度本身就有 4 μm，“误差 ≤ 3 μm”只是在筛噪声，不是判收敛。
MICRO_LOOP_POSITION_TOL_MODE = "noise_relative"
MICRO_LOOP_NOISE_SIGMA_MULT = 2.0
MICRO_LOOP_CONVERGE_COUNT = 3
# 判 CONVERGED 之前还必须确认"确实朝目标方向累计走过幅值的这个比例"。
# 为什么需要它：当有效容差已经和幅值同量级时（5 μm 档很可能如此），
# "误差落在容差内"几乎不构成证据——机器人一步不走，误差恰好就是幅值本身，
# 也满足 |error| ≤ tol_eff。没有这道门，5 μm 档会把"完全没动"报成收敛。
# 门槛还会再抬到本组噪声底之上（σ 以下的位移与没动统计上不可分）。
# 0.5 的含义：至少要看到走了一半路程，才承认这是朝目标逼近而不是噪声。
MICRO_LOOP_MIN_DIRECTIONAL_FRACTION = 0.5

# 单次迭代的时间结构：一段连续录制 = 前段 PRE + (运动与稳定) + 后段 POST。
# 只录一段而不是两段，这样过冲的瞬态才在片子里。
# 前/后段的长度**不影响测量精度**：在线只分析窗口末尾的
# MICRO_LOOP_MEASURE_FRAMES 帧，而在"只分析尾部"的架构下，
# 后段是唯一决定"测到的是停稳后位置"的部分，前段只负责让运动落在窗口内。
# 后段 0.25 s ≈ 33 帧，是 16 帧分析窗的 2 倍余量；前段 0.15 s 足够覆盖开窗延迟。
MICRO_LOOP_PRE_SECONDS = 0.15
MICRO_LOOP_POST_SECONDS = 0.25

# ---- 在线分析的帧预算（决定整轮运行时长，是本轮最关键的取舍） ----
# 实测：全幅 1936×1464 跑一次 findChessboardCornersSB 要 163–168 ms（本机实测
# 902 帧，100% 识别成功）。所以在线时长几乎完全等于"分析多少帧"，与等待时间无关：
# 一段 1.3 s 的窗口有 172 帧，逐帧跑要 28 s，而机械等待只占其中 1.3 s（4%）。
# 结论：**只分析窗口末尾的固定帧数**——"运动后位置"就在那里，前面的瞬态帧
# 不进在线路径（窗口照录不误，RAW 全量保留，事后可离线复查）。
# 实测逐帧视觉抖动约 1.0 μm，N 帧中位数的测量不确定度 ≈ 1.25 × 1.0 / √N：
#   N=16 → 0.32 μm，比 5 μm 目标细 15 倍，单步分析 2.6 s。
MICRO_LOOP_MEASURE_FRAMES = 16
# 取"最后这段时间内的合格帧"的中位数作为本轮稳定位置。
# 用时间戳而不是"最后 N 帧"：相机实测会零星丢帧，“最后 N 帧”对应的真实时长会变。
# 窗口要宽到能装下 MEASURE_FRAMES（16 帧 ≈ 0.12 s），0.20 s 留了约 1.7 倍余量。
MICRO_LOOP_TAIL_WINDOW_S = 0.20
# 尾部窗口里至少要有多少合格帧，本轮测量才算可用。
# 低于 MEASURE_FRAMES 是给"个别帧被接受判据剔掉"留余量，不是放宽门禁。
MICRO_LOOP_MIN_TAIL_FRAMES = 12

# 参考片段与静止基线时长。
MICRO_LOOP_REFERENCE_SECONDS = 1.0
# 建立组零点时扫描的帧数。零点只需要"足够好的中位数"，不需要整段：
# 24 帧的中位数不确定度约 0.26 μm，远好于测量本身，而整段扫 2 遍要几分钟。
# 建零点时扫的帧数（**两遍**，见 prime：一遍挑 medoid 帧，一遍让正式 tracker 锁存）。
# 取 20 而不是 24：参考帧只用来挑 medoid 和定零点中位数，多扫 4 帧对这两件事
# 都没有可测量的改善，却要在 6 个闭环组 + 探针 + 静止基线上各多花
# 4 × 0.165 × 2 ≈ 1.3 s。必须**大于** MICRO_LOOP_MEASURE_FRAMES——
# 零点的不确定度不该比单次测量本身还差，这条不变量由配置校验守着。
MICRO_LOOP_REFERENCE_FRAMES = 20
MICRO_LOOP_STATIC_SECONDS = 5.0
# 静止基线只分析其中均匀分布的这么多帧。5 s × 132.23 fps = 661 帧，
# 全扫要 110 s；抽 40 帧覆盖同样长的时间跨度，统计量足够，耗时 6.6 s。
MICRO_LOOP_STATIC_ANALYZE_FRAMES = 40

# ---- 机械臂完全断电的相机-only 静止测试 ----
# 使用与闭环静止基线相同的 5 s 窗口和 40 帧跨段分析。重复三次是为了避免把一次
# 偶发桌面扰动误判成视觉底噪；该模式不会创建任何机器人或 RTDE 接口。
OFFLINE_STATIC_SECONDS = MICRO_LOOP_STATIC_SECONDS
OFFLINE_STATIC_REPEATS = 3
# 单次 RAW 之外额外要求的磁盘余量；每次分析完成后立即删除该次 RAW。
OFFLINE_STATIC_DISK_RESERVE_GB = 2.0

# 相机进程预分配缓冲能容纳的最长窗口（秒）。当前每次多目标迭代会在活动窗口内
# 分析命令前 16 帧（约 2.64 s），再发送命令并等待稳定；7.5 s 可覆盖这段分析延迟
# 与全部既有运动超时，约预留 2.64 GiB。它不改变任何机器人速度/加速度。
# 本机 15.77 GiB 物理内存，而 np.empty 在 Windows 上是**惰性提交**的——
# 只有真正写进去的帧才占用物理页，所以典型窗口（约 2 s / 0.7 GiB 实际提交）
# 与预留上限是两回事，预留给大一点几乎不花钱。反过来预留不够就等于丢数据。
# 窗口内各阶段的最坏耗时之和约 5.6 s（见上面几个超时），留了约 0.4 s 余量。
MICRO_LOOP_MAX_WINDOW_SECONDS = 7.5
# 尾部窗口自身的静态性门禁。片段最后 TAIL_WINDOW_S 内的标准差超过这个值，
# 说明这一轮测的不是"停稳后的位置"而是"还在动"，该轮测量不可用。
MICRO_LOOP_TAIL_SIGMA_MAX_UM = 2.0

# 逐帧接受判据。棋盘格**索引错位一格**会让所有点位移同一个向量（约 3 mm），
# estimateAffinePartial2D 会把它拟合成残差≈0、内点率 1.0、质量分≈1.0 的纯平移——
# 与"真的移动了 3 mm"在返回字典里完全无法区分。所以必须有一组独立的物理门禁。
MICRO_LOOP_MAX_PLAUSIBLE_SHIFT_UM = 2000.0
# 迭代间连续性联锁：单次迭代最大合法变化就是限幅加上漂移，超过 3× 限幅一定是检测问题。
MICRO_LOOP_MAX_ITER_JUMP_UM = 300.0
MICRO_LOOP_QUALITY_MIN = 0.5
MICRO_LOOP_ANGLE_MAX_DEG = 1.0
MICRO_LOOP_SCALE_TOL = 0.01

# 相机与预览。全幅 1936×1464，不裁剪、不缩放——实测裁剪会把识别率从 100% 打到 35%。
MICRO_LOOP_CAMERA_WARMUP_SECONDS = 1.0
MICRO_LOOP_PREVIEW_FPS = 10.0
MICRO_LOOP_PREVIEW_MAX_WIDTH = 700
MICRO_LOOP_PROCESS_EVERY_N_FRAMES = 1
# 画质指标（亮度/模糊/过暗/过曝，4 次全幅遍历，实测约 40 ms/帧）在闭环里没有用处：
# 图像质量只用于显示，绝不参与控制。关掉它不影响识别链路——识别部分
# 与 camera.preprocess_frame 逐字等价（同样的高斯模糊核、同一个 CheckerboardTracker）。
MICRO_LOOP_COMPUTE_METRICS = False

# 轴向 / 符号探针。视觉 +x/+y 与机器人 base X/Y 的对应关系（含符号）无法从代码推出，
# 而符号反了会让误差每步翻倍直冲限幅，所以启动时必须实测一次。
MICRO_LOOP_PROBE_UM = 1000.0
MICRO_LOOP_PROBE_REFERENCE_SECONDS = 1.0
# 探针允许 SB 检测偶发回落，但绝不接受 legacy 与参考混用：通过扩大探针专用
# 样本池，在保持“至少 12 帧、标准差 ≤2 μm”不变的前提下获得足够合格帧。
# 闭环单步仍使用上面的 16 帧 / 0.20 s，不受这些探针专用参数影响。
MICRO_LOOP_PROBE_REFERENCE_FRAMES = 32
MICRO_LOOP_PROBE_MEASURE_FRAMES = 64
MICRO_LOOP_PROBE_TAIL_WINDOW_S = 0.50
# 探针专用的运动超时。1 mm 在 MICRO_LOOP_SPEED_MM_S=1 mm/s 下要走 1.0 s，
# 加上加减速约 1.2 s——若沿用给微动的 MICRO_LOOP_STEP_TIMEOUT_S=1.0 s，
# 探针会**每一次都超时**，然后退化成"靠固定 settle 猜"，而探针失败是具名中止。
# 微动的超时和探针的超时必须分开，因为两者的移动量差 10~200 倍。
MICRO_LOOP_PROBE_TIMEOUT_S = 3.0
# 每次探针运动后的视觉测量若不合格，保持机器人当前位置、不再发送 MOVE，
# 最多重新等待稳定并采集这么多次。初测不计入这个数字。
MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS = 5
# micro_closed_loop 启动时记录的 TCP 是不可移动的安全包络中心。任何目标点和
# 运动中实测 TCP 到该中心的三维距离都不得超过 20 cm。
MICRO_LOOP_FIXED_SAFETY_RADIUS_M = 0.20
# 增益门禁：|g| > 2 有过冲振荡风险；|g| < 0.3 表示 20 次迭代也走不完 50 μm。
MICRO_LOOP_GAIN_MIN = 0.3
MICRO_LOOP_GAIN_MAX = 2.0
# 正向与反向增益的相对差异上限。回差大到这个程度时闭环不是良定的。
MICRO_LOOP_PROBE_HYSTERESIS_RATIO = 0.2
# 交叉耦合只在超过该比例时告警并记录，不补偿（命令律仍是 Kp·error）。
MICRO_LOOP_CROSS_COUPLING_WARN = 0.3

# 运行期联锁。探针通过仍可能发散，所以运行中还要盯。
# 每条修正的实测位移与指令方向相反，连续出现该次数即中止。
MICRO_LOOP_SIGN_FLIP_ABORT_COUNT = 2
# 本组累计 |命令| 超过该值即中止。这一条必要，因为 validate_trajectory 的
# 相对工作区**每次调用都以当前位姿重新锚定**（robot.py:215-220），
# 结构上无法察觉缓慢棘轮式走位。
MICRO_LOOP_CUM_COMMAND_ABORT_UM = 2000.0
# 漂移守卫已上移到"按轴共用零点"的 MICRO_LOOP_AXIS_DRIFT_*（见下面）。
# 上一版按"本组幅值 + 20 μm"设门是个真实缺陷：+A 与 −A 共用参考时，
# −A 的第一步必须合法地走约 2A 的行程，会被误判成棘轮走位而中止。
# 保留这两个常量只为了让离线分析脚本还能读到旧名，运行路径不再使用。
MICRO_LOOP_DRIFT_WARN_UM = 5.0
MICRO_LOOP_DRIFT_ABORT_UM = 20.0

# 极限环判据。全部可配，便于离线重调而不用改代码。
# 窗口从 6 收到 3：判据要比较"最近一窗"与"前一窗"，所以可用的迭代上限
# 必须 ≥ 2×窗口。迭代上限由本轮定位钉死在 6 次，窗口就只能 ≤ 3，
# 否则 LIMIT_CYCLE 这个状态在整轮实验里**永远不会被判定出来**。
# 3 帧窗口下的签名是"连续 3 次误差换号且幅值不收缩"——
# 配合 CYCLE_ABS_KEEP=0.7 仍能区分“收敛型振荡”（环路增益 1.5 时三拍衰到 0.125）
# 与“真极限环”，不会把正在收敛的振荡误报成极限环。
MICRO_LOOP_CYCLE_WINDOW = 3
MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES = 2
MICRO_LOOP_CYCLE_PTP_KEEP = 0.5
MICRO_LOOP_CYCLE_ABS_KEEP = 0.7

# ---- 开环微动模式（seq / alt） ----
# 闭环之前先跑开环，用固定命令序列量出"命令 → 实际位移"的原始响应。
# 两者指标**严格分开**：闭环的最终残差是"逼近能力"，开环的响应统计才是
# "最小可靠微动"。绝不能拿闭环残差当最小微动单位。
#
# seq（连续同向）：+Δ +Δ +Δ −Δ −Δ −Δ，共 6 步。
#   看微小命令是否真的产生运动、连续同向能否累积、正负是否对称、
#   以及有没有"前两步不动、第三步突然跳"的死区/积累。
# alt（频繁换向）：+Δ −Δ +Δ −Δ +Δ −Δ，共 6 步。
#   模拟未来 SFC 可能出现的高频正负修正，重点看换向死区、回差、
#   摩擦、方向延迟，以及是否明显劣于连续同向。
# 两种模式每个轴每个档位都跑，步数对称，便于直接对比。
MICRO_LOOP_OPEN_MODES: Final[tuple[str, ...]] = ("seq", "alt")
MICRO_LOOP_OPEN_SEQ_POSITIVE_STEPS = 3
MICRO_LOOP_OPEN_SEQ_NEGATIVE_STEPS = 3
MICRO_LOOP_OPEN_ALT_STEPS = 6
# 单步"命令 → 实测位移"的合格判据：实测位移落在 [lo, hi] 内算 VALID。
# 与机器人侧断言（ACHIEVED_*）是两套：这里看的是视觉，那里看的是编码器。
MICRO_LOOP_OPEN_MIN_RATIO = 0.5
MICRO_LOOP_OPEN_MAX_RATIO = 1.5
MICRO_LOOP_OPEN_ZERO_UM = 0.5
MICRO_LOOP_OPEN_JUMP_UM = 200.0

# 全局漂移守卫（相对**本轴零点**，整个轴的所有块共用一个基准）。
# 本轮各块都是"走出去再走回来"（seq 与 alt 都以净位移 ≈0 结束），
# 所以不需要按幅值设门——按幅值设门恰好是上一版的缺陷：−A 目标第一步
# 合法地跨过 2A 的行程，会被误判成漂移。这里只守"跑飞了"这一件事。
MICRO_LOOP_AXIS_DRIFT_WARN_UM = 300.0
MICRO_LOOP_AXIS_DRIFT_ABORT_UM = 1500.0

# 运动参数。1.0 mm/s + 0.10 m/s² 时，加速段本身就覆盖约 5 μm，
# 因此 5 μm 档全程处于加减速斜坡内（这是被测现象，不是缺陷）。
MICRO_LOOP_SPEED_MM_S = 1.0
MICRO_LOOP_ACCELERATION_M_S2 = 0.10

# 基于位置的稳定判据，取代 robot.motion_in_progress()。
# 后者带固定 200 ms 宽限期，是 5 μm 运动全程（约 14 ms）的 14 倍，对微动零信息量。
# 窗口取 5 μm 而不是开环版的 1 μm：CB3 的 actual_tcp_pose 由关节编码器换算，
# 笛卡尔分辨率本身可能就有几 μm，1 μm 窗口可能永远无法满足，
# 那会让每次迭代白等 10 s 再抛错。超时只降级为"按固定时间等一等"，
# 真正的有效性由**视觉片段自身的窗内标准差**判定——闭环本就该以视觉为准。
MICRO_LOOP_STABLE_WINDOW_MS = 100.0
MICRO_LOOP_STABLE_WINDOW_UM = 5.0
MICRO_LOOP_STABLE_SPEED_MM_S = 0.01
MICRO_LOOP_STABLE_HOLD_SECONDS = 0.3
# 独立 STABILIZE 命令用的超时。它在录制窗口之外，长短不影响内存占用。
MICRO_LOOP_STABLE_TIMEOUT_S = 3.0
# 运动前/后的稳定等待超时。它们**在录制窗口之内**，所以必须短：
# 窗口越短，相机进程需要的内存缓冲越小，见 MICRO_LOOP_MAX_WINDOW_SECONDS。
# 100 μm 以 1 mm/s 走完约 0.1 s（加加减速约 0.35 s），0.8 / 1.2 s 是 2–12 倍余量。
MICRO_LOOP_PRE_STABLE_TIMEOUT_S = 0.8
MICRO_LOOP_POST_STABLE_TIMEOUT_S = 0.8
# 稳定等待超时后的降级等待。真正的有效性由视觉片段尾窗自身的标准差判定，
# 编码器稳定性只是旁证，所以降级不是"放弃判断"。
MICRO_LOOP_MOTION_SETTLE_SECONDS = 0.5
# motion_in_progress() 的轮询上限。同样在窗口之内。100 μm 以 1 mm/s 走完约 0.35 s
# （含加减速），1.0 s 是它的约 3 倍。这个值同时是"最坏窗口"的主要构成项，
# 而最坏窗口 × 窗口数决定了整轮的绝对上界，所以不能随手放大。
MICRO_LOOP_STEP_TIMEOUT_S = 1.0

# ---- 时长提示线（**只提示，不中止**）----
# 这个数**不是截止时间**，程序不会因为它跳过任何实验步骤。
#
# 上一版把它当成硬预算来用：每开一个窗口前外推总时长，投影超过就抛
# TimeBudgetExceeded 安全收尾，未跑的目标记 TIME_BUDGET_EXCEEDED。
# 那会造出一种最坏的结果——X 全跑完了、Y 的闭环还没轮到，就因为"到点了"
# 直接结束，而机器人、相机、磁盘当时全都正常。**这条策略已删除。**
#
# 现在的原则：只要机器人/相机/磁盘正常、没有触发任何真实安全异常，
# 既定的全部实验（2 轴 × 3 档 × seq/alt 开环 + 12 个闭环目标）就必须跑完。
# 12~15 分钟是可接受的；真正允许中止的只有通信异常、相机异常、超出工作
# 空间、异常累计漂移、方向异常、磁盘不足、内存不足这类真实故障，
# 以及单个目标自身的 MAX_ITER / STALLED / LIMIT_CYCLE（只结束那个目标，
# 继续下一个目标）。
#
# 保留这个常数的用途只有两个：启动前把预计时长显示给操作者，以及运行中
# 超过它时打一条提示，好让"今天比平时慢"这件事被看见。
#
# 取值 900 s（15 分钟）而不是 600 s：按规定规模（12 个闭环目标）算出来，
# 正常运行就要约 11.3 分钟，取 600 s 会让**每一轮**都触发提示，提示就变成
# 噪声，真正异常的那一次反而看不见了。取 15 分钟只在明显跑偏时才说话。
MICRO_LOOP_TIME_NOTICE_S = 900.0
MICRO_LOOP_RETURN_SPEED_MM_S = 5.0

# 机器人侧位移断言（复用开环版已验证的判据），与视觉侧的 measured_um 并排，
# 就能把"控制器忽略了命令"与"机械柔性吸收掉了"分开。
MICRO_LOOP_ACHIEVED_MIN_RATIO = 0.5
MICRO_LOOP_ACHIEVED_MAX_RATIO = 1.5
MICRO_LOOP_ACHIEVED_SLACK_UM = 2.0
MICRO_LOOP_ACHIEVED_FLOOR_UM = 2.0

# 磁盘保护。检查的是**原始数据所在卷**（见 MICRO_LOOP_RAW_ROOT）。
MICRO_LOOP_MIN_FREE_GB_START = 20.0
MICRO_LOOP_MIN_FREE_GB_CONTINUE = 10.0
MICRO_LOOP_TRASH_QUOTA_GB = 2.0

# ---- 原始数据存放策略 ----
# 每个实验组运行期间完整保留该组 RAW；该组的 CSV/JSON/证据图落盘后，立即删除
# 该组的大体积视频载荷，再进入下一组。异常退出时另有整轮兜底清理。因此磁盘峰值
# 是“最大单组 + 余量”，不是整轮 100+ GiB 的累计量。
MICRO_LOOP_RAW_ROOT: Final[Path] = Path("D:/UR10_micro_raw")
# True = 当前实验组内全量归档，供在线分析和该组汇总使用。
MICRO_LOOP_KEEP_ALL_RAW = True
# 每组结果文件落盘后，立即删除该组的 .raw/.avi/.mp4/.mkv。
MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP = True
# 用户停止与正常/异常结束都删除本次 run_id 下的 .raw/.avi/.mp4/.mkv 大文件。
# 这是异常/停止路径的兜底；只清本次任务目录，不碰以往运行，也保留旁车文件。
MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT = True
# 单组暂存时要求预留的余量（GiB）。低于它就算"放得下"也不开始——
# 中途窗口因稳定超时被拖长会额外吃空间，留余量比事后补救便宜。
MICRO_LOOP_KEEP_ALL_RESERVE_GB = 15.0

# ---- 内存保护 ----
# 相机进程为窗口预分配一块缓冲（MICRO_LOOP_MAX_WINDOW_SECONDS 决定虚拟预留上限，
# Windows 上 np.empty 是惰性提交的，真正吃内存的是实际写进去的帧数）。
# 启动前检查可用物理内存，明显不足就拒绝启动；预期占用同时打进日志与 README。
MICRO_LOOP_MIN_FREE_RAM_GB = 1.5

# 每个目标最多保存几张全分辨率 PNG 证据帧（绝不用 JPEG）。
# 本轮定位是"快速判断"，所以默认只留 1 张；RAW 会在组末删除，证据图与
# 逐帧 CSV 是长期保留、用于追查识别质量的材料。
MICRO_LOOP_EVIDENCE_MAX = 1
# 收敛后是否再录一段零命令静止片段做验证。这是唯一能把"真不动点"与
# "运气好落进去"分开的动作，成本只有一段片段。
MICRO_LOOP_VERIFY_EFFORT = True
# 验证不通过时最多重开几次闭环。重开不会重置迭代预算——
# 总迭代数始终由 MICRO_LOOP_MAX_ITER 封顶，所以验证不会把运行时长拖长。
MICRO_LOOP_VERIFY_MAX_ATTEMPTS = 3

# 相机就绪前的数据量估算用全幅尺寸；就绪后一律以实际帧尺寸为准。
MICRO_LOOP_FULL_FRAME_WIDTH = 1936
MICRO_LOOP_FULL_FRAME_HEIGHT = 1464


# 计划最大包络人工检查使用固定低速参数；检查完成后不会自动进入批量实验。
BOUNDARY_CHECK_SPEED_MM_S = 10.0
BOUNDARY_CHECK_ACCELERATION_M_S2 = 0.05
BOUNDARY_CHECK_DWELL_SECONDS = 2.0

# 本段给三个相对运动实验定义“以本次实际 A 点为中心”的动态工作区半径。
# 输入：机器人连接后读取的当前 TCP；输出：本次运行 X/Y/Z 各自 [A-0.20, A+0.20] m 的边界。
# 实验作用：相对运动不再受示例绝对坐标边界影响，同时仍限制目标不能偏离实际起点超过 20 cm。
ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M = 0.20

# 本段给旧的绝对 A/B/C 轨迹保留固定 TCP 工作区边界。
# 输入：绝对轨迹中的所有 TCP 点；输出：通过检查或拒绝执行。
# 实验作用：它不用于三个相对运动实验；相对模式使用上面的“实际 A 点 ±20 cm”动态边界。
# 这是额外保险，不等于 UR 控制器自身安全设置，也无法识别桌面、夹具和电缆。
WORKSPACE_LIMITS_M: Final[dict[str, tuple[float, float]]] = {
    "x": (-0.80, 0.20),
    "y": (-0.80, 0.80),
    "z": (0.10, 1.20),
}

# 本段防止轨迹点之间出现离谱跳变。
# 输入：A/B/C 中相邻点的空间距离；输出：允许生成轨迹或拒绝真机运动。
# 实验作用：相邻点距离过长通常意味着单位写错或误抄示教坐标。
MAX_SEGMENT_LENGTH_M = 0.30

# 本段判断机器人当前 TCP 是否已经足够接近安全起点。
# 输入：当前 TCP 位置和 POINT_A 的前三项；输出：是否允许进入后续轨迹。
# 实验作用：避免机器人实际不在起点，却直接执行以 A 点为起点设计的轨迹。
START_POSE_TOLERANCE_M = 0.005

# 本段给每段机器人运动设置兜底等待上限。
# 输入：robot.py 等待运动完成的循环；输出：正常完成或超时报错。
# 实验作用：避免控制器异常、网络异常或轨迹无法完成时程序无限等待。
ROBOT_MOTION_TIMEOUT_S = 60.0


# =============================================================================
# 10. 控制器预留接口
# =============================================================================

# 本段说明当前是否启用在线控制。
# 输入：控制器模式和执行器模式；输出：main.py/robot.py 是否允许进入对应控制流程。
# 实验作用：当前代码主体是预实验和开环轨迹，SFC 只保留名字和检查口，还没有真正实现在线修正。
CONTROL_MODE = "disabled"
EXECUTOR_MODE = "open_loop"

# 本段定义控制模式的合法集合。
# 输入：CONTROL_MODE/EXECUTOR_MODE 字符串；输出：validate_config() 的合法性判断。
# 实验作用：如果误选 sfc 或 servo，当前版本会在接触机器人前明确停止。
VALID_CONTROL_MODES: Final[set[str]] = {"disabled", "sfc"}
VALID_EXECUTOR_MODES: Final[set[str]] = {"open_loop", "servo"}


# =============================================================================
# 11. 完整实验的时间和进程参数
# =============================================================================

# 本段定义完整实验的时间结构。
# 输入：主进程发出的 start_event；输出：相机和机器人日志中的 baseline、motion、post 三段数据。
# 实验作用：先静止记录建立噪声/静止基线，再运动激励，最后观察残余振动衰减。
PRE_RECORD_SECONDS = 3.0

POST_RECORD_SECONDS = 5.0

# 本段控制多进程启动阶段的等待上限。
# 输入：相机进程和机器人进程的 ready 信号；输出：继续实验或判定启动失败。
# 实验作用：避免某个子进程卡住时，主程序还误以为完整实验已经同步开始。
WORKER_READY_TIMEOUT_S = 20.0

# 本段控制多进程结束阶段的等待上限。
# 输入：实验结束后各子进程的退出状态；输出：正常收尾或强制终止异常进程。
# 实验作用：尽量让日志正常写完，同时避免异常进程长期占用相机或机器人连接。
WORKER_JOIN_TIMEOUT_S = 8.0

# 本段限制进程间记录队列的容量。
# 输入：相机/机器人持续产生的记录；输出：等待写入 run_log.txt 的队列。
# 实验作用：磁盘写入变慢时队列不能无限增长，否则会逐渐吃完内存。
RECORD_QUEUE_MAXSIZE = 20000
ERROR_QUEUE_MAXSIZE = 100


# =============================================================================
# 12. 离线分析参数
# =============================================================================

# 本段选择离线分析读取哪一次实验记录。
# 输入：run_log.txt 或 vision_results.txt；输出：analyze.py 用于计算 RMS、频谱和恢复时间的数据源。
# 实验作用：None 会自动选择 outputs 下最新记录；填具体 Path 可以固定复查某一次实验。
ANALYSIS_FILE: Path | None = None

# 本段选择分析阶段使用哪套视觉结果。
# 输入：每帧 JSON 中的 circles/checkerboard 结果；输出：用于统计和画图的一条位移序列。
# 实验作用：前期 compare 会同时保存两套结果，分析时可以分别选用，判断问题出在识别方法还是实验本身。
ANALYSIS_VISION_METHOD = "checkerboard"

# 本段选择振动分析观察哪个方向。
# 输入：视觉算法输出的 dx、dy；输出：x、y 或合位移 magnitude 时间序列。
# 实验作用：若相机坐标轴已经和实验方向对齐，可直接看 x/y；方向不确定时先看 magnitude 更稳妥。
ANALYSIS_AXIS = "x"
VALID_ANALYSIS_AXES: Final[set[str]] = {"x", "y", "magnitude"}

# 本段选择“哪一段时间”作为主要分析对象。
# 输入：完整时间序列和实验事件时间；输出：拿去算 RMS、主频、频带能量的时间窗口。
# 实验作用：baseline 看静止噪声，motion 看整个运动过程，steady_motion 裁掉运动前后各一段以粗略聚焦匀速段。
ANALYSIS_PRIMARY_WINDOW = "motion"
VALID_ANALYSIS_PRIMARY_WINDOWS: Final[set[str]] = {
    "full",
    "baseline",
    "motion",
    "steady_motion",
    "post",
}
STEADY_MOTION_TRIM_FRACTION = 0.20

# 本段选择 analyze.py 如何把一长串数据切成“静止-运动-静止”的实验片段。
# 输入：EVENT 时间戳、ROBOT 速度记录、VISION 位移曲线；输出：motion_001、static_001、steady_motion_001 等窗口。
# 实验作用：auto 会优先使用机器人真实速度分段，适合反复启停和多段轨迹；没有机器人速度时再退回事件或视觉位移变化。
ANALYSIS_SEGMENTATION_SOURCE = "auto"
VALID_ANALYSIS_SEGMENTATION_SOURCES: Final[set[str]] = {
    "auto",
    "robot_speed",
    "events",
    "vision_velocity",
}

# 本段定义“机器人到底算不算正在运动”的速度迟滞阈值，单位 m/s。
# 输入：ROBOT 记录中的 actual_tcp_speed；输出：运动段起止时间。
# 实验作用：开阈值高、关阈值低，可以避免速度在零附近轻微抖动时把一段静止误切成很多小段。
ROBOT_MOTION_ON_SPEED_M_S = 0.002
ROBOT_MOTION_OFF_SPEED_M_S = 0.001

# 本段定义“运动段里面哪一小段更像匀速段”。
# 输入：TCP 速度和由速度变化估算的加速度；输出：steady_motion_* 窗口。
# 实验作用：速度基本稳定、加速度接近零时才作为匀速段；如果判断失败，仍会用 STEADY_MOTION_TRIM_FRACTION 兜底裁剪。
ROBOT_STEADY_ACCELERATION_M_S2 = 0.02
ROBOT_STEADY_SPEED_RELATIVE_TOLERANCE = 0.08

# 本段清理过短的自动分段。
# 输入：自动识别出的候选静止段/运动段；输出：去掉明显过短、缺乏分析意义的小片段。
# 实验作用：避免一次采样毛刺或极短停顿被当成一个完整工况写进摘要。
SEGMENT_MIN_MOTION_SECONDS = 0.20
SEGMENT_MIN_STATIC_SECONDS = 0.20
SEGMENT_MERGE_GAP_SECONDS = 0.10

# 本段只在没有机器人速度且要求视觉兜底分段时使用。
# 输入：视觉位移的一阶变化速度；输出：粗略运动/静止判断。
# 实验作用：视觉兜底只能辅助离线复查，正式分段仍建议依赖机器人速度或明确事件。
VISION_MOTION_VELOCITY_FACTOR = 6.0

# 本段选择如何从位移曲线中去掉慢变化趋势。
# 输入：原始位移序列；输出：更接近“振动分量”的去趋势序列。
# 实验作用：linear 适合直线匀速段，savgol 适合缓慢弯曲趋势，highpass 适合明确只关心某频率以上振动。
DETREND_METHOD = "linear"
VALID_DETREND_METHODS: Final[set[str]] = {"linear", "savgol", "highpass"}

# 本段只在 DETREND_METHOD="savgol" 时生效。
# 输入：窗口秒数和多项式阶数；输出：从原始曲线估计出的慢变化趋势。
# 实验作用：窗口太短会把振动也当趋势扣掉，窗口太长则可能跟不上路径缓慢弯曲。
SAVGOL_WINDOW_SECONDS = 0.50
SAVGOL_POLYORDER = 2

# 本段只在 DETREND_METHOD="highpass" 时生效。
# 输入：截止频率和滤波器阶数；输出：高通后的振动序列。
# 实验作用：截止频率不应高于你希望保留的最低振动频率，否则真实低频振动会被滤掉。
HIGHPASS_CUTOFF_HZ = 2.0
HIGHPASS_ORDER = 4

# 本段限制主频搜索范围。
# 输入：去趋势后的频谱；输出：dominant_frequency_hz。
# 实验作用：避开零频漂移，也避免在超过奈奎斯特频率的无效区里误找主频。
DOMINANT_FREQ_MIN_HZ = 1.0
DOMINANT_FREQ_MAX_HZ = 45.0

# 本段把频谱按实验关心的频带做能量汇总。
# 输入：频谱功率；输出：每个频带的能量指标。
# 实验作用：后续比较不同速度、姿态、负载时，可以比单一主频更稳定地观察某段频率能量变化。
FREQUENCY_BANDS_HZ: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
)

# 本段定义“运动结束后多久算振动恢复”。
# 输入：post 窗口中的振动幅值和 baseline 噪声水平；输出：recovery_time_s。
# 实验作用：阈值取“静止基线 RMS×倍数”和绝对阈值中的较大者；连续低于阈值一段时间才算真正恢复。
RECOVERY_BASELINE_FACTOR = 3.0
RECOVERY_ABSOLUTE_THRESHOLD_MM = 0.02
RECOVERY_HOLD_SECONDS = 0.30


# =============================================================================
# 13. 配置检查
# =============================================================================

def _check_pose(name: str, pose: list[float]) -> None:
    """
    检查机器人位姿是否能作为轨迹端点使用。

    输入：POINT_A/B/C 这类 [x, y, z, rx, ry, rz] 列表。
    输出：无返回值；发现长度错误、非数值或无穷值时直接抛错。
    实验作用：提前拦住少抄、多抄或复制异常值的位姿，避免后续轨迹检查建立在坏数据上。
    """

    import math

    if len(pose) != 6:
        raise ValueError(f"{name} 必须包含 6 个数：[x, y, z, rx, ry, rz]。")

    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pose):
        raise ValueError(f"{name} 中存在非数值或无穷值。")


def _check_vector(name: str, values: list[float] | None, length: int) -> None:
    """
    检查空间坐标标定中的向量参数是否完整。

    输入：平移向量、平面点或平面法向等一维数值列表。
    输出：无返回值；缺失、长度不对或含非法数值时抛错。
    实验作用：空间坐标一旦启用，就不能让“只填了一半的外参”继续参与计算。
    """

    import math

    if values is None or len(values) != length:
        raise ValueError(f"{name} 必须填写 {length} 个数值。")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError(f"{name} 中存在非数值或无穷值。")


def _check_matrix(name: str, values: list[list[float]] | None, rows: int, columns: int) -> None:
    """
    检查空间坐标标定中的矩阵参数是否完整。

    输入：相机内参或相机到机器人旋转矩阵。
    输出：无返回值；维度不对或含非法数值时抛错。
    实验作用：防止错误尺寸的标定矩阵进入像素到空间坐标的投影计算。
    """

    import math

    if values is None or len(values) != rows or any(len(row) != columns for row in values):
        raise ValueError(f"{name} 必须填写 {rows}×{columns} 个数值。")
    for row in values:
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in row):
            raise ValueError(f"{name} 中存在非数值或无穷值。")


def validate_config(run_mode: str | None = None) -> None:
    """
    启动前统一检查本文件是否足以支撑本次实验流程。

    输入：可选 run_mode。main.py 可用命令行参数临时覆盖文件顶部的 RUN_MODE。
    输出：无返回值；配置可用时只创建输出目录，配置明显错误时抛出 ValueError。

    实验作用：它是“进入相机/机器人/分析流程之前的门卫”。它不会连接相机，
    也不会连接机器人，只用纯 Python 检查模式名称、视觉参数、安全开关、
    机器人轨迹参数和离线分析参数是否自相矛盾。
    """

    selected_mode = run_mode or RUN_MODE

    # 本段检查所有“选择题”式配置。
    # 输入：RUN_MODE、视觉来源、视觉算法、轨迹类型、控制器模式等字符串。
    # 输出：确认每个字符串都落在合法集合内；拼写错误会在这里停止。
    # 实验作用：防止因为一个模式名拼错，程序进入错误分支或后续报出更难懂的异常。
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

    if SPATIAL_COORDINATE_MODE not in VALID_SPATIAL_COORDINATE_MODES:
        raise ValueError(
            f"未知 SPATIAL_COORDINATE_MODE={SPATIAL_COORDINATE_MODE!r}，"
            f"可选值为 {sorted(VALID_SPATIAL_COORDINATE_MODES)}。"
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

    # 本段检查会影响真机安全的流程级开关。
    # 输入：当前运行模式、控制模式、执行器模式和机器人地址。
    # 输出：确认本版代码不会把“预留接口”误当成已经实现的正式在线控制。
    # 实验作用：SFC/servo 相关名称目前只作为接口占位；真机模式下误选这些值必须提前停止。
    robot_modes = {
        "robot_connection_test",
        "robot_test",
        "experiment",
        "x_line_experiment",
        "xy_line_experiment",
        "xy_l_experiment",
        "batch_experiment",
        "micro_closed_loop",
        "boundary_check",
    }
    relative_motion_modes = {
        "x_line_experiment",
        "xy_line_experiment",
        "xy_l_experiment",
        "batch_experiment",
        "micro_closed_loop",
        "boundary_check",
    }

    if selected_mode in robot_modes and CONTROL_MODE == "sfc":
        raise ValueError(
            "当前版本只预留了 SFC 接口，尚未实现在线控制。"
            "为了避免把开环运动误当成 SFC，程序拒绝连接机器人。"
        )

    if selected_mode in {"experiment", *relative_motion_modes} and EXECUTOR_MODE != "open_loop":
        raise ValueError("当前正式/相对运动实验只实现 open_loop 轨迹执行器。")

    if selected_mode in robot_modes and not ROBOT_HOST.strip():
        raise ValueError("真机模式必须填写非空 ROBOT_HOST。")

    if ROBOT_CONNECTION_TEST_SECONDS <= 0:
        raise ValueError("ROBOT_CONNECTION_TEST_SECONDS 必须为正数。")

    if ROBOT_RTDE_CONNECT_ATTEMPTS < 1:
        raise ValueError("ROBOT_RTDE_CONNECT_ATTEMPTS 必须至少为 1。")
    if ROBOT_RTDE_CONNECT_RETRY_DELAY_S < 0:
        raise ValueError("ROBOT_RTDE_CONNECT_RETRY_DELAY_S 不能为负数。")

    if ROBOT_RECORD_HZ <= 0:
        raise ValueError("ROBOT_RECORD_HZ 必须为正数。")

    if ROBOT_EXPERIMENT_MIN_SPEED_MM_S <= 0:
        raise ValueError("ROBOT_EXPERIMENT_MIN_SPEED_MM_S 必须为正数。")

    if ROBOT_EXPERIMENT_MAX_SPEED_MM_S < ROBOT_EXPERIMENT_MIN_SPEED_MM_S:
        raise ValueError("ROBOT_EXPERIMENT_MAX_SPEED_MM_S 不能小于最小速度。")

    if ROBOT_EXPERIMENT_MAX_SEGMENT_MM <= 0:
        raise ValueError("ROBOT_EXPERIMENT_MAX_SEGMENT_MM 必须为正数。")

    if ROBOT_EXPERIMENT_ACCELERATION_M_S2 <= 0:
        raise ValueError("ROBOT_EXPERIMENT_ACCELERATION_M_S2 必须为正数。")

    if ROBOT_EXPERIMENT_STOP_SPEED_MM_S <= 0:
        raise ValueError("ROBOT_EXPERIMENT_STOP_SPEED_MM_S 必须为正数。")

    if ROBOT_RETURN_WARNING_MM <= 0:
        raise ValueError("ROBOT_RETURN_WARNING_MM 必须为正数。")

    if ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M <= 0:
        raise ValueError("ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M 必须为正数。")

    for name, value in {
        "BATCH_STATIC_BASELINE_SECONDS": BATCH_STATIC_BASELINE_SECONDS,
        "BATCH_PRE_MOTION_SECONDS": BATCH_PRE_MOTION_SECONDS,
        "BATCH_POST_MOTION_SECONDS": BATCH_POST_MOTION_SECONDS,
        "BATCH_CAMERA_WARMUP_SECONDS": BATCH_CAMERA_WARMUP_SECONDS,
        "BATCH_PREVIEW_FPS": BATCH_PREVIEW_FPS,
        "BATCH_RECORDING_PREVIEW_FPS": BATCH_RECORDING_PREVIEW_FPS,
        "BATCH_PREVIEW_MAX_WIDTH": BATCH_PREVIEW_MAX_WIDTH,
        "BOUNDARY_CHECK_SPEED_MM_S": BOUNDARY_CHECK_SPEED_MM_S,
        "BOUNDARY_CHECK_ACCELERATION_M_S2": BOUNDARY_CHECK_ACCELERATION_M_S2,
        "BOUNDARY_CHECK_DWELL_SECONDS": BOUNDARY_CHECK_DWELL_SECONDS,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数。")

    if len(BATCH_VIDEO_CODEC) != 4:
        raise ValueError("BATCH_VIDEO_CODEC 必须是 4 个字符的 FourCC。")

    if not 0.0 <= BATCH_MAX_MISSING_RATIO <= 1.0:
        raise ValueError("BATCH_MAX_MISSING_RATIO 必须在 0 到 1 之间。")

    if BATCH_MAX_CONSECUTIVE_MISSING_FRAMES < 1:
        raise ValueError("BATCH_MAX_CONSECUTIVE_MISSING_FRAMES 必须至少为 1。")

    # 本段检查一键 XY 微动闭环能力测试的参数。
    # 输入：目标幅值、增益、死区、终止判据、稳定判据、探针门禁和磁盘配额。
    # 输出：参数自洽时继续；死区过小、阈值颠倒或磁盘配额顺序写反时抛错。
    # 实验作用：这些值直接决定“能否逼近 5 μm”与“是否形成极限环”的结论是否成立，
    # 而且它们全部是软件安全的依据，写错时必须在接触硬件之前停止。
    if not MICRO_LOOP_AMPLITUDES_UM:
        raise ValueError("MICRO_LOOP_AMPLITUDES_UM 不能为空。")
    if any(
        not isinstance(level, int) or level < 1 for level in MICRO_LOOP_AMPLITUDES_UM
    ):
        raise ValueError("MICRO_LOOP_AMPLITUDES_UM 必须都是不小于 1 的整数（μm）。")
    if list(MICRO_LOOP_AMPLITUDES_UM) != sorted(set(MICRO_LOOP_AMPLITUDES_UM)):
        raise ValueError("MICRO_LOOP_AMPLITUDES_UM 必须是从小到大且不重复的幅值序列。")
    if set(MICRO_LOOP_AXES) - {"X", "Y"} or not MICRO_LOOP_AXES:
        raise ValueError('MICRO_LOOP_AXES 只能包含 "X" 和 "Y"。')

    if tuple(MICRO_TARGET_RELATIVE_STEPS_UM) != (100, 150, 210, -100, -150, -210):
        raise ValueError(
            "MICRO_TARGET_RELATIVE_STEPS_UM 必须保持为 "
            "(100, 150, 210, -100, -150, -210)。"
        )
    cumulative: list[int] = []
    position = 0
    for step in MICRO_TARGET_RELATIVE_STEPS_UM:
        position += int(step)
        cumulative.append(position)
    if tuple(cumulative) != tuple(MICRO_TARGET_ABSOLUTE_UM):
        raise ValueError(
            "MICRO_TARGET_ABSOLUTE_UM 必须等于相对目标序列的累计和。"
        )
    if MICRO_TARGET_MAX_ITER != 12:
        raise ValueError("MICRO_TARGET_MAX_ITER 必须为 12。")
    if MICRO_TARGET_STABLE_TOL_UM != 3.0 or MICRO_TARGET_STABLE_COUNT != 3:
        raise ValueError("多目标闭环稳定到达判据必须是连续 3 次 |error| <= 3 μm。")
    if MICRO_TARGET_SMALL_COMMAND_UM <= MICRO_LOOP_MIN_COMMAND_UM:
        raise ValueError("MICRO_TARGET_SMALL_COMMAND_UM 必须大于运动命令死区。")
    if MICRO_TARGET_MAX_COMMAND_UM < max(abs(v) for v in MICRO_TARGET_RELATIVE_STEPS_UM):
        raise ValueError("MICRO_TARGET_MAX_COMMAND_UM 不能小于最大相邻目标步长。")
    if MICRO_TARGET_LOCAL_RANGE_UM <= max(abs(v) for v in MICRO_TARGET_ABSOLUTE_UM):
        raise ValueError("MICRO_TARGET_LOCAL_RANGE_UM 必须大于最大绝对目标位置。")
    if MICRO_TARGET_MATRIX_MAX_CONDITION <= 1.0:
        raise ValueError("MICRO_TARGET_MATRIX_MAX_CONDITION 必须大于 1。")
    if MICRO_TARGET_LIMIT_WINDOW < 5:
        raise ValueError("MICRO_TARGET_LIMIT_WINDOW 至少为 5。")
    if MICRO_TARGET_LIMIT_MIN_SIGN_CHANGES < 3:
        raise ValueError("极限振荡至少要求最近窗口发生 3 次误差换号。")
    if not 0.0 < MICRO_TARGET_LIMIT_IMPROVEMENT_RATIO < 1.0:
        raise ValueError("MICRO_TARGET_LIMIT_IMPROVEMENT_RATIO 必须在 0 到 1 之间。")

    # 死区必须严格大于 robot.validate_trajectory 的最小线段 1 μm。
    # 小于等于 1 μm 的命令会命中"两点位置重合"的硬抛，恰好在一个目标
    # 即将收敛的时刻把整轮弄死——这正是死区存在的唯一理由。
    if MICRO_LOOP_MIN_COMMAND_UM <= 1.0:
        raise ValueError(
            "MICRO_LOOP_MIN_COMMAND_UM 必须大于 1.0 μm。"
            "robot.validate_trajectory 拒绝 length <= 1e-6 m，"
            "死区小于等于 1 μm 时收敛前的最后一步会硬抛“两点位置重合”。"
        )
    if MICRO_LOOP_MIN_COMMAND_UM >= MICRO_LOOP_MAX_CORRECTION_UM:
        raise ValueError("MICRO_LOOP_MIN_COMMAND_UM 必须小于 MICRO_LOOP_MAX_CORRECTION_UM。")
    if MICRO_LOOP_KP <= 0:
        raise ValueError("MICRO_LOOP_KP 必须为正数。")
    if MICRO_LOOP_MAX_CORRECTION_UM <= 0:
        raise ValueError("MICRO_LOOP_MAX_CORRECTION_UM 必须为正数。")
    if MICRO_LOOP_MAX_CORRECTION_UM >= ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M * 1e6 / 10.0:
        raise ValueError(
            "MICRO_LOOP_MAX_CORRECTION_UM 相对 200 mm 相对工作区过大，失去限幅意义。"
        )
    if MICRO_LOOP_MAX_CORRECTION_UM < max(MICRO_LOOP_AMPLITUDES_UM):
        raise ValueError(
            "MICRO_LOOP_MAX_CORRECTION_UM 不能小于最大目标幅值，否则该幅值永远无法一步到位。"
        )

    if MICRO_LOOP_MAX_ITER < MICRO_LOOP_CYCLE_WINDOW * 2:
        raise ValueError(
            "MICRO_LOOP_MAX_ITER 必须至少是 MICRO_LOOP_CYCLE_WINDOW 的两倍，"
            "否则极限环判据永远拿不到两个完整窗口。"
        )
    if MICRO_LOOP_CYCLE_WINDOW < 3:
        raise ValueError("MICRO_LOOP_CYCLE_WINDOW 至少为 3，否则换号次数判据没有意义。")
    if MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES < 2:
        raise ValueError(
            "MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES 必须至少为 2；"
            "只换号 1 次说明是单向逼近后的一次过冲，不是极限环。"
        )
    if MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES > MICRO_LOOP_CYCLE_WINDOW - 1:
        raise ValueError(
            "MICRO_LOOP_CYCLE_MIN_SIGN_CHANGES 不能超过窗口内的最大可能换号次数。"
        )
    if MICRO_LOOP_STALL_PATIENCE < 2:
        raise ValueError(
            "MICRO_LOOP_STALL_PATIENCE 至少为 2；只看到 1 次没改善就判 STALLED，"
            "会把正常的第一步大修正误报成停滞。"
        )
    if MICRO_LOOP_STALL_PATIENCE >= MICRO_LOOP_MAX_ITER:
        raise ValueError(
            "MICRO_LOOP_STALL_PATIENCE 必须小于 MICRO_LOOP_MAX_ITER，"
            "否则停滞判据在迭代预算耗尽前永远不会生效。"
        )
    if MICRO_LOOP_STALL_MIN_IMPROVE_UM < 0:
        raise ValueError("MICRO_LOOP_STALL_MIN_IMPROVE_UM 不能为负。")
    if MICRO_LOOP_CONVERGE_COUNT > MICRO_LOOP_MAX_ITER:
        raise ValueError(
            "MICRO_LOOP_CONVERGE_COUNT 不能超过 MICRO_LOOP_MAX_ITER，"
            "否则永远攒不够连续收敛次数。"
        )

    # ---- 在线分析的帧预算 ----
    # 这几个数直接决定整轮时长（在线耗时 ≈ 分析帧数 × 0.165 s），
    # 所以它们之间的顺序写反会让实验要么超时、要么结论不可信，必须在启动前拦住。
    if MICRO_LOOP_MEASURE_FRAMES < 4:
        raise ValueError(
            "MICRO_LOOP_MEASURE_FRAMES 至少为 4；再少中位数的统计意义就不成立了。"
        )
    if MICRO_LOOP_MIN_TAIL_FRAMES > MICRO_LOOP_MEASURE_FRAMES:
        raise ValueError(
            "MICRO_LOOP_MIN_TAIL_FRAMES 不能超过 MICRO_LOOP_MEASURE_FRAMES，"
            "否则每一轮测量都会因为“合格帧不够”被判不可用。"
        )
    if MICRO_LOOP_MIN_TAIL_FRAMES < max(3, MICRO_LOOP_MEASURE_FRAMES // 2):
        raise ValueError(
            "MICRO_LOOP_MIN_TAIL_FRAMES 相对 MICRO_LOOP_MEASURE_FRAMES 过小，"
            "合格帧屈指可数时仍会放行，测量不可信。"
        )
    if MICRO_LOOP_REFERENCE_FRAMES <= MICRO_LOOP_MEASURE_FRAMES:
        raise ValueError(
            "MICRO_LOOP_REFERENCE_FRAMES 必须大于 MICRO_LOOP_MEASURE_FRAMES；"
            "零点的不确定度不能比单次测量本身还差。"
        )
    if MICRO_LOOP_STATIC_ANALYZE_FRAMES < MICRO_LOOP_MEASURE_FRAMES:
        raise ValueError(
            "MICRO_LOOP_STATIC_ANALYZE_FRAMES 不能小于 MICRO_LOOP_MEASURE_FRAMES；"
            "静止基线算不出比单次测量更好的统计量就没有意义。"
        )
    if MICRO_LOOP_PROCESS_EVERY_N_FRAMES < 1:
        raise ValueError("MICRO_LOOP_PROCESS_EVERY_N_FRAMES 必须至少为 1。")

    # ---- 开环模式 ----
    if not MICRO_LOOP_OPEN_MODES:
        raise ValueError("MICRO_LOOP_OPEN_MODES 不能为空。")
    if set(MICRO_LOOP_OPEN_MODES) - {"seq", "alt"}:
        raise ValueError('MICRO_LOOP_OPEN_MODES 只能包含 "seq" 和 "alt"。')
    if MICRO_LOOP_OPEN_SEQ_POSITIVE_STEPS < 1 or MICRO_LOOP_OPEN_SEQ_NEGATIVE_STEPS < 1:
        raise ValueError("开环 seq 模式的正向与反向步数都必须至少为 1。")
    if MICRO_LOOP_OPEN_ALT_STEPS < 2 or MICRO_LOOP_OPEN_ALT_STEPS % 2 != 0:
        raise ValueError(
            "MICRO_LOOP_OPEN_ALT_STEPS 必须是大于等于 2 的偶数；"
            "换向模式按 +Δ/−Δ 成对出现，奇数步会留下净位移。"
        )
    if not 0.0 < MICRO_LOOP_OPEN_MIN_RATIO < 1.0:
        raise ValueError("MICRO_LOOP_OPEN_MIN_RATIO 必须在 0 到 1 之间。")
    if MICRO_LOOP_OPEN_MAX_RATIO <= 1.0:
        raise ValueError("MICRO_LOOP_OPEN_MAX_RATIO 必须大于 1。")
    if MICRO_LOOP_OPEN_ZERO_UM <= 0:
        raise ValueError("MICRO_LOOP_OPEN_ZERO_UM 必须为正数。")
    if MICRO_LOOP_OPEN_JUMP_UM <= max(MICRO_LOOP_AMPLITUDES_UM) * MICRO_LOOP_OPEN_MAX_RATIO:
        raise ValueError(
            "MICRO_LOOP_OPEN_JUMP_UM 太小，正常档位就会触发异常跳变告警。"
        )

    # ---- 漂移守卫 ----
    if MICRO_LOOP_AXIS_DRIFT_ABORT_UM <= MICRO_LOOP_AXIS_DRIFT_WARN_UM:
        raise ValueError(
            "MICRO_LOOP_AXIS_DRIFT_ABORT_UM 必须大于 MICRO_LOOP_AXIS_DRIFT_WARN_UM。"
        )
    if MICRO_LOOP_AXIS_DRIFT_ABORT_UM >= ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M * 1e6 / 10.0:
        raise ValueError(
            "MICRO_LOOP_AXIS_DRIFT_ABORT_UM 相对 200 mm 相对工作区过大，失去守卫意义。"
        )

    # ---- 数据落盘与内存 ----
    if MICRO_LOOP_KEEP_ALL_RESERVE_GB < 0:
        raise ValueError("MICRO_LOOP_KEEP_ALL_RESERVE_GB 不能为负。")
    if MICRO_LOOP_KEEP_ALL_RESERVE_GB > MICRO_LOOP_MIN_FREE_GB_START:
        raise ValueError(
            "MICRO_LOOP_KEEP_ALL_RESERVE_GB 不应大于 MICRO_LOOP_MIN_FREE_GB_START，"
            "两者语义重复，会让“是否全量保留”几乎永远为假。"
        )
    if MICRO_LOOP_MIN_FREE_RAM_GB <= 0:
        raise ValueError("MICRO_LOOP_MIN_FREE_RAM_GB 必须为正数。")
    if MICRO_LOOP_MIN_FREE_RAM_GB >= 8.0:
        raise ValueError(
            "MICRO_LOOP_MIN_FREE_RAM_GB 大得不像可用物理内存门槛；"
            "本机总内存 15.77 GiB，门槛定到 8 GiB 以上等于永远拒绝启动。"
        )
    if MICRO_LOOP_EVIDENCE_MAX < 0 or MICRO_LOOP_EVIDENCE_MAX > 3:
        raise ValueError("MICRO_LOOP_EVIDENCE_MAX 只能在 0 到 3 之间。")
    if not 0.0 < MICRO_LOOP_CYCLE_PTP_KEEP <= 1.0:
        raise ValueError("MICRO_LOOP_CYCLE_PTP_KEEP 必须在 0 到 1 之间。")
    if not 0.0 < MICRO_LOOP_CYCLE_ABS_KEEP <= 1.0:
        raise ValueError("MICRO_LOOP_CYCLE_ABS_KEEP 必须在 0 到 1 之间。")
    if MICRO_LOOP_CONVERGE_COUNT < 1:
        raise ValueError("MICRO_LOOP_CONVERGE_COUNT 必须至少为 1。")
    if MICRO_LOOP_CONVERGE_COUNT > MICRO_LOOP_MAX_ITER:
        raise ValueError("MICRO_LOOP_CONVERGE_COUNT 不能大于 MICRO_LOOP_MAX_ITER。")
    if MICRO_LOOP_POSITION_TOL_UM <= 0:
        raise ValueError("MICRO_LOOP_POSITION_TOL_UM 必须为正数。")
    if MICRO_LOOP_POSITION_TOL_MODE not in {"absolute", "noise_relative"}:
        raise ValueError(
            'MICRO_LOOP_POSITION_TOL_MODE 只能是 "absolute" 或 "noise_relative"。'
        )
    if MICRO_LOOP_NOISE_SIGMA_MULT <= 0:
        raise ValueError("MICRO_LOOP_NOISE_SIGMA_MULT 必须为正数。")
    if not 0.0 < MICRO_LOOP_MIN_DIRECTIONAL_FRACTION <= 1.0:
        raise ValueError(
            "MICRO_LOOP_MIN_DIRECTIONAL_FRACTION 必须落在 (0, 1] 内："
            "它是“至少要走完幅值的多大比例才承认朝目标逼近”的门槛。"
        )

    if not ROBOT_EXPERIMENT_MIN_SPEED_MM_S <= MICRO_LOOP_SPEED_MM_S <= ROBOT_EXPERIMENT_MAX_SPEED_MM_S:
        raise ValueError(
            "MICRO_LOOP_SPEED_MM_S 必须落在 ROBOT_EXPERIMENT_MIN_SPEED_MM_S 与 "
            "ROBOT_EXPERIMENT_MAX_SPEED_MM_S 之间。"
        )
    if not ROBOT_EXPERIMENT_MIN_SPEED_MM_S <= MICRO_LOOP_RETURN_SPEED_MM_S <= ROBOT_EXPERIMENT_MAX_SPEED_MM_S:
        raise ValueError("MICRO_LOOP_RETURN_SPEED_MM_S 必须落在实验允许的速度区间内。")
    if MICRO_LOOP_ACHIEVED_MAX_RATIO < MICRO_LOOP_ACHIEVED_MIN_RATIO:
        raise ValueError("MICRO_LOOP_ACHIEVED_MAX_RATIO 不能小于 MICRO_LOOP_ACHIEVED_MIN_RATIO。")

    # 探针：增益门禁必须自洽，且探针位移本身要能构成合法线段。
    if MICRO_LOOP_PROBE_UM <= 0:
        raise ValueError("MICRO_LOOP_PROBE_UM 必须为正数。")
    if MICRO_LOOP_PROBE_UM / 1000.0 > ROBOT_EXPERIMENT_MAX_SEGMENT_MM:
        raise ValueError("MICRO_LOOP_PROBE_UM 不能超过 ROBOT_EXPERIMENT_MAX_SEGMENT_MM。")
    if (
        not isinstance(MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS, int)
        or isinstance(MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS, bool)
        or MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS < 0
    ):
        raise ValueError("MICRO_LOOP_PROBE_REMEASURE_ATTEMPTS 必须是非负整数。")
    if MICRO_LOOP_PROBE_REFERENCE_FRAMES < MICRO_LOOP_MIN_TAIL_FRAMES:
        raise ValueError("MICRO_LOOP_PROBE_REFERENCE_FRAMES 不能小于最少尾窗帧数。")
    if MICRO_LOOP_PROBE_MEASURE_FRAMES < MICRO_LOOP_MIN_TAIL_FRAMES:
        raise ValueError("MICRO_LOOP_PROBE_MEASURE_FRAMES 不能小于最少尾窗帧数。")
    if MICRO_LOOP_PROBE_TAIL_WINDOW_S <= 0:
        raise ValueError("MICRO_LOOP_PROBE_TAIL_WINDOW_S 必须为正数。")
    if MICRO_LOOP_PROBE_TAIL_WINDOW_S > MICRO_LOOP_MOTION_SETTLE_SECONDS:
        raise ValueError(
            "MICRO_LOOP_PROBE_TAIL_WINDOW_S 不能超过无运动复测的固定等待时间。"
        )
    if not math.isfinite(OFFLINE_STATIC_SECONDS) or OFFLINE_STATIC_SECONDS <= 0.0:
        raise ValueError("OFFLINE_STATIC_SECONDS 必须是有限正数。")
    if (
        not isinstance(OFFLINE_STATIC_REPEATS, int)
        or isinstance(OFFLINE_STATIC_REPEATS, bool)
        or OFFLINE_STATIC_REPEATS < 1
    ):
        raise ValueError("OFFLINE_STATIC_REPEATS 必须是正整数。")
    if (
        not math.isfinite(OFFLINE_STATIC_DISK_RESERVE_GB)
        or OFFLINE_STATIC_DISK_RESERVE_GB < 0.0
    ):
        raise ValueError("OFFLINE_STATIC_DISK_RESERVE_GB 必须是有限非负数。")
    if (
        not math.isfinite(MICRO_LOOP_FIXED_SAFETY_RADIUS_M)
        or MICRO_LOOP_FIXED_SAFETY_RADIUS_M <= 0
        or MICRO_LOOP_FIXED_SAFETY_RADIUS_M > ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M
    ):
        raise ValueError(
            "MICRO_LOOP_FIXED_SAFETY_RADIUS_M 必须是有限正数，且不能超过相对工作区半径。"
        )
    if not isinstance(MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT, bool):
        raise ValueError("MICRO_LOOP_DELETE_VIDEO_FILES_ON_EXIT 必须是布尔值。")
    if not isinstance(MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP, bool):
        raise ValueError("MICRO_LOOP_DELETE_VIDEO_FILES_AFTER_GROUP 必须是布尔值。")
    if not 0.0 < MICRO_LOOP_GAIN_MIN < MICRO_LOOP_GAIN_MAX:
        raise ValueError("MICRO_LOOP_GAIN_MIN 必须为正数且小于 MICRO_LOOP_GAIN_MAX。")
    if MICRO_LOOP_GAIN_MIN * MICRO_LOOP_PROBE_UM < MICRO_LOOP_POSITION_TOL_UM:
        raise ValueError(
            "MICRO_LOOP_GAIN_MIN 与探针位移的乘积小于位置容差，"
            "探针将无法把一个真实位移与噪声区分开。"
        )
    if not 0.0 < MICRO_LOOP_PROBE_HYSTERESIS_RATIO < 1.0:
        raise ValueError("MICRO_LOOP_PROBE_HYSTERESIS_RATIO 必须在 0 到 1 之间。")
    if not 0.0 < MICRO_LOOP_CROSS_COUPLING_WARN < 1.0:
        raise ValueError("MICRO_LOOP_CROSS_COUPLING_WARN 必须在 0 到 1 之间。")

    # 运行期联锁：漂移告警必须早于漂移中止，中止必须远小于 200 mm 相对工作区。
    if MICRO_LOOP_DRIFT_WARN_UM <= 0 or MICRO_LOOP_DRIFT_ABORT_UM <= 0:
        raise ValueError("漂移守卫阈值必须为正数。")
    if MICRO_LOOP_DRIFT_WARN_UM >= MICRO_LOOP_DRIFT_ABORT_UM:
        raise ValueError("MICRO_LOOP_DRIFT_WARN_UM 必须小于 MICRO_LOOP_DRIFT_ABORT_UM。")
    if MICRO_LOOP_DRIFT_ABORT_UM / 1000.0 > ROBOT_RELATIVE_WORKSPACE_HALF_RANGE_M * 1000.0:
        raise ValueError("MICRO_LOOP_DRIFT_ABORT_UM 不能超过相对工作区半径。")
    if MICRO_LOOP_CUM_COMMAND_ABORT_UM <= MICRO_LOOP_MAX_CORRECTION_UM:
        raise ValueError(
            "MICRO_LOOP_CUM_COMMAND_ABORT_UM 必须大于单次限幅，否则第一步就会触发中止。"
        )
    if MICRO_LOOP_SIGN_FLIP_ABORT_COUNT < 1:
        raise ValueError("MICRO_LOOP_SIGN_FLIP_ABORT_COUNT 必须至少为 1。")

    # 磁盘：续跑门槛必须低于启动门槛，回收配额必须低于续跑门槛，
    # 否则会出现"刚通过检查就因为回收目录超配额而中止"的死循环。
    if MICRO_LOOP_MIN_FREE_GB_CONTINUE >= MICRO_LOOP_MIN_FREE_GB_START:
        raise ValueError(
            "MICRO_LOOP_MIN_FREE_GB_CONTINUE 必须小于 MICRO_LOOP_MIN_FREE_GB_START。"
        )
    if MICRO_LOOP_TRASH_QUOTA_GB >= MICRO_LOOP_MIN_FREE_GB_CONTINUE:
        raise ValueError(
            "MICRO_LOOP_TRASH_QUOTA_GB 必须小于 MICRO_LOOP_MIN_FREE_GB_CONTINUE。"
        )
    if MICRO_LOOP_MIN_FREE_GB_CONTINUE <= 0:
        raise ValueError("MICRO_LOOP_MIN_FREE_GB_CONTINUE 必须为正数。")

    if MICRO_LOOP_QUALITY_MIN < 0.0 or MICRO_LOOP_QUALITY_MIN > 1.0:
        raise ValueError("MICRO_LOOP_QUALITY_MIN 必须在 0 到 1 之间。")
    if MICRO_LOOP_SCALE_TOL <= 0 or MICRO_LOOP_SCALE_TOL >= 0.5:
        raise ValueError("MICRO_LOOP_SCALE_TOL 必须为正且远小于 1。")
    if MICRO_LOOP_MAX_PLAUSIBLE_SHIFT_UM <= MICRO_LOOP_MAX_ITER_JUMP_UM:
        raise ValueError(
            "MICRO_LOOP_MAX_PLAUSIBLE_SHIFT_UM 必须大于 MICRO_LOOP_MAX_ITER_JUMP_UM，"
            "否则逐帧物理门禁比迭代间联锁还严，无法区分两种失效。"
        )
    if MICRO_LOOP_MAX_ITER_JUMP_UM <= 0:
        raise ValueError("MICRO_LOOP_MAX_ITER_JUMP_UM 必须为正数。")
    if MICRO_LOOP_MIN_TAIL_FRAMES < 1:
        raise ValueError("MICRO_LOOP_MIN_TAIL_FRAMES 必须至少为 1。")
    if MICRO_LOOP_EVIDENCE_MAX < 0:
        raise ValueError("MICRO_LOOP_EVIDENCE_MAX 不能为负数。")
    if MICRO_LOOP_VERIFY_MAX_ATTEMPTS < 0:
        raise ValueError("MICRO_LOOP_VERIFY_MAX_ATTEMPTS 不能为负数。")

    for name, value in {
        "MICRO_LOOP_PRE_SECONDS": MICRO_LOOP_PRE_SECONDS,
        "MICRO_LOOP_POST_SECONDS": MICRO_LOOP_POST_SECONDS,
        "MICRO_LOOP_TAIL_WINDOW_S": MICRO_LOOP_TAIL_WINDOW_S,
        "MICRO_LOOP_REFERENCE_SECONDS": MICRO_LOOP_REFERENCE_SECONDS,
        "MICRO_LOOP_STATIC_SECONDS": MICRO_LOOP_STATIC_SECONDS,
        "MICRO_LOOP_CAMERA_WARMUP_SECONDS": MICRO_LOOP_CAMERA_WARMUP_SECONDS,
        "MICRO_LOOP_PREVIEW_FPS": MICRO_LOOP_PREVIEW_FPS,
        "MICRO_LOOP_PREVIEW_MAX_WIDTH": MICRO_LOOP_PREVIEW_MAX_WIDTH,
        "MICRO_LOOP_PROCESS_EVERY_N_FRAMES": MICRO_LOOP_PROCESS_EVERY_N_FRAMES,
        "MICRO_LOOP_ACCELERATION_M_S2": MICRO_LOOP_ACCELERATION_M_S2,
        "MICRO_LOOP_STABLE_WINDOW_MS": MICRO_LOOP_STABLE_WINDOW_MS,
        "MICRO_LOOP_STABLE_WINDOW_UM": MICRO_LOOP_STABLE_WINDOW_UM,
        "MICRO_LOOP_STABLE_SPEED_MM_S": MICRO_LOOP_STABLE_SPEED_MM_S,
        "MICRO_LOOP_STABLE_HOLD_SECONDS": MICRO_LOOP_STABLE_HOLD_SECONDS,
        "MICRO_LOOP_STABLE_TIMEOUT_S": MICRO_LOOP_STABLE_TIMEOUT_S,
        "MICRO_LOOP_MOTION_SETTLE_SECONDS": MICRO_LOOP_MOTION_SETTLE_SECONDS,
        "MICRO_LOOP_STEP_TIMEOUT_S": MICRO_LOOP_STEP_TIMEOUT_S,
        "MICRO_LOOP_PRE_STABLE_TIMEOUT_S": MICRO_LOOP_PRE_STABLE_TIMEOUT_S,
        "MICRO_LOOP_POST_STABLE_TIMEOUT_S": MICRO_LOOP_POST_STABLE_TIMEOUT_S,
        "MICRO_LOOP_MAX_WINDOW_SECONDS": MICRO_LOOP_MAX_WINDOW_SECONDS,
        "MICRO_LOOP_TAIL_SIGMA_MAX_UM": MICRO_LOOP_TAIL_SIGMA_MAX_UM,
        "MICRO_LOOP_PROBE_REFERENCE_SECONDS": MICRO_LOOP_PROBE_REFERENCE_SECONDS,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数。")

    # 录制窗口内各阶段的最坏耗时之和必须装得进相机进程预分配的缓冲。
    # 装不进去不会静默丢数据（相机会如实标记截断），但那一刻的数据已经不可用了。
    worst_window_s = (
        MICRO_LOOP_PRE_SECONDS
        # 当前多目标路径在同一活动 RAW 内先分析 16 帧，算出 error 后才发命令。
        + MICRO_LOOP_MEASURE_FRAMES * 0.165
        + MICRO_LOOP_PRE_STABLE_TIMEOUT_S
        + MICRO_LOOP_MOTION_SETTLE_SECONDS
        + MICRO_LOOP_STEP_TIMEOUT_S
        + MICRO_LOOP_POST_STABLE_TIMEOUT_S
        + MICRO_LOOP_MOTION_SETTLE_SECONDS
        + MICRO_LOOP_POST_SECONDS
    )
    if worst_window_s >= MICRO_LOOP_MAX_WINDOW_SECONDS:
        raise ValueError(
            f"录制窗口最坏耗时 {worst_window_s:.2f} s 已达到或超过 "
            f"MICRO_LOOP_MAX_WINDOW_SECONDS={MICRO_LOOP_MAX_WINDOW_SECONDS:g} s。"
            "请调小窗口内的超时，或调大预分配缓冲。"
        )

    # 取尾窗口中位数要求窗口内有足够多的帧，否则"尾部中位数"退化成一帧。
    if MICRO_LOOP_TAIL_WINDOW_S > MICRO_LOOP_POST_SECONDS:
        raise ValueError(
            "MICRO_LOOP_TAIL_WINDOW_S 不能大于 MICRO_LOOP_POST_SECONDS，"
            "否则尾部窗口会伸进运动瞬态。"
        )
    if MICRO_LOOP_FULL_FRAME_WIDTH < 16 or MICRO_LOOP_FULL_FRAME_HEIGHT < 16:
        raise ValueError("MICRO_LOOP_FULL_FRAME_WIDTH/HEIGHT 过小。")

    for name, value in {
        "BRIGHTNESS_BASELINE_SECONDS": BRIGHTNESS_BASELINE_SECONDS,
        "FLASH_WAIT_TIMEOUT_SECONDS": FLASH_WAIT_TIMEOUT_SECONDS,
        "FLASH_MIN_DURATION_SECONDS": FLASH_MIN_DURATION_SECONDS,
        "VISION_RECOVERY_STABLE_SECONDS": VISION_RECOVERY_STABLE_SECONDS,
        "FLASH_RECOVERY_TIMEOUT_SECONDS": FLASH_RECOVERY_TIMEOUT_SECONDS,
        "POST_MOTION_RECORD_SECONDS": POST_MOTION_RECORD_SECONDS,
        "FLASH_MEAN_ABSOLUTE_INCREASE": FLASH_MEAN_ABSOLUTE_INCREASE,
        "FLASH_MEAN_MAD_MULTIPLIER": FLASH_MEAN_MAD_MULTIPLIER,
        "FLASH_RECOVERY_MEAN_TOLERANCE": FLASH_RECOVERY_MEAN_TOLERANCE,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数。")

    if not 0.0 <= FLASH_MEAN_RELATIVE_INCREASE <= 10.0:
        raise ValueError("FLASH_MEAN_RELATIVE_INCREASE 必须在合理范围内。")

    if not 0.0 <= FLASH_SATURATION_RELATIVE_INCREASE <= 1.0:
        raise ValueError("FLASH_SATURATION_RELATIVE_INCREASE 必须在 0 到 1 之间。")

    if not 0 <= FLASH_SATURATION_THRESHOLD <= 255:
        raise ValueError("FLASH_SATURATION_THRESHOLD 必须在 0 到 255 之间。")

    if not 0.0 <= FLASH_RECOVERY_SATURATION_TOLERANCE <= 1.0:
        raise ValueError("FLASH_RECOVERY_SATURATION_TOLERANCE 必须在 0 到 1 之间。")

    # 本段检查视觉识别、时间轴和空间坐标参数。
    # 输入：棋盘格尺寸、圆点数量、滤波核、相机内参、fps 提示阈值、空间坐标开关等。
    # 输出：确认每帧结果可以被解释为可信的像素/空间/时间数据。
    # 实验作用：这些值不会让机器人运动，但会直接决定视觉结果是否有物理意义。
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

    if IMAGE_FOLDER_FPS <= 0:
        raise ValueError("IMAGE_FOLDER_FPS 必须为正数。")

    if EXPECTED_VISION_FPS is not None and EXPECTED_VISION_FPS <= 0:
        raise ValueError("EXPECTED_VISION_FPS 必须为正数或 None。")

    if FPS_WARNING_RELATIVE_TOLERANCE < 0 or FPS_WARNING_ABSOLUTE_TOLERANCE_HZ < 0:
        raise ValueError("FPS 警告容差不能为负数。")

    if VISION_TIMING_WARMUP_FRAMES < 0:
        raise ValueError("VISION_TIMING_WARMUP_FRAMES 不能为负数。")

    if CAPTURE_DURATION_S is not None and CAPTURE_DURATION_S <= 0:
        raise ValueError("CAPTURE_DURATION_S 必须为 None 或大于 0。")

    if CAPTURE_START_COUNTDOWN_S < 0:
        raise ValueError("CAPTURE_START_COUNTDOWN_S 不能为负数。")

    if CAPTURE_STORAGE_MODE not in {"stream_raw", "memmap_raw", "ram_then_raw"}:
        raise ValueError(
            "CAPTURE_STORAGE_MODE 只允许 'stream_raw'、'memmap_raw' 或 'ram_then_raw'。"
        )

    if CAPTURE_DURATION_S is None and CAPTURE_STORAGE_MODE != "stream_raw":
        raise ValueError("CAPTURE_DURATION_S=None 时，CAPTURE_STORAGE_MODE 必须为 'stream_raw'。")

    if CAPTURE_RAW_WRITE_CHUNK_FRAMES < 1:
        raise ValueError("CAPTURE_RAW_WRITE_CHUNK_FRAMES 必须至少为 1。")

    if CAPTURE_PREVIEW_FPS <= 0:
        raise ValueError("CAPTURE_PREVIEW_FPS 必须大于 0。")

    if OFFLINE_PROGRESS_EVERY_N_FRAMES < 1:
        raise ValueError("OFFLINE_PROGRESS_EVERY_N_FRAMES 必须至少为 1。")

    if selected_mode == "vision_offline" and RAW_CAPTURE_DIR is not None:
        capture_dir = Path(RAW_CAPTURE_DIR).expanduser()
        if not capture_dir.is_absolute():
            capture_dir = (PROJECT_DIR / capture_dir).resolve()
        if not capture_dir.exists() or not capture_dir.is_dir():
            raise ValueError(f"RAW_CAPTURE_DIR 不存在或不是文件夹：{capture_dir}")

    if not 0 <= STEADY_MOTION_TRIM_FRACTION < 0.5:
        raise ValueError("STEADY_MOTION_TRIM_FRACTION 必须在 [0, 0.5) 范围内。")

    if ANALYSIS_SEGMENTATION_SOURCE not in VALID_ANALYSIS_SEGMENTATION_SOURCES:
        raise ValueError(
            "ANALYSIS_SEGMENTATION_SOURCE 可选值为 "
            f"{sorted(VALID_ANALYSIS_SEGMENTATION_SOURCES)}。"
        )

    if ROBOT_MOTION_ON_SPEED_M_S <= 0 or ROBOT_MOTION_OFF_SPEED_M_S < 0:
        raise ValueError("机器人运动速度阈值必须为正数或非负数。")

    if ROBOT_MOTION_OFF_SPEED_M_S >= ROBOT_MOTION_ON_SPEED_M_S:
        raise ValueError("ROBOT_MOTION_OFF_SPEED_M_S 必须小于 ROBOT_MOTION_ON_SPEED_M_S。")

    if ROBOT_STEADY_ACCELERATION_M_S2 <= 0:
        raise ValueError("ROBOT_STEADY_ACCELERATION_M_S2 必须为正数。")

    if ROBOT_STEADY_SPEED_RELATIVE_TOLERANCE < 0:
        raise ValueError("ROBOT_STEADY_SPEED_RELATIVE_TOLERANCE 不能为负数。")

    if (
        SEGMENT_MIN_MOTION_SECONDS < 0
        or SEGMENT_MIN_STATIC_SECONDS < 0
        or SEGMENT_MERGE_GAP_SECONDS < 0
    ):
        raise ValueError("分段最小时长和合并间隔不能为负数。")

    if VISION_MOTION_VELOCITY_FACTOR <= 0:
        raise ValueError("VISION_MOTION_VELOCITY_FACTOR 必须为正数。")

    if ANALYSIS_PRIMARY_WINDOW not in VALID_ANALYSIS_PRIMARY_WINDOWS:
        raise ValueError(
            f"ANALYSIS_PRIMARY_WINDOW 可选值为 {sorted(VALID_ANALYSIS_PRIMARY_WINDOWS)}。"
        )

    if SPATIAL_COORDINATES_ENABLED:
        # 本段只在空间坐标启用后执行。
        # 输入：像素到空间投影所需的内参、深度平面或机器人外参。
        # 输出：允许 calibration.py 计算空间点，或在缺少关键标定值时停止。
        # 实验作用：空间坐标是第二层结果，必须在标定参数完整时才生成。
        _check_matrix("CAMERA_MATRIX", CAMERA_MATRIX, 3, 3)
        if SPATIAL_COORDINATE_MODE == "camera_plane":
            if SPATIAL_CAMERA_PLANE_Z_M is None or SPATIAL_CAMERA_PLANE_Z_M <= 0:
                raise ValueError("camera_plane 模式必须填写正数 SPATIAL_CAMERA_PLANE_Z_M。")
        if SPATIAL_COORDINATE_MODE == "robot_plane":
            _check_matrix("CAMERA_TO_ROBOT_ROTATION", CAMERA_TO_ROBOT_ROTATION, 3, 3)
            _check_vector("CAMERA_TO_ROBOT_TRANSLATION_M", CAMERA_TO_ROBOT_TRANSLATION_M, 3)
            _check_vector("MEASUREMENT_PLANE_POINT_ROBOT_M", MEASUREMENT_PLANE_POINT_ROBOT_M, 3)
            _check_vector("MEASUREMENT_PLANE_NORMAL_ROBOT", MEASUREMENT_PLANE_NORMAL_ROBOT, 3)

    if VISION_ROI is not None:
        # 本段检查 ROI 是否真的是一个非空图像区域。
        # 输入：VISION_ROI；输出：允许裁剪或提示 ROI 写法错误。
        # 实验作用：避免宽高为 0 或负数时，后续识别算法拿到空图像。
        if len(VISION_ROI) != 4 or any(value < 0 for value in VISION_ROI):
            raise ValueError("VISION_ROI 必须是非负的 (x, y, width, height)。")
        if VISION_ROI[2] == 0 or VISION_ROI[3] == 0:
            raise ValueError("VISION_ROI 的 width 和 height 必须大于 0。")

    # 本段检查机器人轨迹数值是否基本合理。
    # 输入：A/B/C 位姿、速度、加速度、交融半径、记录频率和记录时间。
    # 输出：允许 robot.py 继续做更详细的轨迹/安全区检查，或在明显错误时停止。
    # 实验作用：这里不能替代示教器和现场安全确认，只负责拦住代码层面一眼能看出的坏值。
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

    # 本段检查离线分析会如何解读已有记录。
    # 输入：视觉方法、分析方向和去趋势方法。
    # 输出：确认 analyze.py 后续生成图表和指标时，使用的是明确且合法的解释方式。
    # 实验作用：分析模式不碰硬件，但参数写错会让输出图表的含义变错。
    if ANALYSIS_VISION_METHOD not in {"circles", "checkerboard"}:
        raise ValueError("ANALYSIS_VISION_METHOD 只能是 circles 或 checkerboard。")

    if ANALYSIS_AXIS not in VALID_ANALYSIS_AXES:
        raise ValueError(f"ANALYSIS_AXIS 可选值为 {sorted(VALID_ANALYSIS_AXES)}。")

    if DETREND_METHOD not in VALID_DETREND_METHODS:
        raise ValueError(f"DETREND_METHOD 可选值为 {sorted(VALID_DETREND_METHODS)}。")

    # 本段是配置检查通过后的唯一文件系统动作。
    # 输入：OUTPUT_ROOT；输出：确保结果根目录存在。
    # 实验作用：只创建程序自己的输出目录，不会创建假的输入数据，也不会连接任何硬件。
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
