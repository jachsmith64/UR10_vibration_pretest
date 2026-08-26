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
    "robot_dry_run",  # 只生成并检查轨迹，不导入 UR 库，也不连接真机。
    "robot_test",     # 连接 UR，默认只读取状态；必须再次开关才允许低速运动。
    "experiment",     # 相机、UR 和记录进程共同运行的正式预实验。
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
    r"D:\SOFTWARE\MindVision_cs028_10UM\MVS\Development\Samples\Python\MvImport"
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
ROBOT_HOST = "192.168.0.10"
ROBOT_DASHBOARD_PORT = 29999
ROBOT_CONNECT_TIMEOUT_S = 5.0

# 本段把 robot_test 拆成“只连机读取”和“允许低速动一下”两级。
# 输入：ROBOT_TEST_ALLOW_MOTION；输出：robot_test 是否会发送 A→B 测试运动。
# 实验作用：第一次接触实机必须保持 False，先确认连接、坐标读取和日志流程正常。
ROBOT_TEST_ALLOW_MOTION = False

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

# 本段给代码层面加一个 TCP 工作区边界。
# 输入：待执行轨迹中的所有 TCP 点；输出：通过检查或拒绝执行。
# 实验作用：这是额外保险，不等于 UR 控制器自身安全设置，也无法识别桌面和夹具。
# 三个范围分别限制 TCP 的 x、y、z，单位 m；实机前必须按实验台重新填写。
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
    if selected_mode in {"robot_test", "experiment"} and CONTROL_MODE == "sfc":
        raise ValueError(
            "当前版本只预留了 SFC 接口，尚未实现在线控制。"
            "为了避免把开环运动误当成 SFC，程序拒绝连接机器人。"
        )

    if selected_mode == "experiment" and EXECUTOR_MODE != "open_loop":
        raise ValueError("当前正式实验只实现 open_loop 轨迹执行器。")

    if selected_mode in {"robot_test", "experiment"} and not ROBOT_HOST.strip():
        raise ValueError("真机模式必须填写非空 ROBOT_HOST。")

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
