"""
图像来源、复合测量纸识别和二维位移估计。

这个文件只把“图像”变成“位移结果”。它不知道 UR10 要怎样运动，也不会直接写正式实验日志。

给初学者的文件地图：
1. FramePacket 是“一帧图像 + 时间信息”的小盒子，后面所有图像来源都会产出它。
2. ImageFolderSource、VideoSource、HikCameraSource 是三种“图像从哪里来”的实现。
3. preprocess_frame() 负责把彩色图变成更适合识别的灰度图，并记录亮度、模糊等质量指标。
4. CheckerboardTracker 负责识别棋盘格角点，并计算相对第一帧的位移。
5. CircleTracker 负责识别圆点、跨帧保持圆点身份，并计算相对第一帧的位移。
6. VisionProcessor 把预处理、两种识别方法、调试图绘制统一串起来。
7. run_vision_test() 是离线测试入口；camera_worker() 是正式实验时给 main.py 调用的子进程入口。

理解这个文件时可以抓住一个核心数据流：
图片/视频/相机帧 -> FramePacket -> preprocess_frame -> 棋盘格/圆点识别 -> 位移结果字典 -> 保存或发送给主程序。

安全边界：
- image_folder 和 video 不会连接任何硬件。
- HikCameraSource 只在 VISION_SOURCE="hik_camera" 时才会尝试导入海康 MVS SDK。
- 本文件不发送机器人运动指令。
"""

from __future__ import annotations

import importlib
import csv
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Full
from typing import Any, Iterator, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

import config
from calibration import image_points_to_spatial


_FONT_CACHE: dict[int, Any] = {}


# =============================================================================
# 1. 所有图像来源共用的数据外壳
# =============================================================================

@dataclass(slots=True)
class FramePacket:
    """
    一帧图像及其时间信息。

    frame_id 用于发现丢帧；host_ns 用于和 UR 记录对时；camera_timestamp_raw 保留相机原始时钟。

    字段解释：
    - frame：OpenCV 读到的图像数组，通常是 BGR 彩色图。
    - frame_id：第几帧，从 0 开始计数。
    - host_ns：电脑本地高精度时间戳，单位纳秒。
    - camera_timestamp_raw：图片/视频的相对时间，或真实相机的原始时间戳。
    - source_name：这一帧来自哪里，例如 image_folder:xxx.png 或 video:test.mp4。

    为什么要把这些信息放在一起：
    后面的视觉算法不仅需要图像本身，还需要知道这帧对应哪个时间点，
    否则就无法和机器人日志对齐。
    """

    frame: np.ndarray
    frame_id: int
    host_ns: int
    camera_timestamp_raw: float | int | None
    source_name: str


def _natural_sort_key(path: Path) -> list[int | str]:
    """
    生成“自然排序”用的 key，让 frame2.png 排在 frame10.png 前面。

    普通字符串排序会按字符比较：
    - frame10 会排在 frame2 前面，因为字符 '1' 小于 '2'。
    自然排序会把文件名里的数字片段当成真正的数字。
    """

    # re.split 会把文件名切成文字和数字两类片段。
    # 例如 frame12_test.png -> ["frame", "12", "_test.png"]。
    parts = re.split(r"(\d+)", path.name.lower())

    # 数字段转成 int，非数字段保持字符串。
    # sorted() 使用这个列表比较，就能得到更符合人类直觉的顺序。
    return [int(part) if part.isdigit() else part for part in parts]


def _load_preview_font(size: int) -> Any:
    """
    加载预览图左上角信息面板使用的中文字体。

    OpenCV 自带的 putText 只适合英文和数字，直接画中文会乱码或显示成方块。
    因此这里用 Pillow 从 Windows 字体目录加载微软雅黑/黑体/宋体，再把文字画回 OpenCV 图像。
    """

    # 字体加载相对慢，所以同一个字号只加载一次，后续帧复用缓存。
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    # 按常见 Windows 中文字体优先级寻找。
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    )
    for font_path in candidates:
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), size=size)
            _FONT_CACHE[size] = font
            return font

    # 理论上中文 Windows 都能找到上面的字体；找不到时退回 Pillow 默认字体。
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_chinese_panel(
    canvas: np.ndarray,
    lines: list[str],
    warning_lines: list[str],
) -> None:
    """
    在 OpenCV 图像左上角绘制中文信息面板。

    输入：
    - canvas：BGR 图像，会被原地修改；
    - lines：普通状态行；
    - warning_lines：需要红色显示的提示行。

    输出：
    - canvas 左上角出现半透明黑底、白字/红字的信息面板。
    """

    # Pillow 使用 RGB/RGBA，OpenCV 使用 BGR；先转换到 Pillow 方便画中文。
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_preview_font(22)
    warning_font = _load_preview_font(23)
    all_lines = lines + warning_lines
    if not all_lines:
        return

    # 计算信息面板尺寸，让文字不贴边也不超出图像。
    text_width = 0
    for text in all_lines:
        bbox = draw.textbbox((0, 0), text, font=warning_font if text in warning_lines else font)
        text_width = max(text_width, bbox[2] - bbox[0])

    line_height = 30
    panel_width = min(image.size[0] - 12, text_width + 28)
    panel_height = 16 + line_height * len(all_lines)

    # 半透明黑底保证白板、黑背景或棋盘格上都能看清文字。
    draw.rounded_rectangle(
        (6, 6, 6 + panel_width, 6 + panel_height),
        radius=6,
        fill=(0, 0, 0, 168),
    )

    # 普通行用白色，警告行用红色。
    y = 14
    for text in lines:
        draw.text((18, y), text, font=font, fill=(255, 255, 255, 255))
        y += line_height
    for text in warning_lines:
        draw.text((18, y), text, font=warning_font, fill=(255, 80, 80, 255))
        y += line_height

    # 合成后转回 OpenCV BGR，并写回原数组。
    composed = Image.alpha_composite(image, overlay).convert("RGB")
    canvas[:, :] = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


class ImageFolderSource:
    """
    离线图片序列输入源。

    输入：
    - config.IMAGE_FOLDER 指向的图片文件夹；
    - 可选 timestamps.csv，记录每张图片真实采集时间；
    - 没有 timestamps.csv 时使用 config.IMAGE_FOLDER_FPS 生成兜底时间轴。

    输出：
    - for 循环每次产出一个 FramePacket；
    - FramePacket 中包含图像数组、帧号、电脑读取时间、分析用时间和来源文件名。

    实验作用：
    这部分服务于预实验离线分析。它不连接相机，只把已经采集好的图片整理成和视频/相机一致的帧格式，
    让后面的 VisionProcessor 不需要关心图像到底来自哪里。

    适用场景：
    - 你还没有连接相机；
    - 你已经把视频拆成一张张图片；
    - 你想先离线验证圆点/棋盘格识别是否正常。

    用法上它是一个 context manager：
    with ImageFolderSource(...) as source:
        for packet in source:
            ...
    """

    def __init__(self, folder: Path, fps: float) -> None:
        # 本段保存“离线图片输入源”的基础配置。
        # folder 决定从哪里读图片；fps 只在没有真实时间戳时参与时间轴构造。
        self.folder = Path(folder)
        self.fps = float(fps)

        # 本段保存运行时发现的数据。
        # paths 是按时间顺序排列的图片列表；timestamps_* 是可选真实采集时间索引。
        self.paths: list[Path] = []
        self.timestamps_by_name: dict[str, float] = {}
        self.timestamps_by_index: dict[int, float] = {}

    @staticmethod
    def _load_timestamps(path: Path) -> tuple[dict[str, float], dict[int, float]]:
        """
        读取离线图片时间戳表。

        输入：timestamps.csv，支持 filename,time_s 或 frame_id,time_s。
        输出：按文件名或帧号索引的秒级时间戳。
        实验作用：让离线分析使用真实采集间隔，而不是强行假设每帧严格等于 1/fps。
        """

        by_name: dict[str, float] = {}
        by_index: dict[int, float] = {}

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(2048)
            file.seek(0)
            try:
                has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
            except csv.Error:
                has_header = False

            if has_header:
                reader = csv.DictReader(file)
                for row in reader:
                    time_text = row.get("time_s") or row.get("timestamp_s") or row.get("t")
                    if not time_text:
                        continue
                    time_s = float(time_text)
                    filename = row.get("filename") or row.get("file") or row.get("name")
                    frame_id = row.get("frame_id") or row.get("index")
                    if filename:
                        by_name[filename] = time_s
                    if frame_id not in (None, ""):
                        by_index[int(frame_id)] = time_s
            else:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) < 2:
                        continue
                    key = row[0].strip()
                    time_s = float(row[1])
                    if key.isdigit():
                        by_index[int(key)] = time_s
                    else:
                        by_name[key] = time_s

        return by_name, by_index

    def __enter__(self) -> "ImageFolderSource":
        # 本段完成离线输入源的启动检查。
        # 输入是 config.py 中指定的文件夹；输出是 self.paths 和可选时间戳索引。
        # 如果这里失败，说明后续视觉处理没有可靠图片序列可读，应在正式处理前直接停止。
        if not self.folder.exists():
            raise FileNotFoundError(
                f"图片文件夹不存在：{self.folder}\n"
                "请创建该文件夹并放入按时间排序的图片，或修改 config.py 的 IMAGE_FOLDER。"
            )

        # 本段把文件夹中的图片整理成稳定的时间顺序。
        # 自然排序让 frame2 排在 frame10 前面，避免文件名排序破坏位移-时间曲线。
        self.paths = sorted(
            (
                path
                for path in self.folder.iterdir()
                if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
            ),
            key=_natural_sort_key,
        )

        # 本段阻止“空输入”悄悄进入后续流程。
        # 没有图片时，生成空结果比直接报错更难排查，所以这里立即说明问题。
        if not self.paths:
            raise FileNotFoundError(
                f"图片文件夹中没有支持的图像：{self.folder}\n"
                f"支持的扩展名为：{', '.join(config.IMAGE_EXTENSIONS)}。"
            )

        # 本段尝试载入真实采集时间。
        # 有 timestamps.csv 时，后续每帧使用真实时间；没有时按 IMAGE_FOLDER_FPS 兜底。
        # 如果你明确要求时间戳文件，则缺失会在这里报错，避免后续频率分析使用错误时间轴。
        timestamp_path = config.IMAGE_TIMESTAMPS_CSV
        if timestamp_path is not None:
            timestamp_path = Path(timestamp_path)
            if timestamp_path.exists():
                self.timestamps_by_name, self.timestamps_by_index = self._load_timestamps(
                    timestamp_path
                )
                print(f"[视觉] 已读取图片时间戳：{timestamp_path}")
            elif config.IMAGE_TIMESTAMPS_REQUIRED:
                raise FileNotFoundError(f"要求使用图片时间戳，但文件不存在：{timestamp_path}")

        print(f"[视觉] 已找到 {len(self.paths)} 张图片：{self.folder}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # 图片文件夹没有相机句柄需要释放；保留该方法是为了和视频/相机输入源使用同一种 with 写法。
        return None

    def __iter__(self) -> Iterator[FramePacket]:
        # 本循环是“离线图片 -> FramePacket”的转换流水线。
        # 输入是已经排序好的图片路径；输出是后续视觉算法统一消费的 FramePacket。
        # 每张图片只在即将处理时读取，不会一次性把所有图像加载进内存。
        for index, path in enumerate(self.paths):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)

            # host_ns 表示电脑读取到这一帧的时间；camera_time_s 表示离线分析应使用的采集时间。
            # 对图片文件夹来说，真正影响频谱的是 camera_time_s，而不是电脑解码这一帧花了多久。
            host_ns = time.perf_counter_ns()

            if frame is None:
                print(f"[视觉警告] OpenCV 无法读取，已跳过：{path.name}")
                continue

            camera_time_s = self.timestamps_by_name.get(
                path.name,
                self.timestamps_by_index.get(index, index / self.fps),
            )
            yield FramePacket(
                frame,
                index,
                host_ns,
                camera_time_s,
                f"image_folder:{path.name}",
            )


class VideoSource:
    """
    离线视频输入源。

    输入：
    - config.VIDEO_PATH 指向的视频文件；
    - 视频容器里的帧率和时间戳信息。

    输出：
    - for 循环每次产出一个 FramePacket；
    - FramePacket 中包含当前视频帧、帧号、电脑读取时间和视频自身时间。

    实验作用：
    这部分用于把已录好的视频当成图片序列分析。它不连接真实相机，
    但会尽量保留视频内部时间戳，让后续频谱分析使用采集时间，而不是电脑解码速度。

    它和 ImageFolderSource 的共同点：
    - 都会产出 FramePacket；
    - 都不会连接海康 SDK；
    - 都适合离线调试。

    它和 ImageFolderSource 的区别：
    - 图片文件夹的时间来自 frame_id / IMAGE_FOLDER_FPS；
    - 视频优先使用视频文件内部的时间戳 CAP_PROP_POS_MSEC。
    """

    def __init__(self, video_path: Path) -> None:
        # 本段保存视频输入源的基础配置。
        # video_path 是离线视频文件；capture 和 fps 会在进入 with 时由 OpenCV 打开后确定。
        self.video_path = Path(video_path)
        self.capture: cv2.VideoCapture | None = None
        self.fps = 0.0

    def __enter__(self) -> "VideoSource":
        # 本段完成视频输入源的启动检查。
        # 输入是 config.py 指定的视频路径；输出是可逐帧读取的 OpenCV VideoCapture。
        # 如果这里失败，说明后续视觉处理没有可靠帧序列可读，应在正式处理前直接停止。
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"视频不存在：{self.video_path}\n"
                "请修改 config.py 的 VIDEO_PATH，或切换回 image_folder。"
            )

        # 本段打开视频文件并读取视频时间信息。
        # fps 只作为兜底；若视频提供每帧时间戳，后续会优先用 CAP_PROP_POS_MSEC。
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频：{self.video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
            print("[视觉警告] 视频未提供有效帧率，暂按 30 fps 记录相对时间。")

        print(f"[视觉] 已打开视频：{self.video_path}，标称帧率 {self.fps:.3f} fps")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # 本段释放视频文件句柄。
        # 输出是关闭后的 capture=None，避免下次运行仍占用文件或解码器资源。
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def __iter__(self) -> Iterator[FramePacket]:
        # 本循环是“离线视频 -> FramePacket”的转换流水线。
        # 输入是 OpenCV 解码出的下一帧；输出是后续视觉算法统一消费的 FramePacket。
        # 和图片文件夹一样，这里逐帧读、逐帧交给后续处理，不一次性解码完整视频。
        if self.capture is None:
            raise RuntimeError("VideoSource 必须放在 with 语句中使用。")

        frame_id = 0
        while True:
            ok, frame = self.capture.read()
            host_ns = time.perf_counter_ns()
            if not ok:
                break

            # 本段决定视频帧用于离线分析的时间。
            # 优先使用视频容器时间戳；取不到时才用 frame_id/fps 估算。
            position_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            camera_time_s = position_ms / 1000.0 if position_ms > 0 else frame_id / self.fps

            yield FramePacket(
                frame,
                frame_id,
                host_ns,
                camera_time_s,
                f"video:{self.video_path.name}",
            )
            frame_id += 1


class HikCameraSource:
    """
    海康 MVS 实时相机输入源。

    输入：
    - MVS SDK；
    - config.py 中的相机序列号、曝光、增益和取帧超时；
    - 相机连续取流返回的图像缓冲区。

    输出：
    - for 循环每次产出一个 FramePacket；
    - FramePacket 中包含相机帧号、电脑接收时间、相机原始时间戳和图像数组。

    实验作用：
    这部分只负责把海康相机接入统一的视觉流水线。后面的识别算法仍然只看 FramePacket，
    不需要知道 MVS 的设备枚举、句柄、缓冲区和像素格式细节。

    MVS 不同版本的包装类字段可能略有区别，因此所有厂商调用都集中在这里，算法部分无需跟着改。
    """

    def __init__(self) -> None:
        # 本段保存海康相机运行时资源。
        # sdk/camera/device_info 属于 MVS 控制层；payload/buffer/frame_info 属于连续取帧层。
        # 它们在 __enter__ 中创建，在 close() 中按顺序释放。
        self.sdk: Any = None
        self.camera: Any = None
        self.device_info: Any = None
        self.payload_size = 0
        self.data_buffer: Any = None
        self.frame_info: Any = None

    @staticmethod
    def _decode_c_string(value: Any) -> str:
        """
        把 SDK 的定长 C 字符数组转换成 Python 字符串。

        输入：MVS 设备信息结构里的序列号字段。
        输出：普通字符串，用于显示和按序列号选择相机。
        实验作用：多相机场景下必须能读出序列号，避免预实验连接到错误相机。
        """

        try:
            raw = bytes(value)
        except TypeError:
            raw = bytearray(value)
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    def _load_sdk(self) -> Any:
        # 本段只在 VISION_SOURCE="hik_camera" 时加载 MVS。
        # 输入是可选 HIK_MVS_IMPORT_PATH；输出是 MvCameraControl_class 模块。
        # 离线图片/视频不会执行这里，因此没有安装 MVS 也不影响预实验离线分析。
        if config.HIK_MVS_IMPORT_PATH:
            import_path = str(Path(config.HIK_MVS_IMPORT_PATH).expanduser().resolve())
            if import_path not in sys.path:
                sys.path.insert(0, import_path)

        try:
            return importlib.import_module("MvCameraControl_class")
        except ImportError as exc:
            raise RuntimeError(
                "当前选择了 hik_camera，但没有找到 MvCameraControl_class。\n"
                "请先安装海康 MVS，并在 config.py 填写 HIK_MVS_IMPORT_PATH；"
                "当前无相机时可改用 image_folder 或 video。"
            ) from exc

    def _device_serial(self, device_info: Any) -> str:
        """
        从 MVS 设备结构中读取相机序列号。

        输入：枚举得到的一台相机 device_info。
        输出：序列号字符串，无法读取时返回空字符串。
        实验作用：正式预实验中建议指定序列号，保证数据来自预期那台相机。
        """

        transport = int(device_info.nTLayerType)
        gige_type = int(getattr(self.sdk, "MV_GIGE_DEVICE", -1))
        usb_type = int(getattr(self.sdk, "MV_USB_DEVICE", -2))

        if transport == gige_type:
            return self._decode_c_string(device_info.SpecialInfo.stGigEInfo.chSerialNumber)
        if transport == usb_type:
            return self._decode_c_string(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        return ""

    def _select_device(self, device_list: Any) -> Any:
        """
        从枚举结果中选出本次要使用的相机。

        输入：MVS 枚举到的设备列表，以及 config.HIK_CAMERA_SERIAL。
        输出：被选中的 device_info。
        实验作用：有序列号时严格匹配；没有序列号时给出警告后使用第一台，避免使用者误以为已经精确选机。
        """

        from ctypes import POINTER, cast

        device_info_type = getattr(self.sdk, "MV_CC_DEVICE_INFO")
        candidates: list[tuple[str, Any]] = []

        for index in range(int(device_list.nDeviceNum)):
            pointer = cast(device_list.pDeviceInfo[index], POINTER(device_info_type))
            info = pointer.contents
            candidates.append((self._device_serial(info), info))

        if not candidates:
            raise RuntimeError("MVS 没有枚举到工业相机，请检查供电、网线/USB 和相机 IP。")

        requested = config.HIK_CAMERA_SERIAL.strip()
        if requested:
            for serial, info in candidates:
                if serial == requested:
                    print(f"[视觉] 选择海康相机序列号：{serial}")
                    return info
            available = ", ".join(serial or "<无法读取>" for serial, _ in candidates)
            raise RuntimeError(f"没有找到序列号 {requested}；当前枚举结果：{available}")

        print(
            "[视觉警告] HIK_CAMERA_SERIAL 为空，将使用枚举到的第一台相机："
            f"{candidates[0][0] or '<序列号无法读取>'}"
        )
        return candidates[0][1]

    def _check_ret(self, ret: int, action: str) -> None:
        """
        统一检查 MVS 调用返回值。

        输入：MVS 返回码和当前动作名称。
        输出：成功时不返回内容；失败时抛出包含十六进制错误码的异常。
        实验作用：把硬件接口错误尽早变成可读报错，方便定位是枚举、打开、取流还是参数设置失败。
        """

        if int(ret) != 0:
            raise RuntimeError(f"{action}失败，MVS 返回码 0x{int(ret) & 0xFFFFFFFF:08X}")

    def _try_set_float(self, node_name: str, value: float | None) -> None:
        """
        尝试设置相机浮点参数，例如曝光或增益。

        输入：MVS 节点名和目标数值；value=None 表示保持相机当前配置。
        输出：设置成功或打印警告。
        实验作用：曝光/增益会影响识别质量，但不同相机节点范围不同，因此失败时提醒使用者去 MVS 客户端核对。
        """

        if value is None:
            return
        ret = self.camera.MV_CC_SetFloatValue(node_name, float(value))
        if int(ret) != 0:
            print(
                f"[视觉警告] 无法设置 {node_name}={value}，"
                f"MVS 返回 0x{int(ret) & 0xFFFFFFFF:08X}。请在 MVS 客户端确认节点范围。"
            )

    def __enter__(self) -> "HikCameraSource":
        from ctypes import c_ubyte

        # 本段完成“找到相机并独占打开”的硬件启动流程。
        # 输入是 MVS SDK 和配置中的相机序列号；输出是可控制、可取流的 camera 句柄。
        self.sdk = self._load_sdk()
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        transport_mask = int(self.sdk.MV_GIGE_DEVICE) | int(self.sdk.MV_USB_DEVICE)

        ret = self.sdk.MvCamera.MV_CC_EnumDevices(transport_mask, device_list)
        self._check_ret(ret, "枚举海康相机")
        self.device_info = self._select_device(device_list)

        self.camera = self.sdk.MvCamera()
        self._check_ret(self.camera.MV_CC_CreateHandle(self.device_info), "创建相机句柄")
        self._check_ret(
            self.camera.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0),
            "独占打开相机",
        )

        try:
            # 本段配置相机取流状态。
            # 输出是连续自由运行的相机流；曝光/增益按配置尝试设置，PayloadSize 决定接收缓冲区大小。
            if int(self.device_info.nTLayerType) == int(self.sdk.MV_GIGE_DEVICE):
                packet_size = int(self.camera.MV_CC_GetOptimalPacketSize())
                if packet_size > 0:
                    self.camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

            self.camera.MV_CC_SetEnumValue("TriggerMode", int(self.sdk.MV_TRIGGER_MODE_OFF))

            if config.HIK_EXPOSURE_US is not None:
                self.camera.MV_CC_SetEnumValue("ExposureAuto", 0)
            self._try_set_float("ExposureTime", config.HIK_EXPOSURE_US)

            if config.HIK_GAIN is not None:
                self.camera.MV_CC_SetEnumValue("GainAuto", 0)
            self._try_set_float("Gain", config.HIK_GAIN)

            payload = self.sdk.MVCC_INTVALUE_EX()
            self._check_ret(
                self.camera.MV_CC_GetIntValueEx("PayloadSize", payload),
                "读取 PayloadSize",
            )
            self.payload_size = int(payload.nCurValue)
            self.data_buffer = (c_ubyte * self.payload_size)()
            self.frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()

            self._check_ret(self.camera.MV_CC_StartGrabbing(), "开始取流")
        except Exception:
            # 本段处理半打开失败。
            # 如果配置或取流启动中途失败，必须释放句柄，否则下一次运行可能显示设备仍被占用。
            self.close()
            raise

        print("[视觉] 海康相机已打开并开始连续取流。")
        return self

    def _convert_raw_frame(self, frame_info: Any) -> np.ndarray:
        """
        把 MVS 原始缓冲区转换成 OpenCV 图像。

        输入：MVS 当前帧信息和 self.data_buffer 中的原始字节。
        输出：灰度图或 BGR 图像数组，供 preprocess_frame() 继续处理。
        实验作用：工业相机可能输出 Mono8、RGB/BGR 或 Bayer 格式；这里统一成 OpenCV 能识别的图像格式。
        """

        # 本段读取当前帧的尺寸、像素格式和原始字节范围。
        # 输出 raw 仍然只是字节视图，后面会根据 pixel_type 解释成图像。
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        pixel_type = int(frame_info.enPixelType)
        raw = np.ctypeslib.as_array(self.data_buffer)[: int(frame_info.nFrameLen)]

        mono8 = getattr(self.sdk, "PixelType_Gvsp_Mono8", None)
        bgr8 = getattr(self.sdk, "PixelType_Gvsp_BGR8_Packed", None)
        rgb8 = getattr(self.sdk, "PixelType_Gvsp_RGB8_Packed", None)

        # 本段处理已经是灰度或彩色打包的简单格式。
        # 输出会 copy 一份，避免下一次相机取帧覆盖同一块底层缓冲区时影响当前图像。
        if mono8 is not None and pixel_type == int(mono8):
            return raw.reshape(height, width).copy()

        if bgr8 is not None and pixel_type == int(bgr8):
            return raw.reshape(height, width, 3).copy()

        if rgb8 is not None and pixel_type == int(rgb8):
            rgb = raw.reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 本段处理常见 Bayer8 格式。
        # 输入仍是单通道原始图；输出是插值得到的 BGR 图，用于和其他来源保持一致。
        bayer_codes = {
            getattr(self.sdk, "PixelType_Gvsp_BayerGR8", -101): cv2.COLOR_BayerGR2BGR,
            getattr(self.sdk, "PixelType_Gvsp_BayerRG8", -102): cv2.COLOR_BayerRG2BGR,
            getattr(self.sdk, "PixelType_Gvsp_BayerGB8", -103): cv2.COLOR_BayerGB2BGR,
            getattr(self.sdk, "PixelType_Gvsp_BayerBG8", -104): cv2.COLOR_BayerBG2BGR,
        }
        if pixel_type in {int(key) for key in bayer_codes}:
            code = next(code for key, code in bayer_codes.items() if int(key) == pixel_type)
            return cv2.cvtColor(raw.reshape(height, width), code)

        raise RuntimeError(
            f"当前 MVS 像素格式 0x{pixel_type:08X} 尚未转换。"
            "请先在 MVS 客户端把 PixelFormat 设为 Mono8、BGR8Packed 或常见 Bayer8。"
        )

    def __iter__(self) -> Iterator[FramePacket]:
        # 本循环是“真实相机帧 -> FramePacket”的实时输入流水线。
        # 输入是 MVS 连续取流；输出是统一格式的 FramePacket。
        # 这里会阻塞等待下一帧，超过 HIK_FRAME_TIMEOUT_MS 就报错，避免实验长时间无声卡住。
        if self.camera is None:
            raise RuntimeError("HikCameraSource 必须放在 with 语句中使用。")

        from ctypes import byref, memset, sizeof

        while True:
            memset(byref(self.frame_info), 0, sizeof(self.frame_info))
            ret = self.camera.MV_CC_GetOneFrameTimeout(
                self.data_buffer,
                self.payload_size,
                self.frame_info,
                int(config.HIK_FRAME_TIMEOUT_MS),
            )
            host_ns = time.perf_counter_ns()

            if int(ret) != 0:
                raise TimeoutError(
                    "海康相机在规定时间内没有返回图像，"
                    f"MVS 返回 0x{int(ret) & 0xFFFFFFFF:08X}。"
                )

            # 本段把相机帧变成后续视觉算法可消费的记录。
            # host_ns 用于和 UR/事件日志对时；timestamp_raw 保留相机原始硬件时间供以后扩展。
            frame = self._convert_raw_frame(self.frame_info)
            timestamp_raw = (
                (int(self.frame_info.nDevTimeStampHigh) << 32)
                | int(self.frame_info.nDevTimeStampLow)
            )

            yield FramePacket(
                frame=frame,
                frame_id=int(self.frame_info.nFrameNum),
                host_ns=host_ns,
                camera_timestamp_raw=timestamp_raw,
                source_name="hik_camera",
            )

    def close(self) -> None:
        """
        释放海康相机相关资源。

        输入：当前可能已经打开到一半或完全打开的 camera 句柄。
        输出：停止取流、关闭设备、销毁句柄后的空闲状态。
        实验作用：无论正常结束还是异常退出，都尽量避免相机被上一次程序占用。
        """

        if self.camera is None:
            return

        for method_name in ("MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"):
            method = getattr(self.camera, method_name, None)
            if method is not None:
                try:
                    method()
                except Exception:
                    pass

        self.camera = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def open_image_source() -> ImageFolderSource | VideoSource | HikCameraSource:
    """
    根据配置创建本次视觉流程唯一的图像来源。

    输入：config.VISION_SOURCE。
    输出：ImageFolderSource、VideoSource 或 HikCameraSource 中的一个。

    实验作用：
    后续代码只面对统一的 FramePacket，不关心帧来自图片、视频还是相机。
    同时，未选择的硬件库不会被导入，避免离线分析因为缺 MVS SDK 而失败。

    初学者可以把它看成“开关分流器”：
    - VISION_SOURCE="image_folder"：从 input_images 逐张读图片；
    - VISION_SOURCE="video"：从 input_video/test.mp4 逐帧读视频；
    - VISION_SOURCE="hik_camera"：才会创建海康相机对象。
    """

    if config.VISION_SOURCE == "image_folder":
        return ImageFolderSource(config.IMAGE_FOLDER, config.IMAGE_FOLDER_FPS)
    if config.VISION_SOURCE == "video":
        return VideoSource(config.VIDEO_PATH)
    if config.VISION_SOURCE == "hik_camera":
        return HikCameraSource()
    raise ValueError(f"未知 VISION_SOURCE：{config.VISION_SOURCE}")


# =============================================================================
# 2. 图像预处理与通用运动拟合
# =============================================================================

def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, dict[str, float], tuple[int, int]]:
    """
    把原始图像转换成视觉识别输入。

    原始图片 frame
    ↓
    检查是不是空图
    ↓
    可选：畸变校正
    ↓
    可选：裁剪 ROI
    ↓
    转成灰度图 gray
    ↓
    可选：轻微高斯模糊
    ↓
    计算画质指标 metrics
    ↓
    返回 gray, metrics, roi_origin

    输入：FramePacket.frame，即图片/视频/相机来源提供的原始 OpenCV 图像。
    输出：
    - gray：后续圆点和棋盘格算法使用的灰度图；
    - metrics：亮度、模糊、过暗/过曝比例；
    - roi_origin：ROI 左上角在原图中的位置，用于把识别点画回完整图像。

    实验作用：
    这里不计算位移，只做“让图像适合识别”和“记录图像质量”。
    如果某帧识别失败，metrics 能帮助判断是算法问题，还是曝光/模糊/ROI 设置问题。
    """

    if frame is None or frame.size == 0:
        raise ValueError("收到空图像，无法预处理。")

    # 本段建立识别工作图像。
    # 输入是原始 frame；输出 working 可能经过畸变校正和 ROI 裁剪，但原始 frame 仍保留给调试图使用。
    working = frame

    # 本段可选做镜头畸变校正。如果你填了相机内参和畸变参数，就先把图像做去畸变。如果没填，就跳过。
    # 输入是相机内参/畸变；输出是几何上更接近真实投影的图像，有利于像素坐标和空间坐标一致。
    if config.CAMERA_MATRIX is not None:
        camera_matrix = np.asarray(config.CAMERA_MATRIX, dtype=np.float64)
        distortion = np.asarray(config.DISTORTION_COEFFICIENTS, dtype=np.float64)
        working = cv2.undistort(working, camera_matrix, distortion)

    # 本段可选裁剪 ROI。
    # 输入是完整图像和 VISION_ROI；输出是只包含测量纸附近区域的 working。
    # 实验中固定相机后，ROI 能减少背景误识别，也能加快离线批处理。
    roi_origin = (0, 0)
    if config.VISION_ROI is not None:
        x, y, width, height = config.VISION_ROI
        if x + width > working.shape[1] or y + height > working.shape[0]:
            raise ValueError(
                f"VISION_ROI={config.VISION_ROI} 超出当前图像尺寸 "
                f"{working.shape[1]}×{working.shape[0]}。"
            )
        working = working[y : y + height, x : x + width]
        roi_origin = (x, y)

    # 本段把工作图像统一成灰度图。
    # 输出 gray 是后续棋盘格角点和圆点轮廓检测的共同输入。
    if working.ndim == 2:
        gray = working.copy()
    else:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    # 本段保留一份“未做高斯模糊”的灰度图，专门用于计算清晰度。
    # 如果在已经模糊后的 gray 上计算 Laplacian 方差，清晰度数值会被人为压低，
    # 尤其是白色背景板占画面大部分时，更容易出现过于敏感的 blurred warning。
    sharpness_gray = gray.copy()

    # 本段可选做轻微高斯滤波。
    # 输入是灰度图；输出是噪声略低的灰度图。
    # 这有助于阈值分割稳定，但核太大会吃掉边缘，所以由 config.py 控制。
    if config.GAUSSIAN_BLUR_KERNEL > 1:
        gray = cv2.GaussianBlur(
            gray,
            (config.GAUSSIAN_BLUR_KERNEL, config.GAUSSIAN_BLUR_KERNEL),
            0,
        )

    # 本段生成图像质量指标。
    # 输出写入每帧 VISION 记录，以后如果某一帧识别失败，用于回头解释识别失败或位移异常。
    metrics = {
        "mean_brightness": float(np.mean(gray)),#平均亮度
        "blur_variance": float(cv2.Laplacian(sharpness_gray, cv2.CV_64F).var()),#清晰度指标
        "dark_fraction": float(np.mean(gray <= config.DARK_PIXEL_THRESHOLD)),#过暗的像素的比例
        "bright_fraction": float(np.mean(gray >= config.BRIGHT_PIXEL_THRESHOLD)),#过亮的像素的比例
    }
    return gray, metrics, roi_origin


def _estimate_rigid_motion(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    mm_per_pixel: float,
    residual_warning_px: float,
) -> dict[str, float | bool]:
    """
    用一组对应点估计当前帧相对参考帧的整体运动。

    输入：
    - reference_points：参考帧中的点坐标；
    - current_points：当前帧中与参考点一一对应的点坐标；
    - mm_per_pixel：像素位移换算成毫米的比例；
    - residual_warning_px：残差质量评分的参考阈值。

    输出：
    - dx_mm / dy_mm：当前帧相对参考帧的整体平移；
    - angle_deg / scale：平面内转角和尺度变化；
    - residual_px / quality：拟合残差和质量分。
    
    
    我已经拿到了两组对应点了吗？
    输入来自：
    - CheckerboardTracker.process() 传入的 reference_corners 和 corners
    或者
    - CircleTracker.process() 传入的 reference_points 和 current_points
    ↓
    把输入点统一整理成 N×2 数组
    ↓
    调用opencv2, 根据标定图案移动距离，计算移动量（像素）
    ↓
    用camera.py 的 CheckerboardTracker._calculate_mm_per_pixel()把像素位移换成毫米位移
    ↓
    返回 dx_mm / dy_mm / angle_deg旋转角度 / scale尺度变化 / residual_px拟合误差(可以理解为置信度) / quality可信点比例


    实验作用：
    棋盘格和圆点法最后都会走到这里。它把“多个点的像素坐标变化”压缩成
    “测量纸整体在图像平面内怎么动”，这是后续振动分析的位移来源。
    """

    # 本段统一输入点格式。
    # 输出 ref/cur 都是 N×2 坐标数组，保证后续拟合函数看到的是同一种数据结构。
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)

    # 本段检查是否有足够点建立刚体运动。
    # 少于 3 个点或点数不对应时，无法可靠估计平移/旋转/尺度，因此返回无效结果。
    if len(ref) < 3 or len(cur) != len(ref):
        return {
            "is_valid": False,
            "dx_mm": math.nan,
            "dy_mm": math.nan,
            "angle_deg": math.nan,
            "scale": math.nan,
            "residual_px": math.nan,
            "quality": 0.0,
        }

    # 本段拟合参考点到当前点的相似变换。
    # 输入是一一对应的像素点；输出 matrix 描述整体平移、旋转和等比例缩放。
    # RANSAC 让少量误识别点不至于拖垮整帧结果。
    matrix, inliers = cv2.estimateAffinePartial2D(
        ref,
        cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=max(1.0, residual_warning_px * 2.0),
        maxIters=2000,
        confidence=0.99,
        refineIters=20,
    )
    if matrix is None:
        return {
            "is_valid": False,
            "dx_mm": math.nan,
            "dy_mm": math.nan,
            "angle_deg": math.nan,
            "scale": math.nan,
            "residual_px": math.nan,
            "quality": 0.0,
        }

    # 本段评估“拟合出来的整体运动”是否能解释当前点。
    # 输出 errors 和 inlier_mask 用于计算质量分；残差大通常说明串点、漏点或图像质量差。
    predicted = cv2.transform(ref.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - cur, axis=1)

    if inliers is None:
        inlier_mask = np.ones(len(ref), dtype=bool)
    else:
        inlier_mask = inliers.ravel().astype(bool)
    if not np.any(inlier_mask):
        inlier_mask[:] = True

    # 本段把像素级整体运动整理成实验记录字段。
    # 平移量使用可信点平均位移，再乘 mm_per_pixel 换成毫米；角度和尺度从拟合矩阵读取。
    centroid_shift = np.mean(cur[inlier_mask] - ref[inlier_mask], axis=0)
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    angle_deg = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    residual = float(np.sqrt(np.mean(errors[inlier_mask] ** 2)))

    # 本段计算质量分。
    # 内点比例反映“多少点支持同一个整体运动”；残差反映“这个整体运动解释点坐标的精度”。
    inlier_ratio = float(np.mean(inlier_mask))
    residual_score = math.exp(-residual / max(residual_warning_px, 1e-6))
    quality = float(np.clip(inlier_ratio * residual_score, 0.0, 1.0))

    return {
        "is_valid": True,
        "dx_mm": float(centroid_shift[0] * mm_per_pixel),
        "dy_mm": float(centroid_shift[1] * mm_per_pixel),
        "angle_deg": angle_deg,
        "scale": scale,
        "residual_px": residual,
        "quality": quality,
    }


# =============================================================================
# 3. 棋盘格识别
# =============================================================================

class CheckerboardTracker:
    """
    棋盘格法的跨帧状态管理器。

    输入：每帧预处理后的灰度图。
    输出：checker_* 字段和当前帧角点像素坐标。

    实验作用：
    第一次成功识别到棋盘格时，把角点保存为参考状态；
    后续帧都与这组参考角点比较，得到相对起始状态的二维位移、转角和质量分。
    """

    def __init__(self) -> None:
        # 本段保存棋盘格参考状态。
        # reference_corners 是第一帧角点；mm_per_pixel 是由参考帧方格尺寸估计的像素/毫米关系。
        self.reference_corners: np.ndarray | None = None
        self.mm_per_pixel: float | None = None

    def _find_corners(self, gray: np.ndarray) -> np.ndarray | None:
        """
        在一帧灰度图中寻找棋盘格角点。

        输入：preprocess_frame() 输出的灰度图。
        输出：N×2 的角点像素坐标，找不到时返回 None。
        实验作用：这是棋盘格法的原始检测步骤，后面的位移计算完全依赖这些角点是否稳定。
        """

        # 本段准备棋盘格规格。
        # 输入是 config.py 里的内角点数量；输出是 OpenCV 需要的 pattern。
        pattern = tuple(int(value) for value in config.CHECKERBOARD_INNER_CORNERS)

        # 本段优先使用 OpenCV 的 SB 棋盘格检测。
        # 输出是亚像素级角点坐标；如果当前 OpenCV 没有 SB 方法，后面会自动退回传统检测。
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
            if found:
                return corners.reshape(-1, 2).astype(np.float32)

        # 本段是传统棋盘格检测兜底。
        # 输入仍是同一张灰度图；输出会再经过 cornerSubPix 细化，尽量减少角点量化误差。
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not found:
            return None

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            int(config.CHECKER_SUBPIX_MAX_ITER),
            float(config.CHECKER_SUBPIX_EPS),
        )
        refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        return refined.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _calculate_mm_per_pixel(corners: np.ndarray) -> float:
        """
        由棋盘格角点估计图像平面的毫米/像素比例。

        输入：参考帧中识别到的棋盘格角点。
        输出：mm_per_pixel。
        实验作用：把像素位移转换成纸面内毫米位移，让后续 dx/dy 不只是像素数。
        """

        # 本段把角点还原成棋盘格网格结构。
        # 输出 grid 让横向/纵向相邻点间距都能参与比例尺估计。
        columns, rows = config.CHECKERBOARD_INNER_CORNERS
        grid = corners.reshape(rows, columns, 2)

        horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
        vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()

        # 本段用相邻角点距离的中位数建立比例尺。
        # 中位数能降低个别角点抖动或局部模糊对比例尺的影响。
        spacing_px = float(np.median(np.concatenate((horizontal, vertical))))

        if spacing_px <= 0:
            raise ValueError("棋盘格角点间距为零，无法建立毫米比例。")
        return float(config.CHECKER_SQUARE_MM / spacing_px)

    def process(self, gray: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        """
        完成一帧棋盘格法测量。

        输入：当前帧灰度图。
        输出：
        - checker_* 结果字典；
        - 当前帧角点坐标，用于日志保存和调试图绘制。

        实验作用：
        这里把“当前帧棋盘格角点”转换成“相对第一帧的位移/转角/质量”。
        """

        # 本段先做原始角点检测。
        # 找不到角点时仍返回固定字段，保证逐帧 JSON 结构稳定，后续分析可以明确跳过无效帧。
        corners = self._find_corners(gray)
        if corners is None:
            return {
                "checker_is_valid": False,
                "checker_dx_mm": math.nan,
                "checker_dy_mm": math.nan,
                "checker_angle_deg": math.nan,
                "checker_scale": math.nan,
                "checker_residual_px": math.nan,
                "checker_quality": 0.0,
                "checker_corner_count": 0,
            }, None

        # 本段建立棋盘格参考状态。
        # 第一次成功识别的角点就是位移零点；后续帧都和它比较。
        if self.reference_corners is None:
            self.reference_corners = corners.copy()
            self.mm_per_pixel = self._calculate_mm_per_pixel(corners)

        # 本段读取参考状态并防御异常流程。
        # 如果 reference 或比例尺缺失，说明初始化链路被破坏，应直接报错。
        reference_corners = self.reference_corners
        mm_per_pixel = self.mm_per_pixel
        if reference_corners is None or mm_per_pixel is None:
            raise RuntimeError("Checkerboard reference was not initialized.")

        # 本段把角点坐标变化转换成整体运动。
        # 输入是参考角点和当前角点；输出是毫米位移、转角、尺度和质量分。
        motion = _estimate_rigid_motion(
            reference_corners,
            corners,
            float(mm_per_pixel),
            config.CHECKER_RESIDUAL_WARNING_PX,
        )

        # 本段整理当前帧棋盘格法日志。
        # 输出字段统一以 checker_ 开头，避免和圆点法结果混淆。
        result = {
            "checker_is_valid": bool(motion["is_valid"]),
            "checker_dx_mm": motion["dx_mm"],
            "checker_dy_mm": motion["dy_mm"],
            "checker_angle_deg": motion["angle_deg"],
            "checker_scale": motion["scale"],
            "checker_residual_px": motion["residual_px"],
            "checker_quality": motion["quality"],
            "checker_corner_count": int(len(corners)),
            "checker_mm_per_pixel": float(mm_per_pixel),
        }
        return result, corners


# =============================================================================
# 4. 圆点识别与跨帧身份保持
# =============================================================================

def _find_circle_candidates(gray: np.ndarray) -> list[dict[str, float]]:
    """
    从一帧灰度图中寻找圆点候选。

    输入：preprocess_frame() 输出的灰度图。
    输出：候选圆点列表，每个候选包含圆心、面积、圆度、轴比和直径。

    实验作用：
    这里还没有决定圆点身份，也不计算位移；它只回答“这一帧里哪些黑色轮廓可能是圆点”。
    后续 CircleTracker 会再根据第一帧参考和上一帧位置判断这些候选点是否可信。
    """

    # 本段把灰度图转换成轮廓检测用的二值图。
    # Otsu 会根据当前亮度自动选阈值；反色后黑色标记变成白色前景，方便 findContours。
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    # 本段提取所有前景轮廓。
    # 输出 contours 只是候选边界集合，里面可能混有噪声、反光、棋盘格边缘或背景圆形物。
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates: list[dict[str, float]] = []
    for contour in contours:
        # 本段按尺寸排除明显不是目标圆点的轮廓。
        # 面积阈值来自 config.py，通常需要根据相机距离和分辨率实拍调整。
        area = float(cv2.contourArea(contour))
        if area < config.CIRCLE_MIN_AREA_PX or area > config.CIRCLE_MAX_AREA_PX:
            continue

        # 本段检查轮廓是否足够描述形状。
        # 周长为 0 或轮廓点太少时，圆度和椭圆拟合都没有意义。
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0 or len(contour) < 5:
            continue

        # 本段按圆度筛选。
        # 输出通过筛选的轮廓更接近圆形，但透视下允许它不是完美圆。
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < config.CIRCLE_MIN_CIRCULARITY:
            continue

        # 本段用多边形顶点数排除棋盘格方块。
        # 方格边界通常近似为 4 个顶点，圆点边界会有更多顶点。
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) < 7:
            continue

        # 本段用椭圆轴比保留透视下的圆点。
        # 真实圆点在斜视时会变成椭圆，因此只排除过于细长的轮廓。
        ellipse = cast(
            tuple[tuple[float, float], tuple[float, float], float],
            cv2.fitEllipse(contour),
        )
        (cx, cy), (axis_a, axis_b), _ = ellipse
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if major <= 0 or minor / major < config.CIRCLE_MIN_AXIS_RATIO:
            continue

        # 本段把通过筛选的轮廓整理成普通数值字段。
        # 输出字段既用于后续圆心匹配，也方便未来需要时写入更详细的候选点调试日志。
        candidates.append(
            {
                "x": float(cx),
                "y": float(cy),
                "area": area,
                "circularity": circularity,
                "axis_ratio": minor / major,
                "diameter_px": 0.5 * (major + minor),
            }
        )

    # 本段限制候选点数量。
    # 若环境里有其他圆孔或反光点，先保留面积较大的预期数量，再交给跨帧匹配继续筛选。
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates[: config.CIRCLE_EXPECTED_COUNT]


def _layout_span_mm() -> float:
    """
    计算圆点理论布局的最大跨度。

    输入：config.CIRCLE_LAYOUT_MM 中的圆点纸真实坐标。
    输出：最远两圆点之间的真实距离，单位 mm。
    实验作用：圆点法没有棋盘格那种固定相邻间距，所以用布局跨度和图像跨度建立毫米/像素比例。
    """

    # 本段读取当前启用的理论圆点布局。
    # 输出 layout 是 N×2 的真实纸面坐标。
    layout = np.asarray(config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT], dtype=float)

    # 本段计算所有两两距离并取最大值。
    # 这个最大跨度会和第一帧图像中的最大像素跨度对应。
    deltas = layout[:, None, :] - layout[None, :, :]
    return float(np.max(np.linalg.norm(deltas, axis=2)))


class CircleTracker:
    """
    圆点法的跨帧状态管理器。

    输入：每帧预处理后的灰度图。
    输出：circle_* 字段和当前帧圆心像素坐标。

    实验作用：
    圆点法需要同时解决两个问题：
    - 当前帧识别到了哪些圆点；
    - 这些圆点分别对应第一帧中的哪个圆点身份。

    因此它保存第一帧作为位移参考，又保存上一帧作为身份匹配参考。
    """

    def __init__(self) -> None:
        # 本段保存圆点法的跨帧状态。
        # reference_points 是位移零点；last_points/last_step 用于下一帧身份匹配；mm_per_pixel 用于单位换算。
        self.reference_points: np.ndarray | None = None
        self.last_points: np.ndarray | None = None
        self.last_step = np.zeros(2, dtype=np.float32)
        self.mm_per_pixel: float | None = None

    @staticmethod
    def _initial_order(points: np.ndarray) -> np.ndarray:
        """
        给第一帧圆点建立初始身份顺序。

        输入：第一帧完整识别出的圆心坐标。
        输出：按 y 后 x 排列的圆点坐标。
        实验作用：第一帧排序结果会成为后续所有帧的身份编号基础。
        后续帧不再重新排序，而是通过上一帧位置持续匹配，避免振动时身份跳变。
        """

        # 本段执行稳定的二维排序。
        # 这不是几何标定，只是给第一帧检测点一个可重复的编号顺序。
        order = np.lexsort((points[:, 0], points[:, 1]))
        return points[order]

    def _initialize(self, points: np.ndarray) -> None:
        """
        用第一帧完整圆点建立圆点法参考状态。

        输入：第一帧完整识别出的圆点坐标。
        输出：reference_points、last_points 和 mm_per_pixel。
        实验作用：第一帧是圆点法的位移零点。只有完整识别到全部预期圆点时才建立参考，
        否则后续很难判断缺失的是哪个点，身份编号会不可靠。
        """

        # 本段先建立第一帧身份编号。
        # 输出 ordered 后，后续第 i 个点就对应同一个物理圆点身份。
        ordered = self._initial_order(points)

        # 本段保存位移参考和下一帧匹配参考。
        # reference_points 不再变化；last_points 会随每帧匹配结果更新。
        self.reference_points = ordered.copy()
        self.last_points = ordered.copy()

        # 本段用第一帧图像跨度建立毫米/像素比例。
        # 输入是已编号圆点；输出 span_px，与理论布局跨度共同得到 mm_per_pixel。
        pairwise = ordered[:, None, :] - ordered[None, :, :]
        span_px = float(np.max(np.linalg.norm(pairwise, axis=2)))

        if span_px <= 0:
            raise ValueError("圆点中心重合，无法建立毫米比例。")

        self.mm_per_pixel = _layout_span_mm() / span_px

    def _match_to_previous(self, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        将当前帧候选圆点匹配到已有圆点身份。

        输入：当前帧检测到的圆心候选。
        输出：
        - reference：这些圆点在第一帧中的参考坐标；
        - current：这些圆点在当前帧中的坐标。

        实验作用：
        这里解决“当前帧每个圆点是谁”的问题。只有身份匹配稳定，
        后面的相对第一帧位移才有物理意义。
        """

        # 本段读取身份匹配所需状态。
        # reference_points 提供第一帧身份，last_points 提供上一帧位置预测基础。
        reference_points = self.reference_points
        last_points = self.last_points
        if reference_points is None or last_points is None:
            raise RuntimeError("Circle reference was not initialized.")

        # 本段预测当前帧圆点位置。
        # 输入是上一帧位置和上一帧平均移动量；输出 predicted，用来降低匀速运动中的匹配误差。
        predicted = last_points + self.last_step

        # 本段建立匹配代价矩阵。
        # distances[i, j] 越小，说明第 i 个旧身份越可能对应第 j 个新检测点。
        distances = np.linalg.norm(predicted[:, None, :] - detected[None, :, :], axis=2)

        # 本段用匈牙利算法选择全局最优一一匹配。
        # 这能避免多个旧点同时抢同一个新点。
        previous_indices, detected_indices = linear_sum_assignment(distances)

        # 本段剔除距离过大的匹配。
        # 输出 accepted_* 只保留合理范围内的身份对应关系，避免把背景误检硬配成圆点。
        accepted_previous: list[int] = []
        accepted_detected: list[int] = []
        for old_index, new_index in zip(previous_indices, detected_indices):
            if distances[old_index, new_index] <= config.CIRCLE_MAX_MATCH_DISTANCE_PX:
                accepted_previous.append(int(old_index))
                accepted_detected.append(int(new_index))

        # 本段保护圆点法最低有效点数。
        # 匹配点太少时，当前帧不具备可靠刚体估计条件。
        if len(accepted_previous) < config.MIN_VALID_CIRCLES:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # 本段取出已接受匹配的当前点和参考点。
        # current 是当前帧坐标；reference 是相同物理圆点在第一帧中的坐标。
        old_idx = np.asarray(accepted_previous, dtype=int)
        new_idx = np.asarray(accepted_detected, dtype=int)
        current = detected[new_idx]
        reference = reference_points[old_idx]

        # 本段更新跨帧状态。
        # 成功匹配的身份更新为当前坐标；未匹配身份保留旧位置，等待后续帧重新出现。
        old_positions = last_points[old_idx].copy()
        last_points[old_idx] = current
        self.last_points = last_points

        # 本段估计下一帧预测步长。
        # 中位移动量比平均值更不容易被个别坏点影响。
        self.last_step = np.median(current - old_positions, axis=0).astype(np.float32)
        return reference, current

    def process(self, gray: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        """
        完成一帧圆点法测量。

        输入：当前帧灰度图。
        输出：
        - circle_* 结果字典；
        - 当前帧圆心坐标，用于日志保存和调试图绘制。

        我这一帧看到棋盘格了吗? 调用camera.py的_find_corners()检测棋盘格角点 (calibrate_camera_intrinsics.py也有find_corners, 不过是给内参用的）
        ↓
        没看到：返回 checker_is_valid=False
        ↓
        看到了：
            如果这是第一次看到，就把这帧当参考零点
        ↓
        拿当前帧角点和参考帧角点，调用camera.py的calculate_mm_per_pixel()去算位移
        ↓
        把算出来的位移、角度、质量分整理成 checker_* 结果


        实验作用：
        这里把“当前帧圆点检测与身份匹配”转换成“相对第一帧的位移/转角/质量”。
        """

        # 本段先做当前帧圆点候选检测。
        # 输出 detected 只是候选圆心坐标，未必已经具备稳定身份。
        candidates = _find_circle_candidates(gray)
        detected = np.asarray(
            [[item["x"], item["y"]] for item in candidates],
            dtype=np.float32,
        )

        # 本段处理候选点明显不足的帧。
        # 输出仍保留检测到的点，方便调试图和日志显示“算法看见了什么”。
        if len(detected) < config.MIN_VALID_CIRCLES:
            return {
                "circle_is_valid": False,
                "circle_dx_mm": math.nan,
                "circle_dy_mm": math.nan,
                "circle_angle_deg": math.nan,
                "circle_scale": math.nan,
                "circle_residual_px": math.nan,
                "circle_quality": 0.0,
                "circle_valid_count": int(len(detected)),
                "circle_points_ordered": False,
            }, detected if len(detected) else None

        # 本段建立或读取圆点身份参考。
        # 第一帧必须完整识别，后续帧则用上一帧预测位置保持身份。
        if self.reference_points is None:
            if len(detected) != config.CIRCLE_EXPECTED_COUNT:
                return {
                    "circle_is_valid": False,
                    "circle_dx_mm": math.nan,
                    "circle_dy_mm": math.nan,
                    "circle_angle_deg": math.nan,
                    "circle_scale": math.nan,
                    "circle_residual_px": math.nan,
                    "circle_quality": 0.0,
                    "circle_valid_count": int(len(detected)),
                    "circle_points_ordered": False,
                }, detected
            self._initialize(detected)
            reference = self.reference_points
            current = self.last_points
        else:
            reference, current = self._match_to_previous(detected)

        # 本段读取比例尺和当前匹配点。
        # 如果这里缺失，说明初始化或匹配流程异常，应立即报错。
        mm_per_pixel = self.mm_per_pixel
        if reference is None or current is None or mm_per_pixel is None:
            raise RuntimeError("Circle reference was not initialized.")

        # 本段处理匹配后有效点不足的情况。
        # 这说明虽然检测到了候选点，但身份匹配不足以支撑刚体运动估计。
        if len(current) < config.MIN_VALID_CIRCLES:
            return {
                "circle_is_valid": False,
                "circle_dx_mm": math.nan,
                "circle_dy_mm": math.nan,
                "circle_angle_deg": math.nan,
                "circle_scale": math.nan,
                "circle_residual_px": math.nan,
                "circle_quality": 0.0,
                "circle_valid_count": int(len(current)),
                "circle_points_ordered": False,
            }, detected

        # 本段把已匹配圆点转换成整体运动结果。
        # 输入是同一批物理圆点的第一帧坐标和当前帧坐标；输出是毫米位移、转角、尺度和残差。
        motion = _estimate_rigid_motion(
            reference,
            current,
            float(mm_per_pixel),
            config.CIRCLE_RESIDUAL_WARNING_PX,
        )

        # 本段把匹配点数量纳入质量分。
        # 5 点勉强可算不应和 7 点完整识别得到同样置信度。
        point_fraction = len(current) / config.CIRCLE_EXPECTED_COUNT
        quality = float(motion["quality"]) * point_fraction

        # 本段整理当前帧圆点法日志。
        # circle_points_ordered=True 表示返回的圆心坐标已经按身份匹配后输出。
        result = {
            "circle_is_valid": bool(motion["is_valid"]),
            "circle_dx_mm": motion["dx_mm"],
            "circle_dy_mm": motion["dy_mm"],
            "circle_angle_deg": motion["angle_deg"],
            "circle_scale": motion["scale"],
            "circle_residual_px": motion["residual_px"],
            "circle_quality": quality,
            "circle_valid_count": int(len(current)),
            "circle_mm_per_pixel": float(mm_per_pixel),
            "circle_points_ordered": True,
        }
        return result, current


# =============================================================================
# 5. 单帧统一处理与调试画面
# =============================================================================

class VisionProcessor:
    """
    单帧视觉测量的统一入口。

    输入：FramePacket，包含一帧图像、帧号、时间戳和来源名称。
    输出：result 字典和 debug 图像。

    实验作用：
    - 把每帧图像转换成机器可读的测量记录；
    - 在同一个对象中保存第一帧参考点和上一帧圆点身份；
    - 同时保留像素坐标、二维位移和可选空间坐标，方便判断问题发生在哪一步。
    """

    def __init__(self) -> None:
        self.checker_tracker = CheckerboardTracker()
        self.circle_tracker = CircleTracker()

    @staticmethod
    def _points_to_full_image(
        points: np.ndarray | None,
        roi_origin: tuple[int, int],
    ) -> np.ndarray | None:
        """
        把 ROI 内识别坐标转回整幅图像坐标。

        输入：识别器输出的点坐标，坐标系可能是 ROI 左上角；
        输出：整张原图上的像素坐标，供日志、调试图和空间转换共用。
        """

        if points is None:
            return None
        return np.asarray(points, dtype=np.float64).reshape(-1, 2) + np.asarray(
            roi_origin,
            dtype=np.float64,
        )

    @staticmethod
    def _add_point_outputs(
        result: dict[str, Any],
        prefix: str,
        points: np.ndarray | None,
        roi_origin: tuple[int, int],
    ) -> None:
        """
        将识别点坐标写入当前帧结果。

        输入：某种算法识别出的像素点；
        输出：
        - {prefix}_points_px 或 {prefix}_corners_px：整幅图像中的原始像素坐标；
        - {prefix}_points_spatial / {prefix}_corners_spatial：可选空间坐标结果。

        这层输出不改变位移计算，只提供排错证据：先看像素坐标是否可信，再看空间坐标是否可信。
        """

        field = "corners" if prefix == "checker" else "points"
        points_full = VisionProcessor._points_to_full_image(points, roi_origin)
        if not config.SAVE_POINT_COORDINATES:
            return

        coordinate_key = f"{prefix}_{field}_px"
        spatial_key = f"{prefix}_{field}_spatial"

        if points_full is None:
            result[coordinate_key] = []
            result[spatial_key] = {
                "enabled": bool(config.SPATIAL_COORDINATES_ENABLED),
                "mode": config.SPATIAL_COORDINATE_MODE,
                "camera_m": None,
                "robot_m": None,
            }
            return

        result[coordinate_key] = points_full.tolist()
        try:
            result[spatial_key] = image_points_to_spatial(points_full)
        except Exception as exc:
            result[spatial_key] = {
                "enabled": bool(config.SPATIAL_COORDINATES_ENABLED),
                "mode": config.SPATIAL_COORDINATE_MODE,
                "camera_m": None,
                "robot_m": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def process_frame(self, packet: FramePacket) -> tuple[dict[str, Any], np.ndarray]:
        # 本段通过棋盘格和圆点的识别算法，把“原始图像帧”转换成“可识别的灰度图 + 图像质量记录”。
        #一帧 FramePacket
        #↓
        #预处理成灰度图，记录画质 preprocess_frame()
        #↓
        #创建 result 基础信息
        #↓
        #决定这一帧的分析时间 analysis_time_s
        #↓
        #按配置运行棋盘格法 CheckerboardTracker.process--estimate_rigid_motion()
        #↓
        #按配置运行圆点法 CircleTracker.process--estimate_rigid_motion()
        #↓
        #汇总这一帧是否有效
        #↓
        #画 debug 调试图
        #↓
        #返回 result, debug

        # 输入：FramePacket.frame，即 OpenCV 图像数组。
        # 输出：gray 进入圆点/棋盘格识别；image_metrics 写入日志，帮助判断某帧失败是否由曝光或模糊导致。
        gray, image_metrics, roi_origin = preprocess_frame(packet.frame)

        # 本段建立当前帧的基础日志记录。
        # 输入：FramePacket 的帧号、时间戳、来源名称，以及上一步得到的图像质量指标。
        # 输出：result 字典的公共字段；后续棋盘格法和圆点法只需要继续往里面补测量结果。
        result: dict[str, Any] = {
            "kind": "VISION",
            "host_ns": int(packet.host_ns),
            "frame_id": int(packet.frame_id),
            "camera_timestamp_raw": packet.camera_timestamp_raw,
            "source_name": packet.source_name,
            **image_metrics,
        }

        # 本段决定这帧进入离线分析时使用哪个时间轴。
        # 如果是离线的图片/视频，优先使用图片自己自带的时间戳。如果没有，就用电脑当前时间。（一般是有的）
        if (
            packet.source_name.startswith(("image_folder:", "video:"))
            and packet.camera_timestamp_raw is not None
        ):
            result["analysis_time_s"] = float(packet.camera_timestamp_raw)
        else:
            result["analysis_time_s"] = float(packet.host_ns) * 1e-9

        # 本段 预留 “算法看见的点”。
        # 这些点既会进入日志成为排错证据，也会画到 debug 图上给人肉眼检查。
        checker_corners: np.ndarray | None = None
        circle_centers: np.ndarray | None = None

        # 本段运行棋盘格测量链路。
        # 输入：灰度图和第一帧参考角点状态。
        # 输出：checker_* 字段，包括角点像素坐标、相对参考帧位移、转角、比例尺和质量分。
        if config.VISION_METHOD in {"checkerboard", "compare"}:
            checker_result, checker_corners = self.checker_tracker.process(gray)
            result.update(checker_result)
            self._add_point_outputs(result, "checker", checker_corners, roi_origin)

        # 本段运行圆点测量链路。
        # 输入：灰度图、第一帧参考圆点、上一帧圆点身份状态。
        # 输出：circle_* 字段，包括圆心像素坐标、身份匹配状态、相对参考帧位移、转角和质量分。
        if config.VISION_METHOD in {"circles", "compare"}:
            circle_result, circle_centers = self.circle_tracker.process(gray)
            result.update(circle_result)
            self._add_point_outputs(result, "circle", circle_centers, roi_origin)

        # 本段给整帧结果一个总有效性标记。只要棋盘格法或圆点法有一种有效，就认为这一帧总体有效。
        # 后续严肃分析仍会检查具体方法自己的有效性，而不是只依赖这个总标记。
        valid_flags = [
            bool(result.get("checker_is_valid", False)),
            bool(result.get("circle_is_valid", False)),
        ]
        result["is_valid"] = any(valid_flags)

        # 本段生成给人看的调试图。
        # 输入：原图、当前帧 result，以及两种算法识别到的点。
        # 输出：debug 图像，只用于人工质检，不作为后续计算输入。
        # 如果日志数值异常，先看这张图能快速判断是识别点错了、编号串了，还是图像质量本身有问题。
        debug = self._draw_debug(
            packet.frame,
            result,
            checker_corners,
            circle_centers,
            roi_origin,
        )
        return result, debug

    @staticmethod
    def _draw_debug(
        original: np.ndarray,
        result: dict[str, Any],
        checker_corners: np.ndarray | None,
        circle_centers: np.ndarray | None,
        roi_origin: tuple[int, int],
    ) -> np.ndarray:
        """
        生成当前帧的人工质检图。

        输入：
        - original：原始图像；
        - result：当前帧机器可读测量结果；
        - checker_corners / circle_centers：算法在 ROI 坐标系中识别到的点；
        - roi_origin：ROI 在原图中的偏移。

        输出：
        - 带角点、圆心编号、位移和画质指标的 debug 图。

        实验作用：
        debug 图不参与识别和后续分析，只用于人眼检查。若某帧数值异常，
        它能帮助判断问题来自漏点、串号、模糊、过曝，还是算法看见的点本身就不对。

        返回值是带标注的图像副本：
        - 绿色小点：棋盘格角点；
        - 红色圆圈和编号：圆点法识别到的圆心；
        - 左上角文字：位移、质量分、亮度和模糊指标。
        """

        # 本段准备可绘制的画布。
        # 输入可能是灰度或彩色图；输出统一为 BGR 彩色副本，避免标注修改原始 frame。
        if original.ndim == 2:
            canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            canvas = original.copy()

        # 本段把 ROI 坐标转换回整幅图像坐标。
        # 识别可能只在 ROI 内做，但人工质检图要画在原图上。
        offset = np.asarray(roi_origin, dtype=np.float32)

        # 本段绘制棋盘格法看到的角点。
        # 输出的绿色点用于检查角点是否落在真实棋盘格交点上。
        if checker_corners is not None:
            for point in checker_corners:
                x, y = np.rint(point + offset).astype(int)
                cv2.circle(canvas, (x, y), 3, (0, 180, 0), -1, cv2.LINE_AA)

        # 本段绘制圆点法看到的圆心和身份编号。
        # 编号用于观察跨帧身份是否串号；位置用于检查圆心是否偏离真实标记。
        if circle_centers is not None:
            for index, point in enumerate(circle_centers):
                x, y = np.rint(point + offset).astype(int)
                cv2.circle(canvas, (x, y), 6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 本段提供调试图文字读取的兜底函数。
        # 输入是 result 字段名；输出是可格式化的 float，避免缺字段导致整张调试图生成失败。
        def metric(name: str, default: float) -> float:
            value = result.get(name, default)
            return float(default if value is None else value)

        def short_number(name: str, digits: int = 3) -> str:
            value = metric(name, math.nan)
            if not math.isfinite(value):
                return "--"
            return f"{value:.{digits}f}"

        # 本段组织左上角质检文字。
        # 输出让使用者快速看到帧号、是否识别到完整棋盘格、角点数量、位移和图像质量。
        expected_checker_corners = (
            int(config.CHECKERBOARD_INNER_CORNERS[0])
            * int(config.CHECKERBOARD_INNER_CORNERS[1])
        )
        checker_count = int(result.get("checker_corner_count", 0))
        checker_valid = bool(result.get("checker_is_valid", False))
        if checker_valid:
            status = "已识别棋盘格"
        else:
            status = "未识别完整棋盘格"

        lines = [
            f"帧号 {result['frame_id']}｜{status}",
            f"角点 {checker_count}/{expected_checker_corners}｜质量 {short_number('checker_quality', 2)}",
            f"位移 X {short_number('checker_dx_mm')} mm｜Y {short_number('checker_dy_mm')} mm",
            f"亮度 {short_number('mean_brightness', 1)}｜清晰 {short_number('blur_variance', 1)}",
        ]

        # 本段生成更直观的中文提示。
        # 它只影响预览和 debug 图，不改变任何识别结果。
        warning_lines: list[str] = []
        if result["blur_variance"] < config.BLUR_WARNING_THRESHOLD:
            warning_lines.append("提示：可能失焦/运动模糊")
        if result["bright_fraction"] > 0.20:
            warning_lines.append("提示：画面过亮/过曝")
        if result["dark_fraction"] > 0.20:
            warning_lines.append("提示：画面过暗")
        if not checker_valid:
            warning_lines.append("提示：请让 12×9 方格完整入画")

        # 本段真正绘制中文面板。
        # 使用 Pillow 字体绘制，避免 OpenCV putText 无法显示中文。
        _draw_chinese_panel(canvas, lines, warning_lines)
        return canvas


# =============================================================================
# 6. 复合测量纸与合成测试图
# =============================================================================

def generate_marker_sheet(output_path: Path | None = None) -> Path:
    """
    生成可打印的复合测量纸 SVG。

    输入：可选输出路径；未传入时使用 config.MARKER_SHEET_PATH。
    输出：一份按毫米定义尺寸的 SVG 文件。

    实验作用：
    测量纸同时包含棋盘格和不对称圆点阵列。棋盘格用于 checkerboard 方法，
    圆点阵列用于 circles 方法；两种方法可以在预实验中互相对照。

    打印对话框必须选择 100%/实际大小；打印后应再用游标卡尺测量方格边长确认比例。

    输出内容：
    - 左侧：棋盘格，用于 checkerboard 方法；
    - 右侧：不对称圆点阵列，用于 circles 方法；
    - 外框：浅灰裁剪参考线，尽量不干扰黑色标记检测。
    """

    # 本段确定 SVG 输出位置。
    # 输出目录会自动创建，避免首次运行时因 outputs 不存在而失败。
    output = Path(output_path or config.MARKER_SHEET_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 本段读取测量纸几何参数。
    # 输入来自 config.py；输出是后续绘制棋盘格、圆点和外框所需的毫米坐标。
    width = float(config.MARKER_SHEET_WIDTH_MM)
    height = float(config.MARKER_SHEET_HEIGHT_MM)
    margin = float(config.MARKER_MARGIN_MM)
    columns, rows = config.CHECKERBOARD_INNER_CORNERS
    square = float(config.CHECKER_SQUARE_MM)
    board_columns = columns + 1
    board_rows = rows + 1

    # 本段计算棋盘格区域尺寸。
    # OpenCV 配置的是内角点数量，真实黑白方格数量需要在行列方向各加 1。
    checker_width = board_columns * square
    checker_height = board_rows * square

    # 本段安排棋盘格和圆点区域在同一张纸上的位置。
    # 输出 board_* 和 circle_* 坐标，确保两种图案分区清楚，减少互相干扰。
    board_x = margin
    board_y = (height - checker_height) / 2.0

    layout = np.asarray(
        config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT],
        dtype=float,
    )
    circle_x = width - margin - float(np.max(layout[:, 0]))
    circle_y = (height - float(np.max(layout[:, 1]))) / 2.0

    # 本段防止生成不可用测量纸。
    # 如果两种图案会重叠，直接报错比生成一张误导性的 SVG 更安全。
    if board_x + checker_width + margin > circle_x:
        raise ValueError("当前测量纸太窄，棋盘格与圆点区域会重叠。")

    # 本段开始组织 SVG 文本。
    # 输出 lines 是最终文件的 XML 行列表；使用 SVG 是为了避免 PNG 打印 DPI 带来的缩放问题。
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" '
            f'height="{height}mm" viewBox="0 0 {width} {height}">'
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    # 本段绘制棋盘格黑色方块。
    # 白色格子由背景提供，黑白交界处会成为 checkerboard 方法识别的内角点。
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                x = board_x + column * square
                y = board_y + row * square
                lines.append(
                    f'<rect x="{x:.4f}" y="{y:.4f}" '
                    f'width="{square:.4f}" height="{square:.4f}" fill="black"/>'
                )

    # 本段绘制不对称圆点阵列。
    # 第 0 个点稍大，用作人工观察方向锚点；算法当前仍主要依赖布局和跨帧匹配。
    for index, (x_mm, y_mm) in enumerate(layout):
        diameter = (
            config.CIRCLE_ANCHOR_DIAMETER_MM
            if index == 0
            else config.CIRCLE_DIAMETER_MM
        )
        lines.append(
            f'<circle cx="{circle_x + x_mm:.4f}" cy="{circle_y + y_mm:.4f}" '
            f'r="{diameter / 2.0:.4f}" fill="black"/>'
        )

    # 本段绘制浅灰裁剪外框。
    # 颜色很浅，目的是给人裁剪参考，同时尽量不被圆点轮廓检测当成黑色目标。
    lines.append(
        f'<rect x="0.2" y="0.2" width="{width - 0.4}" height="{height - 0.4}" '
        'fill="none" stroke="#BBBBBB" stroke-width="0.2"/>'
    )
    lines.append("</svg>")

    # 本段写出 SVG 文件。
    # 输出文件可直接打开或打印，后续 vision_test 会提示路径。
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def create_synthetic_demo_sequence(
    output_folder: Path,
    frame_count: int = 90,
    fps: float = 60.0,
) -> Path:
    """
    生成仅用于程序烟测的合成图片序列。

    输入：输出文件夹、帧数和合成帧率。
    输出：一组按 frame_0000.png 命名的 PNG 图片。

    实验作用：
    它只验证“读图 -> 识别 -> 写日志 -> 分析”链路能跑通，不代表真实相机精度、
    真实噪声、真实标定或真实机械臂振动。

    它会给复合图案加入小幅正弦平移与转动，适合在没有设备时验证识别、日志和分析函数。

    注意：
    - 这不是相机标定数据；
    - 这不是实验数据；
    - 它只是为了验证程序的输入/处理/输出链路能跑通。
    """

    # 本段准备合成序列输出位置。
    # 输出文件夹存在时会复用，生成的图片按帧号命名，便于 ImageFolderSource 自然排序。
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 本段创建一张包含棋盘格和圆点的基础图。
    # 后续每一帧都从这张基础图变换而来，模拟测量纸整体运动。
    image = np.full((520, 960, 3), 255, dtype=np.uint8)

    # 本段定义合成棋盘格的像素尺寸和位置。
    # 它只服务于代码链路测试，不用于真实实验标定。
    square_px = 46
    columns, rows = config.CHECKERBOARD_INNER_CORNERS
    board_columns, board_rows = columns + 1, rows + 1
    board_x, board_y = 70, 110

    # 本段绘制合成棋盘格。
    # 输出的黑白交界让 CheckerboardTracker 能检测角点。
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                p1 = (board_x + column * square_px, board_y + row * square_px)
                p2 = (p1[0] + square_px, p1[1] + square_px)
                cv2.rectangle(image, p1, p2, (0, 0, 0), -1)

    # 本段绘制合成圆点阵列。
    # 输入是 config.CIRCLE_LAYOUT_MM；输出是可供 CircleTracker 检测的黑色圆点。
    layout = np.asarray(
        config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT],
        dtype=float,
    )
    circle_origin = np.asarray([600.0, 135.0])
    circle_scale_px_per_mm = 9.0
    for index, point in enumerate(layout):
        center = np.rint(circle_origin + point * circle_scale_px_per_mm).astype(int)
        radius = 19 if index == 0 else 13
        cv2.circle(image, tuple(center), radius, (0, 0, 0), -1, cv2.LINE_AA)

    # 本段生成多帧模拟运动。
    # 输入是基础图；输出是带正弦平移/转角的图片序列，用于测试位移曲线是否能被恢复。
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    for frame_id in range(frame_count):
        # 本段计算当前合成帧的运动状态。
        # 三个正弦项分别模拟 x 位移、y 位移和平面内轻微转动。
        t = frame_id / fps
        dx_px = 3.2 * math.sin(2.0 * math.pi * 8.0 * t)
        dy_px = 2.2 * math.sin(2.0 * math.pi * 13.0 * t + 0.4)
        angle_deg = 0.12 * math.sin(2.0 * math.pi * 5.0 * t)

        # 本段把运动状态应用到基础图。
        # 输出 moved 是当前帧图片，保存后即可被离线图片输入源读取。
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        matrix[0, 2] += dx_px
        matrix[1, 2] += dy_px
        moved = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        # 本段保存当前合成帧。
        # 文件名保持固定宽度，确保自然排序和普通字符串排序都能得到正确帧顺序。
        cv2.imwrite(str(output_folder / f"frame_{frame_id:04d}.png"), moved)

    return output_folder


# =============================================================================
# 7. 视觉测试模式与完整实验子进程
# =============================================================================

def _resize_for_preview(image: np.ndarray) -> np.ndarray:
    """
    生成适合屏幕预览的图像副本。

    输入：debug 图或其他 OpenCV 图像。
    输出：宽度不超过 config.PREVIEW_MAX_WIDTH 的显示副本。

    实验作用：
    只影响人眼预览窗口，不改变算法输入、不改变日志、不改变保存的原始调试图。
    """

    # 本段判断是否需要缩小。
    # 图像本来就适合屏幕显示时，直接返回，避免无意义重采样。
    if image.shape[1] <= config.PREVIEW_MAX_WIDTH:
        return image

    # 本段按比例缩放显示图。
    # 输出只用于 cv2.imshow，保持宽高比例不变，避免预览图变形误导判断。
    scale = config.PREVIEW_MAX_WIDTH / image.shape[1]
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _json_line(record: dict[str, Any]) -> str:
    """
    把一条视觉记录转换成一行 JSON 文本。

    输入：META 或 VISION 字典。
    输出：不带换行符的 JSON 字符串。

    实验作用：
    vision_results.txt 和正式 run_log.txt 都采用“一行一条记录”的形式。
    这样文件既能被记事本查看，也能被 analyze.py 稳定逐行恢复。
    """

    # 本段执行 JSON 序列化。
    # ensure_ascii=False 让中文保持可读；allow_nan=True 保留无效测量的 NaN 状态供后续分析跳过。
    import json
    return json.dumps(record, ensure_ascii=False, allow_nan=True)


def run_vision_test() -> Path:
    """
    离线单进程运行视觉模块，保存逐帧结果和可选调试图。
    
    已有图片/视频
    ↓
    程序按顺序逐帧读取
    ↓
    每帧识别圆点/棋盘格
    ↓
    输出位移、质量指标、调试图
    ↓
    后续再分析振动

    该函数不会导入 robot.py，也不会检查 UR IP，更不会创建机器人连接。
    """

    # 视觉测试开始时顺手生成一张可打印测量纸。
    # 这只是生成 SVG 文件，不会读取相机，也不会影响输入图片。
    if config.GENERATE_MARKER_SHEET_ON_VISION_TEST:
        marker_path = generate_marker_sheet()
        print(f"[视觉] 已生成可打印测量纸：{marker_path}")

    # 每次运行都放进新的时间戳文件夹，避免覆盖上一次测试结果。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_ROOT / f"vision_test_{timestamp}"
    debug_dir = run_dir / "debug_images"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.SAVE_DEBUG_IMAGE:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # 本段建立“逐帧视觉测量”的工作状态。
    # 输入来自 source 产生的一帧帧图像；输出是每帧像素坐标、二维位移、可选空间坐标和调试图。
    # VisionProcessor 会在第一次成功识别标记点时建立参考状态，因此一次测试只创建一个实例。
    processor = VisionProcessor()

    # vision_results.txt 是离线分析的主要输入。
    # 第一行写本次测试说明，后续每行写一帧测量结果，供 analyze.py 继续做振动分析。
    result_path = run_dir / "vision_results.txt"

    # 这些计数器用于结束时给出本组数据的可用性摘要，不参与视觉计算。
    processed_count = 0
    valid_count = 0
    saved_debug_count = 0

    try:
        # 本段同时打开“图像来源”和“结果文件”。
        # source 负责按顺序产出 FramePacket；file 负责接收每帧处理后的 JSON 记录。
        with open_image_source() as source, result_path.open("w", encoding="utf-8") as file:
            # META 是本次视觉测试的运行说明，只写一次。
            # 它让后续分析知道这些逐帧结果来自什么图像来源、使用了什么视觉方法。
            meta = {
                "kind": "META",
                "host_ns": time.perf_counter_ns(),
                "mode": "vision_test",
                "vision_source": config.VISION_SOURCE,
                "vision_method": config.VISION_METHOD,
                "image_folder_fps": config.IMAGE_FOLDER_FPS,
                "expected_vision_fps": config.EXPECTED_VISION_FPS,
                "save_point_coordinates": config.SAVE_POINT_COORDINATES,
                "spatial_coordinates_enabled": config.SPATIAL_COORDINATES_ENABLED,
                "spatial_coordinate_mode": config.SPATIAL_COORDINATE_MODE,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            file.write(_json_line(meta) + "\n")

            # 本循环是离线视觉测试的主流水线。
            # 输入：一帧 FramePacket；输出：一行机器可读测量结果，以及可选调试图。
            # 图像来源每吐出一帧，我就把这一帧交给视觉处理器。处理器返回两样东西：
            # 一个是机器可读的测量结果 result，一个是给人看的标注图 debug。然后把 result 写成一行 JSON。
            # 本函数只把原始图像序列转换成逐帧视觉测量结果，不做频谱和恢复时间分析。
            for packet in source:
                result, debug = processor.process_frame(packet)

                # 当前帧 result 写入一行 JSON。
                # 后续 analyze.py 会读取这些行，继续完成去趋势、频谱和指标计算。
                file.write(_json_line(result) + "\n")
                processed_count += 1
                valid_count += int(bool(result["is_valid"]))

                # 本段控制调试图保存量。
                # 调试图用于人工检查识别点、编号和图像质量；它不是后续算法输入。
                # 为避免批量离线数据产生过多图片，只按相机原始帧号间隔和最大张数保存。
                # 对在线相机来说，若 frame_id 大幅跳跃，说明当前程序处理链路没有跟上相机实际帧率。
                should_save = (
                    config.SAVE_DEBUG_IMAGE
                    and packet.frame_id % config.DEBUG_IMAGE_EVERY_N_FRAMES == 0
                    and saved_debug_count < config.MAX_DEBUG_IMAGES
                )
                if should_save:
                    cv2.imwrite(
                        str(debug_dir / f"frame_{packet.frame_id:08d}.jpg"),
                        debug,
                    )
                    saved_debug_count += 1

                # 本段只负责人眼预览。
                # 预览窗口不改变已经写入的测量结果；按 Esc 或 q 只是提前结束本次离线测试。
                if config.SHOW_PREVIEW:
                    cv2.imshow("UR10 vibration vision test", _resize_for_preview(debug))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("[视觉] 操作者按键结束预览。")
                        break
    finally:
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()

    if processed_count == 0:
        raise RuntimeError("视觉测试没有成功读取任何一帧。")

    print(
        f"[视觉] 完成 {processed_count} 帧，其中至少一种方法有效 {valid_count} 帧；"
        f"结果：{result_path}"
    )
    return run_dir


def _put_record(record_queue: Any, record: dict[str, Any], stop_event: Any) -> bool:
    """
    将相机测量结果送入正式实验记录队列。

    输入：当前帧 VISION 记录、主程序提供的 record_queue 和 stop_event。
    输出：
    - True：记录已交给 writer 进程；
    - False：队列满，已经请求整套实验停止。

    实验作用：
    正式预实验中相机进程不直接写文件，而是把结果交给 main.py 的 writer 进程统一落盘。
    队列满通常意味着写盘或主流程跟不上，继续采集会破坏时间对齐，因此主动停止。
    """

    try:
        # 本段尝试把当前帧结果交给 writer。
        # timeout 防止队列满时相机进程永久卡住。
        record_queue.put(record, timeout=1.0)
        return True
    except Full:
        # 本段处理记录队列拥塞。
        # 输出 stop_event，让机器人和主程序也尽快进入安全收尾。
        stop_event.set()
        return False


def camera_worker(
    record_queue: Any,
    error_queue: Any,
    start_event: Any,
    stop_event: Any,
    camera_ready: Any,
) -> None:
    """
    正式预实验中的相机子进程入口。

    输入：
    - record_queue：把 VISION 记录送给 writer 进程；
    - error_queue：把相机异常报告给主程序；
    - start_event：主程序确认相机和机器人都 ready 后发出的统一开始信号；
    - stop_event：任一进程要求停止时亮起；
    - camera_ready：相机完成真实取帧和第一帧处理后置位。

    输出：
    - 连续 VISION 记录进入 record_queue；
    - 异常文本进入 error_queue；
    - 必要时设置 stop_event。

    实验作用：
    这里服务于“相机 + 机器人 + writer”三进程正式预实验。
    它复用 VisionProcessor 的逐帧算法，但不自己写文件；所有正式日志由 main.py 的 writer 统一写入。
    """

    try:
        # 本段建立正式实验相机测量状态。
        # 输出 processor 会保存第一帧参考点和圆点跨帧身份，保证完整实验期间位移序列连续。
        processor = VisionProcessor()
        with open_image_source() as source:
            iterator = iter(source)

            # 本段完成相机 ready 前的真实检查。
            # 只有成功取到并处理第一帧，才说明“图像来源 + 视觉算法”都能工作，而不是只创建了对象。
            first_packet = next(iterator)
            first_result, _ = processor.process_frame(first_packet)
            first_result["analysis_time_s"] = first_result["host_ns"] * 1e-9
            camera_ready.set()

            if config.VISION_SOURCE == "hik_camera":
                # 本段处理真实相机等待操作者确认的时间。
                # 输入是连续相机流；输出只保留最新一帧结果。
                # 这样 start_event 到来时，正式日志不会从几秒前的相机缓冲旧帧开始。
                while not start_event.is_set():
                    if stop_event.is_set():
                        return
                    warmup_packet = next(iterator)
                    first_result, _ = processor.process_frame(warmup_packet)
                    first_result["analysis_time_s"] = first_result["host_ns"] * 1e-9
            else:
                # 本段处理离线图片/视频作为实验输入的等待时间。
                # 离线来源不能在等待确认时提前高速消耗帧，所以这里只等待 start_event，不继续取下一帧。
                while not start_event.is_set():
                    if stop_event.wait(0.05):
                        return

            # 本段写入正式开始后的第一条视觉记录。
            # 对真实相机来说，这是等待期间保留的最新帧；对离线来源来说，是第一帧。
            if not _put_record(record_queue, first_result, stop_event):
                raise RuntimeError("记录队列已满，相机结果无法写入。")

            # 本循环是正式预实验相机记录主流程。
            # 输入是 start_event 后的连续帧；输出是逐帧 VISION 记录进入 record_queue。
            # 循环直到主程序要求停止、其他进程出错，或图像来源提前耗尽。
            for packet in iterator:
                if stop_event.is_set():
                    break

                # 本段处理当前实验帧。
                # 输出 result 是正式日志数据；debug 只在 SHOW_PREVIEW=True 时用于人眼预览。
                result, debug = processor.process_frame(packet)

                # 本段统一正式实验时间轴。
                # 正式 run_log 中 VISION、EVENT、ROBOT 必须共用电脑时基，不能使用离线视频自己的相对零点。
                result["analysis_time_s"] = result["host_ns"] * 1e-9

                if not _put_record(record_queue, result, stop_event):
                    raise RuntimeError("记录队列已满，相机结果无法写入。")

                # 本段处理正式实验预览。
                # 预览不是算法输入；若操作者按 q/Esc，视为请求整套实验停止。
                if config.SHOW_PREVIEW:
                    cv2.imshow("UR10 vibration experiment", _resize_for_preview(debug))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        stop_event.set()
                        break

            # 本段处理图像来源提前结束。
            # 正式实验应由主程序 stop_event 收尾；若相机/离线序列自己先结束，说明实验数据覆盖时间不足。
            if not stop_event.is_set():
                raise RuntimeError(
                    "图像来源在实验停止信号到来前已经结束。"
                    "正式实验应使用连续相机流，离线文件需保证长度覆盖全部运动和后记录。"
                )
    except StopIteration:
        # 本段处理连第一帧都没有的输入源。
        # 输出错误给主程序，并请求其他进程停止。
        error_queue.put("相机来源没有返回任何一帧。")
        stop_event.set()
    except Exception as exc:
        # 本段把相机进程异常汇报给主程序。
        # 主程序会据此停止机器人和 writer，避免相机失败后实验仍继续运动或记录。
        error_queue.put(f"相机进程异常：{type(exc).__name__}: {exc}")
        stop_event.set()
    finally:
        # 本段释放 OpenCV 预览窗口资源。
        # 无论正常结束还是异常退出，打开过窗口都应关闭。
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()
