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
from scipy.optimize import linear_sum_assignment

import config


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


class ImageFolderSource:
    """
    按顺序读取一个文件夹中的静态图片，不需要相机 SDK。

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
        # folder 是图片所在目录。
        self.folder = Path(folder)

        # fps 用来把“第几张图片”换算成“第几秒”。
        self.fps = float(fps)

        # __enter__ 中会扫描文件夹，把实际图片路径放进这里。
        self.paths: list[Path] = []

    def __enter__(self) -> "ImageFolderSource":
        # 进入 with 时先确认文件夹存在。
        if not self.folder.exists():
            raise FileNotFoundError(
                f"图片文件夹不存在：{self.folder}\n"
                "请创建该文件夹并放入按时间排序的图片，或修改 config.py 的 IMAGE_FOLDER。"
            )

        # 扫描所有支持扩展名的图片，并按自然顺序排序。
        self.paths = sorted(
            (
                path
                for path in self.folder.iterdir()
                if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
            ),
            key=_natural_sort_key,
        )

        # 没有图片时直接报错，避免后面 for 循环静默什么都不做。
        if not self.paths:
            raise FileNotFoundError(
                f"图片文件夹中没有支持的图像：{self.folder}\n"
                f"支持的扩展名为：{', '.join(config.IMAGE_EXTENSIONS)}。"
            )

        print(f"[视觉] 已找到 {len(self.paths)} 张图片：{self.folder}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # 静态图片没有需要释放的设备句柄，这个方法只是让三种来源拥有统一用法。
        return None

    def __iter__(self) -> Iterator[FramePacket]:
        # enumerate 同时给出图片序号 index 和路径 path。
        for index, path in enumerate(self.paths):
            # cv2.imread 返回 BGR 图像；读取失败时返回 None。
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)

            # host_ns 记录“电脑读到这一帧”的时间，方便和其他日志对齐。
            host_ns = time.perf_counter_ns()

            if frame is None:
                print(f"[视觉警告] OpenCV 无法读取，已跳过：{path.name}")
                continue

            # 离线图片没有相机硬件时间，使用帧号和设定帧率构造相对秒数。
            camera_time_s = index / self.fps
            yield FramePacket(
                frame,
                index,
                host_ns,
                camera_time_s,
                f"image_folder:{path.name}",
            )


class VideoSource:
    """
    用 OpenCV 解码普通视频；后续识别算法并不知道它与真实相机有什么区别。

    它和 ImageFolderSource 的共同点：
    - 都会产出 FramePacket；
    - 都不会连接海康 SDK；
    - 都适合离线调试。

    它和 ImageFolderSource 的区别：
    - 图片文件夹的时间来自 frame_id / IMAGE_FOLDER_FPS；
    - 视频优先使用视频文件内部的时间戳 CAP_PROP_POS_MSEC。
    """

    def __init__(self, video_path: Path) -> None:
        # video_path 是要打开的视频文件。
        self.video_path = Path(video_path)

        # capture 是 OpenCV 的视频读取对象；进入 with 后才真正创建。
        self.capture: cv2.VideoCapture | None = None

        # fps 会在打开视频后读取；若视频没写帧率，就退回 30。
        self.fps = 0.0

    def __enter__(self) -> "VideoSource":
        # 先检查路径，避免 OpenCV 给出不清楚的打开失败。
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"视频不存在：{self.video_path}\n"
                "请修改 config.py 的 VIDEO_PATH，或切换回 image_folder。"
            )

        # 创建 OpenCV 视频读取器。
        self.capture = cv2.VideoCapture(str(self.video_path))

        # isOpened() 为 False 说明文件不存在、格式不支持或解码器不可用。
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频：{self.video_path}")

        # 读取视频标称帧率；有些视频容器可能返回 0 或 NaN。
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
            print("[视觉警告] 视频未提供有效帧率，暂按 30 fps 记录相对时间。")

        print(f"[视觉] 已打开视频：{self.video_path}，标称帧率 {self.fps:.3f} fps")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # 退出 with 时释放视频文件句柄。
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def __iter__(self) -> Iterator[FramePacket]:
        # 如果用户没有放进 with 里使用，capture 还没创建，应立即报错。
        if self.capture is None:
            raise RuntimeError("VideoSource 必须放在 with 语句中使用。")

        frame_id = 0
        while True:
            # read() 读取下一帧；ok=False 表示读完或解码失败。
            ok, frame = self.capture.read()
            host_ns = time.perf_counter_ns()
            if not ok:
                break

            # 视频容器里的当前位置通常以毫秒表示。
            position_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))

            # 若当前位置无效，就用 frame_id/fps 估算相对秒数。
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
    海康 MVS Python 接口适配器。

    MVS 不同版本的包装类字段可能略有区别，因此所有厂商调用都集中在这里，算法部分无需跟着改。
    """

    def __init__(self) -> None:
        # sdk 是 MvCameraControl_class 模块本身。
        self.sdk: Any = None

        # camera 是 SDK 创建出来的相机句柄对象。
        self.camera: Any = None

        # device_info 保存枚举到并最终选中的相机信息。
        self.device_info: Any = None

        # payload_size 是每帧图像数据最大字节数，用来分配接收缓冲区。
        self.payload_size = 0

        # data_buffer 和 frame_info 是 SDK 取帧时反复复用的底层结构。
        self.data_buffer: Any = None
        self.frame_info: Any = None

    @staticmethod
    def _decode_c_string(value: Any) -> str:
        """把 SDK 的定长 C 字符数组安全转成普通字符串。"""

        try:
            raw = bytes(value)
        except TypeError:
            raw = bytearray(value)
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")

    def _load_sdk(self) -> Any:
        # 只有真正选择 hik_camera 才执行这里，所以没装 MVS 不会影响离线模式。
        # 如果 config.py 写了 MVS 的 MvImport 路径，就临时加入 sys.path。
        # 只有真正选择 hik_camera 才执行这里，所以没装 MVS 不会影响离线模式。
        if config.HIK_MVS_IMPORT_PATH:
            import_path = str(Path(config.HIK_MVS_IMPORT_PATH).expanduser().resolve())
            if import_path not in sys.path:
                sys.path.insert(0, import_path)

        # MVS Python 示例通常提供 MvCameraControl_class.py。
        # import_module 成功后，后续才可以枚举设备和取帧。
        try:
            return importlib.import_module("MvCameraControl_class")
        except ImportError as exc:
            raise RuntimeError(
                "当前选择了 hik_camera，但没有找到 MvCameraControl_class。\n"
                "请先安装海康 MVS，并在 config.py 填写 HIK_MVS_IMPORT_PATH；"
                "当前无相机时可改用 image_folder 或 video。"
            ) from exc

    def _device_serial(self, device_info: Any) -> str:
        """同时兼容 GigE 与 USB3 设备结构，尽量读取序列号用于防止连错相机。"""

        transport = int(device_info.nTLayerType)
        gige_type = int(getattr(self.sdk, "MV_GIGE_DEVICE", -1))
        usb_type = int(getattr(self.sdk, "MV_USB_DEVICE", -2))

        if transport == gige_type:
            return self._decode_c_string(device_info.SpecialInfo.stGigEInfo.chSerialNumber)
        if transport == usb_type:
            return self._decode_c_string(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        return ""

    def _select_device(self, device_list: Any) -> Any:
        """按序列号选择相机；未填序列号时明确使用枚举到的第一台。"""

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
        """MVS 通常用 0 表示成功；把十进制错误码转为更常见的十六进制便于查手册。"""

        if int(ret) != 0:
            raise RuntimeError(f"{action}失败，MVS 返回码 0x{int(ret) & 0xFFFFFFFF:08X}")

    def _try_set_float(self, node_name: str, value: float | None) -> None:
        """曝光或增益节点不是每台相机都同名，设置失败时给出警告而不是静默忽略。"""

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
            # GigE 使用合适包长可减少丢帧；USB 相机通常不会进入这个分支。
            if int(self.device_info.nTLayerType) == int(self.sdk.MV_GIGE_DEVICE):
                packet_size = int(self.camera.MV_CC_GetOptimalPacketSize())
                if packet_size > 0:
                    self.camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

            # 本实验采用连续自由运行，不等待外部触发信号。
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
            # 打开过程中任一步失败都要主动释放句柄，否则下一次运行可能显示设备正被占用。
            self.close()
            raise

        print("[视觉] 海康相机已打开并开始连续取流。")
        return self

    def _convert_raw_frame(self, frame_info: Any) -> np.ndarray:
        """把 MVS 缓冲区转换成 OpenCV 的灰度或 BGR 图像。"""

        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        pixel_type = int(frame_info.enPixelType)
        raw = np.ctypeslib.as_array(self.data_buffer)[: int(frame_info.nFrameLen)]

        mono8 = getattr(self.sdk, "PixelType_Gvsp_Mono8", None)
        bgr8 = getattr(self.sdk, "PixelType_Gvsp_BGR8_Packed", None)
        rgb8 = getattr(self.sdk, "PixelType_Gvsp_RGB8_Packed", None)

        if mono8 is not None and pixel_type == int(mono8):
            return raw.reshape(height, width).copy()

        if bgr8 is not None and pixel_type == int(bgr8):
            return raw.reshape(height, width, 3).copy()

        if rgb8 is not None and pixel_type == int(rgb8):
            rgb = raw.reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Bayer 具体排列必须与相机 PixelFormat 一致；这里只处理 SDK 中能明确识别的常见类型。
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
        """按“停止取流→关闭设备→销毁句柄”顺序释放资源，并容忍只打开了一半的情况。"""

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
    根据 config.py 只创建一种图像来源，未选择的硬件库不会被导入。

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
    把原图变成识别用灰度图，并记录亮度、过曝比例和清晰度。

    返回的 ROI 原点用于把识别坐标画回原图；运动计算本身始终在同一 ROI 坐标系内完成。
    """

    if frame is None or frame.size == 0:
        raise ValueError("收到空图像，无法预处理。")

    # working 是后续真正拿去识别的图像副本/视图。
    # 原始 frame 仍保留，用来把调试点画回完整图像。
    working = frame

    # 如果做过相机标定，就可以在这里消除镜头畸变。
    # 没做标定时 CAMERA_MATRIX 为 None，这一步会被跳过。
    if config.CAMERA_MATRIX is not None:
        camera_matrix = np.asarray(config.CAMERA_MATRIX, dtype=np.float64)
        distortion = np.asarray(config.DISTORTION_COEFFICIENTS, dtype=np.float64)
        working = cv2.undistort(working, camera_matrix, distortion)

    # ROI 是 Region Of Interest，意思是“只看图像中的某一小块”。
    # 这样既能加快识别，也能避免背景中的其他圆形/棋盘格干扰。
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

    # OpenCV 的很多识别函数更喜欢灰度图。
    # 如果输入已经是灰度，就直接复制；如果是彩色 BGR，就转成灰度。
    if working.ndim == 2:
        gray = working.copy()
    else:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    # 轻微模糊可以压低传感器噪声，让阈值分割更稳定。
    # 但模糊太强会吃掉圆点/角点边缘，所以核大小由 config.py 控制。
    if config.GAUSSIAN_BLUR_KERNEL > 1:
        gray = cv2.GaussianBlur(
            gray,
            (config.GAUSSIAN_BLUR_KERNEL, config.GAUSSIAN_BLUR_KERNEL),
            0,
        )

    # 这些 metrics 不直接决定结果，而是帮助你判断“为什么某帧识别失败”。
    # 例如 blur_variance 很低通常说明图像糊了，bright_fraction 很高通常说明过曝。
    metrics = {
        "mean_brightness": float(np.mean(gray)),
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray <= config.DARK_PIXEL_THRESHOLD)),
        "bright_fraction": float(np.mean(gray >= config.BRIGHT_PIXEL_THRESHOLD)),
    }
    return gray, metrics, roi_origin


def _estimate_rigid_motion(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    mm_per_pixel: float,
    residual_warning_px: float,
) -> dict[str, float | bool]:
    """
    用多点最小二乘估计平移、平面转角和尺度变化。

    位移使用对应点质心差，避免“绕纸张中心旋转”被错误解释成额外平移。

    这里的“刚体运动”是指：测量纸整体移动、旋转、轻微缩放，但纸上点之间的相对布局基本不变。
    如果某些点被误识别，RANSAC 会尽量剔除它们；如果有效点太少，就返回无效结果。
    """

    # 不管外面传来的数组形状是什么，先强制整理成 N 行 2 列的浮点坐标。
    # 每一行表示一个点：(x_pixel, y_pixel)。
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)

    # 至少 3 个点才能稳定估计二维平移、旋转和尺度。
    # 两组点数量必须相同，因为第 i 个参考点要对应第 i 个当前点。
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

    # estimateAffinePartial2D 会拟合一个“相似变换”：
    # current ~= reference 经过平移 + 旋转 + 等比例缩放。
    # RANSAC 参数让它能容忍少量错误点，而不是被一个坏点拖偏。
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

    # 用拟合出来的 matrix 把参考点变换到当前帧，再和真实当前点比较。
    # 差值越小，说明本帧识别越可靠。
    predicted = cv2.transform(ref.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - cur, axis=1)

    # inlier 是 RANSAC 认为“可信”的点。若 OpenCV 没有返回这个信息，就退回使用全部点。
    if inliers is None:
        inlier_mask = np.ones(len(ref), dtype=bool)
    else:
        inlier_mask = inliers.ravel().astype(bool)
    if not np.any(inlier_mask):
        inlier_mask[:] = True

    # 平移量用可信点的平均位移表示；再乘 mm_per_pixel 从像素换成毫米。
    centroid_shift = np.mean(cur[inlier_mask] - ref[inlier_mask], axis=0)
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    angle_deg = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    residual = float(np.sqrt(np.mean(errors[inlier_mask] ** 2)))

    # 质量分同时考虑内点比例和拟合残差，范围限制在 0～1 便于快速比较。
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
    保存第一帧棋盘格作为固定参考，后续每帧都与同一参考比较。

    为什么要保存第一帧：
    程序关心的是“相对起始状态移动了多少”，不是棋盘格在图像中的绝对坐标。
    第一帧角点就是参考坐标系，后续帧角点与它比较得到 dx/dy/转角。
    """

    def __init__(self) -> None:
        # reference_corners 保存第一帧识别到的角点。
        self.reference_corners: np.ndarray | None = None

        # mm_per_pixel 保存第一帧估计出的毫米/像素比例。
        self.mm_per_pixel: float | None = None

    def _find_corners(self, gray: np.ndarray) -> np.ndarray | None:
        """
        在灰度图中寻找棋盘格内角点。

        返回值：
        - 找到时：形状为 (N, 2) 的角点坐标数组；
        - 找不到时：None。
        """

        # OpenCV 需要传入 (列数, 行数)，这些值来自 config.py。
        pattern = tuple(int(value) for value in config.CHECKERBOARD_INNER_CORNERS)

        # SB 算法对光照和透视通常更稳；旧版 OpenCV 没有该函数时自动退回传统算法。
        if hasattr(cv2, "findChessboardCornersSB"):
            # NORMALIZE_IMAGE 可减轻光照差异；EXHAUSTIVE 会更努力搜索，但稍慢。
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
            if found:
                # 统一整理成 N 行 2 列，并使用 float32，方便后续 OpenCV 函数处理。
                return corners.reshape(-1, 2).astype(np.float32)

        # 传统算法作为兜底，适配没有 SB 函数的 OpenCV 版本。
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not found:
            return None

        # 传统算法找到的是较粗略角点，cornerSubPix 会做亚像素级细化。
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
        由相邻内角点间距与真实方格边长估计参考帧的毫米/像素比例。

        简单说：
        - 已知真实棋盘格边长是 CHECKER_SQUARE_MM；
        - 图像中相邻角点距离是 spacing_px；
        - 所以 1 像素约等于 CHECKER_SQUARE_MM / spacing_px 毫米。
        """

        # corners 原本是 N 行 2 列；reshape 后变成“行、列、坐标”的网格结构。
        columns, rows = config.CHECKERBOARD_INNER_CORNERS
        grid = corners.reshape(rows, columns, 2)

        # 横向相邻角点距离和纵向相邻角点距离都参与估计，提高稳健性。
        horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
        vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()

        # 用中位数降低个别误差角点的影响。
        spacing_px = float(np.median(np.concatenate((horizontal, vertical))))

        if spacing_px <= 0:
            raise ValueError("棋盘格角点间距为零，无法建立毫米比例。")
        return float(config.CHECKER_SQUARE_MM / spacing_px)

    def process(self, gray: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        """
        处理一帧灰度图，输出棋盘格法的结果字典和角点坐标。

        结果字典字段统一以 checker_ 开头，避免和圆点法字段混淆。
        """

        # 第一步：尝试找棋盘格角点。
        corners = self._find_corners(gray)

        # 找不到角点时仍返回完整字段，只是标记 checker_is_valid=False。
        # 这样日志每一行字段结构尽量一致，后续分析更容易。
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

        # 第二步：第一帧成功识别时，把它保存为参考帧。
        if self.reference_corners is None:
            self.reference_corners = corners.copy()
            self.mm_per_pixel = self._calculate_mm_per_pixel(corners)

        # 第三步：取出参考点和比例尺。
        # 这些检查主要是帮助静态分析器，也让异常状态下报错更明确。
        reference_corners = self.reference_corners
        mm_per_pixel = self.mm_per_pixel
        if reference_corners is None or mm_per_pixel is None:
            raise RuntimeError("Checkerboard reference was not initialized.")

        # 第四步：把当前角点与参考角点配对，估计整体位移和转角。
        motion = _estimate_rigid_motion(
            reference_corners,
            corners,
            float(mm_per_pixel),
            config.CHECKER_RESIDUAL_WARNING_PX,
        )

        # 第五步：整理成写日志用的普通字典。
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
    从黑底/白底不确定的灰度图中寻找近似圆或椭圆轮廓。

    棋盘方格会在多边形近似中呈现约 4 个顶点，因此要求轮廓至少 7 个顶点以降低误检。
    """

    # Otsu 根据当前光照自动选阈值，THRESH_BINARY_INV 把黑色标记变成白色前景。
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    # findContours 会找出所有白色连通区域的边界。
    # 每个 contour 都可能是一个圆点，也可能是噪声、反光或背景物体。
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates: list[dict[str, float]] = []
    for contour in contours:
        # 第一层筛选：面积。太小多半是噪声，太大多半不是目标圆点。
        area = float(cv2.contourArea(contour))
        if area < config.CIRCLE_MIN_AREA_PX or area > config.CIRCLE_MAX_AREA_PX:
            continue

        # 第二层筛选：周长。周长为 0 或点太少，无法可靠描述形状。
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0 or len(contour) < 5:
            continue

        # 第三层筛选：圆度。圆度公式 4*pi*面积/周长^2，越接近 1 越像圆。
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < config.CIRCLE_MIN_CIRCULARITY:
            continue

        # 第四层筛选：多边形近似顶点数。
        # 棋盘格方块通常近似为 4 个顶点，圆形边界会有更多顶点。
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) < 7:
            continue

        # 第五层筛选：椭圆拟合。
        # 透视下圆点会变成椭圆，所以不要求完美圆，只要求长短轴比例别太极端。
        ellipse = cast(
            tuple[tuple[float, float], tuple[float, float], float],
            cv2.fitEllipse(contour),
        )
        (cx, cy), (axis_a, axis_b), _ = ellipse
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if major <= 0 or minor / major < config.CIRCLE_MIN_AXIS_RATIO:
            continue

        # 能走到这里的 contour 被认为是圆点候选。
        # 这里保存的都是普通 float，方便后续转 JSON 或 NumPy 数组。
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

    # 若环境里还有圆孔或反光点，优先保留面积较大的预期数量，再交给跨帧匹配约束。
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates[: config.CIRCLE_EXPECTED_COUNT]


def _layout_span_mm() -> float:
    """
    用理论布局中最远两点距离作为无需点身份的比例尺。

    圆点法没有像棋盘格那样的规则网格间距，
    所以这里用“理论圆点布局中最远两点距离”与图像中的最远距离建立比例。
    """

    # 从 config.py 读取预期圆点布局，只取当前启用的圆点数量。
    layout = np.asarray(config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT], dtype=float)

    # deltas[i, j] 是第 i 个点到第 j 个点的二维向量。
    deltas = layout[:, None, :] - layout[None, :, :]

    # 所有两两距离中的最大值就是布局跨度。
    return float(np.max(np.linalg.norm(deltas, axis=2)))


class CircleTracker:
    """
    用上一帧位置预测当前点身份，再始终回到第一帧计算总位移。

    这种设计适合高帧率连续视频；若两张离线图片之间跳得太远，应调大匹配距离或增加中间帧。
    """

    def __init__(self) -> None:
        # reference_points 是第一帧中各圆点的参考位置。
        self.reference_points: np.ndarray | None = None

        # last_points 是上一帧成功匹配后的圆点位置，用来预测下一帧。
        self.last_points: np.ndarray | None = None

        # last_step 是上一帧到当前帧的平均移动量，用来做简单预测。
        self.last_step = np.zeros(2, dtype=np.float32)

        # mm_per_pixel 是圆点布局估计出的毫米/像素比例。
        self.mm_per_pixel: float | None = None

    @staticmethod
    def _initial_order(points: np.ndarray) -> np.ndarray:
        """
        第一帧只需要给点一个稳定编号，按 y 后 x 排序即可。

        后续帧不再重新排序，因为振动时点会移动；
        后续身份由 _match_to_previous() 使用最近邻/匈牙利算法持续保持。
        """

        # np.lexsort((x, y)) 表示主要按 y 排序，y 相同或接近时按 x 排序。
        order = np.lexsort((points[:, 0], points[:, 1]))
        return points[order]

    def _initialize(self, points: np.ndarray) -> None:
        """
        用第一帧完整圆点建立参考坐标和比例尺。

        这一步只应在识别到全部预期圆点时执行。
        如果第一帧缺点，后续很难知道缺的是哪个点，身份编号会不可靠。
        """

        # 先给第一帧圆点一个稳定顺序，后面就把这个顺序当作身份编号。
        ordered = self._initial_order(points)

        # reference_points 永远代表“起始状态”，用于计算总位移。
        self.reference_points = ordered.copy()

        # last_points 代表“上一帧状态”，用于下一帧匹配。
        self.last_points = ordered.copy()

        # 计算图像中最远两圆点的像素距离，用来和理论布局的毫米距离对应。
        pairwise = ordered[:, None, :] - ordered[None, :, :]
        span_px = float(np.max(np.linalg.norm(pairwise, axis=2)))

        # 如果最远距离为 0，说明所有点重合或识别异常，比例尺没有意义。
        if span_px <= 0:
            raise ValueError("圆点中心重合，无法建立毫米比例。")

        # 毫米/像素比例 = 真实布局跨度 / 图像跨度。
        self.mm_per_pixel = _layout_span_mm() / span_px

    def _match_to_previous(self, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        匈牙利算法寻找“总体距离最小”的一一对应，避免多个旧点同时抢同一个新点。

        返回对应的参考点和当前点；超出最大允许距离的配对会被剔除。
        """

        # 取出上一帧状态和第一帧参考状态。
        # 如果还没初始化就调用匹配，说明代码流程错了，应立即报错。
        reference_points = self.reference_points
        last_points = self.last_points
        if reference_points is None or last_points is None:
            raise RuntimeError("Circle reference was not initialized.")

        # 用上一帧位置 + 上一步平均位移，粗略预测这一帧每个点会在哪里。
        # 这比直接拿上一帧位置匹配更能适应连续运动。
        predicted = last_points + self.last_step

        # distances[i, j] 表示“第 i 个旧点预测位置”和“第 j 个新检测点”的距离。
        distances = np.linalg.norm(predicted[:, None, :] - detected[None, :, :], axis=2)

        # 匈牙利算法会找出总体距离最小的一组一一配对。
        previous_indices, detected_indices = linear_sum_assignment(distances)

        # 下面再剔除距离过大的配对，避免把很远的错误点硬匹配上。
        accepted_previous: list[int] = []
        accepted_detected: list[int] = []
        for old_index, new_index in zip(previous_indices, detected_indices):
            if distances[old_index, new_index] <= config.CIRCLE_MAX_MATCH_DISTANCE_PX:
                accepted_previous.append(int(old_index))
                accepted_detected.append(int(new_index))

        # 可用点太少时，不做运动估计，返回空数组表示本帧圆点法无效。
        if len(accepted_previous) < config.MIN_VALID_CIRCLES:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # 把 Python list 转成 NumPy 索引数组，方便一次性取出所有匹配点。
        old_idx = np.asarray(accepted_previous, dtype=int)
        new_idx = np.asarray(accepted_detected, dtype=int)

        # current 是当前帧成功匹配到的新点。
        current = detected[new_idx]

        # reference 是这些点在第一帧里的对应位置，用于计算相对起始位移。
        reference = reference_points[old_idx]

        # 更新成功匹配点的位置；没有匹配到的点保留旧预测，等待下一帧重新出现。
        old_positions = last_points[old_idx].copy()
        last_points[old_idx] = current
        self.last_points = last_points

        # 用本帧匹配点的中位移动量估计下一帧整体移动方向。
        # 中位数比平均数更不容易被个别坏点影响。
        self.last_step = np.median(current - old_positions, axis=0).astype(np.float32)
        return reference, current

    def process(self, gray: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
        """
        处理一帧灰度图，输出圆点法结果字典和当前圆心坐标。

        结果字典字段统一以 circle_ 开头。
        """

        # 第一步：从图像中找出所有“可能是圆点”的轮廓。
        candidates = _find_circle_candidates(gray)

        # 第二步：只取候选点中心坐标，转成 N 行 2 列数组。
        detected = np.asarray(
            [[item["x"], item["y"]] for item in candidates],
            dtype=np.float32,
        )

        # 第三步：如果连最低有效点数都不够，本帧无法估计刚体位移。
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
            }, detected if len(detected) else None

        # 第四步：如果这是第一帧成功识别，需要建立参考。
        if self.reference_points is None:
            # 第一帧必须识别到完整数量，否则无法可靠编号。
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
                }, detected
            # 完整识别后，保存第一帧作为参考状态。
            self._initialize(detected)
            reference = self.reference_points
            current = self.last_points
        else:
            # 后续帧使用上一帧位置预测并匹配，保持圆点身份。
            reference, current = self._match_to_previous(detected)

        # 第五步：取出比例尺和当前/参考点。
        mm_per_pixel = self.mm_per_pixel
        if reference is None or current is None or mm_per_pixel is None:
            raise RuntimeError("Circle reference was not initialized.")

        # 第六步：匹配后如果有效点不足，也不能计算位移。
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
            }, detected

        # 第七步：估计当前帧相对第一帧的平移、转角、尺度和残差。
        motion = _estimate_rigid_motion(
            reference,
            current,
            float(mm_per_pixel),
            config.CIRCLE_RESIDUAL_WARNING_PX,
        )

        # 点数分进一步惩罚丢点情况，使“5 点勉强可算”不会和“7 点完整识别”得到相同质量。
        point_fraction = len(current) / config.CIRCLE_EXPECTED_COUNT
        quality = float(motion["quality"]) * point_fraction

        # 第八步：整理成日志字段。
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
        }
        return result, current


# =============================================================================
# 5. 单帧统一处理与调试画面
# =============================================================================

class VisionProcessor:
    """
    把预处理、两种识别和结果打包集中在一个对象中，以便保存跨帧参考状态。

    为什么需要“对象”而不是普通函数？
    因为棋盘格法和圆点法都要记住第一帧/上一帧的信息：
    - 第一帧提供“参考位置”，后面所有位移都相对它计算；
    - 圆点法还要记住上一帧圆点位置，防止不同圆点互相认错。
    """

    def __init__(self) -> None:
        self.checker_tracker = CheckerboardTracker()
        self.circle_tracker = CircleTracker()

    def process_frame(self, packet: FramePacket) -> tuple[dict[str, Any], np.ndarray]:
        # 第一步：把原始图像变成灰度图，并得到亮度、模糊等质量指标。
        gray, image_metrics, roi_origin = preprocess_frame(packet.frame)

        # 第二步：先建立一个基础结果字典。
        # 后面棋盘格/圆点算法会继续往这个字典里添加自己的字段。
        result: dict[str, Any] = {
            "kind": "VISION",
            "host_ns": int(packet.host_ns),
            "frame_id": int(packet.frame_id),
            "camera_timestamp_raw": packet.camera_timestamp_raw,
            "source_name": packet.source_name,
            **image_metrics,
        }

        # 离线图片/视频应按原始帧率分析，而不是按电脑“解码得有多快”分析。
        # 真实相机的原始 tick 单位尚未标定，因此实时场景仍使用可与 UR 共同对齐的 host_ns。
        if (
            packet.source_name.startswith(("image_folder:", "video:"))
            and packet.camera_timestamp_raw is not None
        ):
            result["analysis_time_s"] = float(packet.camera_timestamp_raw)
        else:
            result["analysis_time_s"] = float(packet.host_ns) * 1e-9

        # 这两个变量只用于画调试图。
        # 如果某种算法没有开启或没有识别成功，就保持 None。
        checker_corners: np.ndarray | None = None
        circle_centers: np.ndarray | None = None

        # 第三步：根据 VISION_METHOD 决定是否运行棋盘格法。
        # compare 模式会同时运行两种方法，方便你后面对比稳定性。
        if config.VISION_METHOD in {"checkerboard", "compare"}:
            checker_result, checker_corners = self.checker_tracker.process(gray)
            result.update(checker_result)

        # 第四步：根据 VISION_METHOD 决定是否运行圆点法。
        # 两种方法的结果字段名前缀不同，所以可以安全放进同一个 result。
        if config.VISION_METHOD in {"circles", "compare"}:
            circle_result, circle_centers = self.circle_tracker.process(gray)
            result.update(circle_result)

        # 第五步：只要任意一种方法有效，就认为这一帧有可用视觉结果。
        valid_flags = [
            bool(result.get("checker_is_valid", False)),
            bool(result.get("circle_is_valid", False)),
        ]
        result["is_valid"] = any(valid_flags)

        # 第六步：生成调试图。调试图不参与计算，只帮助人检查算法到底看见了什么。
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
        把算法看见的点和最终数值画出来，便于判断是曝光问题、漏点还是串点。

        返回值是带标注的图像副本：
        - 绿色小点：棋盘格角点；
        - 红色圆圈和编号：圆点法识别到的圆心；
        - 左上角文字：位移、质量分、亮度和模糊指标。
        """

        # 如果原图是灰度，先转成 BGR 彩色图，方便画彩色标记。
        if original.ndim == 2:
            canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            # 如果原图已经是彩色，就复制一份，避免修改原始 frame。
            canvas = original.copy()

        # 检测是在 ROI 坐标系里做的；画回原图时要加上 ROI 左上角偏移。
        offset = np.asarray(roi_origin, dtype=np.float32)

        # 画棋盘格角点。
        if checker_corners is not None:
            for point in checker_corners:
                x, y = np.rint(point + offset).astype(int)
                cv2.circle(canvas, (x, y), 3, (0, 180, 0), -1, cv2.LINE_AA)

        # 画圆点中心和编号。
        # 编号可以帮助你观察跨帧身份是否发生串号。
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

        # 从 result 字典取数值时统一兜底，避免 None 或缺字段导致调试图绘制失败。
        def metric(name: str, default: float) -> float:
            value = result.get(name, default)
            return float(default if value is None else value)

        # 左上角要显示的调试文字。
        lines = [
            f"frame={result['frame_id']} valid={result['is_valid']}",
            (
                "circle: "
                f"dx={metric('circle_dx_mm', math.nan):.4f} mm  "
                f"dy={metric('circle_dy_mm', math.nan):.4f} mm  "
                f"q={metric('circle_quality', 0.0):.2f}"
            ),
            (
                "checker: "
                f"dx={metric('checker_dx_mm', math.nan):.4f} mm  "
                f"dy={metric('checker_dy_mm', math.nan):.4f} mm  "
                f"q={metric('checker_quality', 0.0):.2f}"
            ),
            (
                f"brightness={metric('mean_brightness', math.nan):.1f}  "
                f"blurVar={metric('blur_variance', math.nan):.1f}"
            ),
        ]

        # 每行文字先画白色粗线，再画黑色细线，保证在亮/暗背景上都能看清。
        for line_index, text in enumerate(lines):
            y = 28 + line_index * 25
            cv2.putText(
                canvas,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

        if result["blur_variance"] < config.BLUR_WARNING_THRESHOLD:
            cv2.putText(
                canvas,
                "WARNING: image may be blurred",
                (12, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas


# =============================================================================
# 6. 复合测量纸与合成测试图
# =============================================================================

def generate_marker_sheet(output_path: Path | None = None) -> Path:
    """
    生成按毫米定义的 SVG，避免普通 PNG 在打印时因 DPI 设置而缩放。

    打印对话框必须选择 100%/实际大小；打印后应再用游标卡尺测量方格边长确认比例。

    输出内容：
    - 左侧：棋盘格，用于 checkerboard 方法；
    - 右侧：不对称圆点阵列，用于 circles 方法；
    - 外框：浅灰裁剪参考线，尽量不干扰黑色标记检测。
    """

    # 如果调用方没给 output_path，就使用 config.py 中的默认 SVG 路径。
    output = Path(output_path or config.MARKER_SHEET_PATH)

    # 输出目录可能还不存在，先创建。
    output.parent.mkdir(parents=True, exist_ok=True)

    # 从 config.py 读取纸张尺寸、边距和棋盘格参数。
    width = float(config.MARKER_SHEET_WIDTH_MM)
    height = float(config.MARKER_SHEET_HEIGHT_MM)
    margin = float(config.MARKER_MARGIN_MM)
    columns, rows = config.CHECKERBOARD_INNER_CORNERS
    square = float(config.CHECKER_SQUARE_MM)
    board_columns = columns + 1
    board_rows = rows + 1

    # 棋盘格实际方格数量 = 内角点数量 + 1。
    checker_width = board_columns * square
    checker_height = board_rows * square

    # 棋盘格放在纸张左侧，垂直方向居中。
    board_x = margin
    board_y = (height - checker_height) / 2.0

    # 圆点布局来自 config.CIRCLE_LAYOUT_MM，单位本来就是毫米。
    layout = np.asarray(
        config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT],
        dtype=float,
    )
    # 圆点区域放在纸张右侧，垂直方向也尽量居中。
    circle_x = width - margin - float(np.max(layout[:, 0]))
    circle_y = (height - float(np.max(layout[:, 1]))) / 2.0

    # 如果纸张太窄，左右两种图案会重叠，此时直接报错，不生成错误图纸。
    if board_x + checker_width + margin > circle_x:
        raise ValueError("当前测量纸太窄，棋盘格与圆点区域会重叠。")

    # SVG 本质上是 XML 文本。这里逐行拼接，最后写入文件。
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" '
            f'height="{height}mm" viewBox="0 0 {width} {height}">'
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    # 绘制棋盘格：只画黑色方块，白色区域由背景矩形提供。
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                x = board_x + column * square
                y = board_y + row * square
                lines.append(
                    f'<rect x="{x:.4f}" y="{y:.4f}" '
                    f'width="{square:.4f}" height="{square:.4f}" fill="black"/>'
                )

    # 绘制圆点：第 0 个点稍大，作为人工观察时的方向锚点。
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

    # 外边界采用浅灰细线，帮助裁剪但尽量不参与黑色轮廓检测。
    lines.append(
        f'<rect x="0.2" y="0.2" width="{width - 0.4}" height="{height - 0.4}" '
        'fill="none" stroke="#BBBBBB" stroke-width="0.2"/>'
    )
    lines.append("</svg>")

    # SVG 是普通文本，使用 UTF-8 写入即可；这里不依赖额外 PDF 库。
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def create_synthetic_demo_sequence(
    output_folder: Path,
    frame_count: int = 90,
    fps: float = 60.0,
) -> Path:
    """
    生成仅用于检查代码链路的合成序列，不代表真实相机精度。

    它会给复合图案加入小幅正弦平移与转动，适合在没有设备时验证识别、日志和分析函数。

    注意：
    - 这不是相机标定数据；
    - 这不是实验数据；
    - 它只是为了验证程序的输入/处理/输出链路能跑通。
    """

    # 准备输出文件夹。
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 先创建一张白色背景的大图。
    image = np.full((520, 960, 3), 255, dtype=np.uint8)

    # 下面这些参数决定合成棋盘格在图像中的大小和位置。
    square_px = 46
    columns, rows = config.CHECKERBOARD_INNER_CORNERS
    board_columns, board_rows = columns + 1, rows + 1
    board_x, board_y = 70, 110

    # 在白底图上画棋盘格黑色方块。
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                p1 = (board_x + column * square_px, board_y + row * square_px)
                p2 = (p1[0] + square_px, p1[1] + square_px)
                cv2.rectangle(image, p1, p2, (0, 0, 0), -1)

    # 读取理论圆点布局，再按固定比例换成像素坐标。
    layout = np.asarray(
        config.CIRCLE_LAYOUT_MM[: config.CIRCLE_EXPECTED_COUNT],
        dtype=float,
    )
    circle_origin = np.asarray([600.0, 135.0])
    circle_scale_px_per_mm = 9.0
    # 在同一张合成图上画圆点阵列。
    for index, point in enumerate(layout):
        center = np.rint(circle_origin + point * circle_scale_px_per_mm).astype(int)
        radius = 19 if index == 0 else 13
        cv2.circle(image, tuple(center), radius, (0, 0, 0), -1, cv2.LINE_AA)

    # 下面开始生成多帧：每一帧都对整张图做一个小幅平移和旋转。
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    for frame_id in range(frame_count):
        # t 是当前帧对应的时间，单位秒。
        t = frame_id / fps

        # 三个正弦项分别模拟 x 方向振动、y 方向振动和平面内轻微转动。
        dx_px = 3.2 * math.sin(2.0 * math.pi * 8.0 * t)
        dy_px = 2.2 * math.sin(2.0 * math.pi * 13.0 * t + 0.4)
        angle_deg = 0.12 * math.sin(2.0 * math.pi * 5.0 * t)

        # getRotationMatrix2D 生成旋转矩阵；后面再手动叠加平移量。
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        matrix[0, 2] += dx_px
        matrix[1, 2] += dy_px
        # warpAffine 根据矩阵把原图变换成当前帧。
        moved = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        # 保存为按序号命名的 PNG，方便 ImageFolderSource 按自然顺序读取。
        cv2.imwrite(str(output_folder / f"frame_{frame_id:04d}.png"), moved)

    return output_folder


# =============================================================================
# 7. 视觉测试模式与完整实验子进程
# =============================================================================

def _resize_for_preview(image: np.ndarray) -> np.ndarray:
    """
    仅缩小显示副本，不改变送入算法和保存到磁盘的原始分辨率。

    OpenCV 预览窗口太大时会超出屏幕，所以这里只为显示缩小。
    算法计算和调试图保存仍使用原始尺寸。
    """

    # 如果图像宽度已经不超过限制，直接原样返回。
    if image.shape[1] <= config.PREVIEW_MAX_WIDTH:
        return image

    # 按宽度比例计算缩放倍数，高度按同一比例缩放，避免图像变形。
    scale = config.PREVIEW_MAX_WIDTH / image.shape[1]
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

    # INTER_AREA 适合缩小图像，视觉上更平滑。
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _json_line(record: dict[str, Any]) -> str:
    """
    把一条记录字典转成一行 JSON 文本。

    ensure_ascii=False 让中文字段和中文报错保持可读；
    allow_nan=True 允许 math.nan 写入日志，后续分析会按无效值处理。
    """

    # json 是标准库，但只在写日志时需要，所以放在这里也可以。
    import json

    # 返回值不带换行；调用方负责追加 "\n"。
    return json.dumps(record, ensure_ascii=False, allow_nan=True)


def run_vision_test() -> Path:
    """
    单进程运行视觉模块，保存逐帧结果和可选调试图。

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

    # Processor 会记住第一帧参考点，所以一次测试只创建一个实例，并连续处理所有帧。
    processor = VisionProcessor()
    result_path = run_dir / "vision_results.txt"
    processed_count = 0
    valid_count = 0
    saved_debug_count = 0

    try:
        with open_image_source() as source, result_path.open("w", encoding="utf-8") as file:
            meta = {
                "kind": "META",
                "host_ns": time.perf_counter_ns(),
                "mode": "vision_test",
                "vision_source": config.VISION_SOURCE,
                "vision_method": config.VISION_METHOD,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            file.write(_json_line(meta) + "\n")

            # source 可以是图片文件夹、视频或相机。
            # 不管来源是什么，for 循环拿到的都是统一的 FramePacket。
            for packet in source:
                result, debug = processor.process_frame(packet)
                file.write(_json_line(result) + "\n")
                processed_count += 1
                valid_count += int(bool(result["is_valid"]))

                # 调试图很有用，但每帧都保存会很占硬盘。
                # 因此这里按帧间隔和最大张数两个条件限制保存数量。
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

                # 预览窗口只影响人眼查看，不影响保存的结果。
                # 按 Esc 或 q 会提前结束当前视觉测试。
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
    把相机结果放入记录队列。

    返回 True 表示写入成功；返回 False 表示队列满了，并且已经设置 stop_event。
    队列满通常意味着写盘或主流程跟不上实时数据，继续采集会造成时间对齐失真。
    """

    try:
        # timeout=1.0 防止队列满时永远卡住。
        record_queue.put(record, timeout=1.0)
        return True
    except Full:
        # 写不进去时主动通知其他进程停下来。
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
    完整实验中的相机子进程。

    它打开相机后先报告 ready，等主程序发出 start_event 才正式把 VISION 结果送入记录队列。

    这个函数和 run_vision_test() 的区别：
    - run_vision_test() 自己写 vision_results.txt；
    - camera_worker() 不直接写文件，而是把结果放到 record_queue；
    - 真正写文件的是 main.py 创建的 record_writer_worker。
    """

    try:
        # 正式实验中的相机进程也使用同一个 VisionProcessor。
        # 这样第一帧参考点和跨帧圆点身份可以连续保存。
        processor = VisionProcessor()
        with open_image_source() as source:
            iterator = iter(source)

            # 先真正取到并处理一帧，才能说明“相机和算法都已就绪”，而不是仅仅创建了对象。
            first_packet = next(iterator)
            first_result, _ = processor.process_frame(first_packet)
            first_result["analysis_time_s"] = first_result["host_ns"] * 1e-9
            camera_ready.set()

            if config.VISION_SOURCE == "hik_camera":
                # 人工确认可能持续数秒；真实相机必须持续取流，避免启动后先读到缓冲区里的旧帧。
                # 这里只保留最新结果，不写正式记录，直到主程序发出统一 start_event。
                while not start_event.is_set():
                    if stop_event.is_set():
                        return
                    warmup_packet = next(iterator)
                    first_result, _ = processor.process_frame(warmup_packet)
                    first_result["analysis_time_s"] = first_result["host_ns"] * 1e-9
            else:
                # 离线图片或视频若在等待期间高速解码会提前耗尽，所以只等待而不继续取下一帧。
                while not start_event.is_set():
                    if stop_event.wait(0.05):
                        return

            # start_event 到来后，先把等待期间保留的第一条/最新一条结果写入正式日志。
            if not _put_record(record_queue, first_result, stop_event):
                raise RuntimeError("记录队列已满，相机结果无法写入。")

            # 正式开始后持续处理后续帧，直到 stop_event 亮起或来源耗尽。
            for packet in iterator:
                # 其他进程要求停止时，相机进程应尽快退出循环。
                if stop_event.is_set():
                    break

                # 处理当前帧，得到结果字典和调试图。
                result, debug = processor.process_frame(packet)

                # 完整实验必须与 EVENT 和 ROBOT 共用电脑时基，不能改用视频内部的相对零点。
                result["analysis_time_s"] = result["host_ns"] * 1e-9

                # 把视觉结果交给记录进程。
                if not _put_record(record_queue, result, stop_event):
                    raise RuntimeError("记录队列已满，相机结果无法写入。")

                # 正式实验中一般不建议开预览；如果开启，按 q/Esc 会请求停止整套实验。
                if config.SHOW_PREVIEW:
                    cv2.imshow("UR10 vibration experiment", _resize_for_preview(debug))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        stop_event.set()
                        break

            # 如果不是收到 stop_event，而是图像来源自己结束了，说明实验数据长度不够。
            if not stop_event.is_set():
                raise RuntimeError(
                    "图像来源在实验停止信号到来前已经结束。"
                    "正式实验应使用连续相机流，离线文件需保证长度覆盖全部运动和后记录。"
                )
    except StopIteration:
        # 连第一帧都没有读到时会进入这里。
        error_queue.put("相机来源没有返回任何一帧。")
        stop_event.set()
    except Exception as exc:
        # 把错误文本发给主程序，让主程序统一停止其他子进程。
        error_queue.put(f"相机进程异常：{type(exc).__name__}: {exc}")
        stop_event.set()
    finally:
        # 如果打开了 OpenCV 窗口，退出前关闭窗口资源。
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()
