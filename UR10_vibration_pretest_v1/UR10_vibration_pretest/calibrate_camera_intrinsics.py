"""
相机内参标定工具：从一批棋盘格照片计算 CAMERA_MATRIX 和 DISTORTION_COEFFICIENTS。

这个文件可以单独运行，不会连接相机、不会连接 UR、不会修改 config.py。
它只读取你提前拍好的标定图片，输出：
- camera_intrinsics.json：机器可读的完整标定结果；
- config_snippet.txt：可直接复制进 config.py 的 CAMERA_MATRIX / DISTORTION_COEFFICIENTS；
- detection_debug/：每张标定图的角点识别检查图；
- undistort_preview.jpg：用求出的内参去畸变后的预览图。

main()
脚本总入口。读取命令行参数，收集标定图片，调用标定流程，输出结果文件夹。

_parse_arguments()
解析命令行参数，比如图片文件夹、棋盘格内角点数量、小格边长。

_collect_images()
从 calibration_images 或指定文件夹里收集标定图片。

_find_corners()
在每张标定图里找棋盘格内角点。

_calibrate()
真正调用 OpenCV 标定，计算 CAMERA_MATRIX 和 DISTORTION_COEFFICIENTS。

_write_outputs()
保存 camera_intrinsics.json、config_snippet.txt、调试图和去畸变预览图。


推荐操作流程：
1. 打印棋盘格标定板。
   - 使用普通平整纸张或更硬的板材固定，标定板必须尽量平整。
   - config.py 当前默认 CHECKERBOARD_INNER_CORNERS = (7, 5)，表示内角点是 7 列、5 行。
   - 注意这是“内角点数量”，不是黑白方格数量；7×5 内角点对应 8×6 个黑白方格。
   - CHECKER_SQUARE_MM = 4.0 表示每个黑白方格边长 4 mm；如果你的标定板不是 4 mm，运行时必须改参数。

2. 拍摄标定照片。
   - 使用正式实验同一台相机、同一分辨率、同一镜头焦距、同一对焦状态。
   - 不要拍完标定后再调焦、变焦或改分辨率；否则内参会失效。
   - 建议拍 20-40 张，至少 12 张能成功识别角点。
   - 标定板要出现在画面不同位置：中心、四角、边缘都要覆盖。
   - 标定板要有不同姿态：正对、左右倾斜、上下倾斜、远近变化都要有。
   - 每张图里整块内角点必须清楚可见，不能被遮挡、严重反光、过曝或运动模糊。
   - 不要所有照片都几乎一模一样；姿态太单一会让标定结果不稳。

3. 把照片放到项目目录下的 calibration_images 文件夹。
   - 默认路径是：UR10_vibration_pretest/calibration_images
   - 支持 .bmp、.png、.jpg、.jpeg、.tif、.tiff。

4. 运行脚本。
   在本目录下执行：
       .venv\\Scripts\\python.exe calibrate_camera_intrinsics.py

   如果你的方格边长不是 config.py 里的默认值，例如 10 mm：
       .venv\\Scripts\\python.exe calibrate_camera_intrinsics.py --square-mm 10

   如果你的图片放在别的文件夹：
       .venv\\Scripts\\python.exe calibrate_camera_intrinsics.py --images D:\\calib_images

   如果你的标定板内角点不是 7×5，例如 9×6：
       .venv\\Scripts\\python.exe calibrate_camera_intrinsics.py --board-cols 9 --board-rows 6

5. 检查输出。
   - 先看终端中的 RMS reprojection error。
   - 再看 detection_debug 文件夹：确认绿色角点都落在真实棋盘格内角点上。
   - 再看 undistort_preview.jpg：确认去畸变后图像没有明显奇怪拉伸。

6. 把 config_snippet.txt 中的两段复制进 config.py。
   复制后，preprocess_frame() 会使用 cv2.undistort() 去畸变；
   calibration.py 也会用这些内参把像素点转换成空间射线。

常见判断标准：
- RMS 重投影误差越小越好，通常小于 0.5 px 较好，0.5-1.0 px 可用但要看调试图，
  大于 1.0 px 应优先检查照片是否模糊、棋盘尺寸是否写错、角点数量是否写错。
- 单张图片误差明显高于其他图片时，可以删掉那张照片后重新运行。
- 如果很多图片识别失败，多半是内角点数量填错、标定板被裁掉、图像太糊或反光过强。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import cv2
import numpy as np

import config


IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


@dataclass(slots=True)
class ImageDetection:
    """一张标定图的角点检测结果，用于终端摘要和 JSON 输出。"""

    image_path: str
    found: bool
    reprojection_error_px: float | None = None
    reason: str | None = None


def _natural_sort_key(path: Path) -> list[int | str]:
    """让 frame2 排在 frame10 前面，避免文件名顺序干扰人工排查。"""

    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _collect_images(image_dir: Path) -> list[Path]:
    """
    收集标定图片。

    输入：包含标定照片的文件夹。
    输出：按自然文件名顺序排列的图片路径列表。
    """

    if not image_dir.exists():
        raise FileNotFoundError(f"标定图片文件夹不存在：{image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"标定图片路径不是文件夹：{image_dir}")

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=_natural_sort_key)
    if not images:
        raise FileNotFoundError(f"{image_dir} 中没有可用标定图片。")
    return images


def _make_object_points(
    board_size: tuple[int, int],
    square_mm: float,
) -> np.ndarray:
    """
    生成标定板在自己纸面坐标系中的真实角点坐标。

    输入：
    - board_size：内角点列数、行数；
    - square_mm：单个黑白方格边长，单位 mm。

    输出：N×3 的三维点，z=0，单位 mm。
    """

    columns, rows = board_size
    object_points = np.zeros((columns * rows, 3), np.float32)
    grid = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * float(square_mm)
    return object_points


def _find_corners(
    gray: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """
    在一张灰度图中寻找棋盘格内角点。

    输入：灰度图和内角点数量。
    输出：是否找到，以及 N×1×2 的角点数组。
    """

    if hasattr(cv2, "findChessboardCornersSB"):
        flags_sb = (
            cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags=flags_sb)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(config.CHECKER_SUBPIX_MAX_ITER),
        float(config.CHECKER_SUBPIX_EPS),
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def _draw_detection_debug(
    image: np.ndarray,
    board_size: tuple[int, int],
    found: bool,
    corners: np.ndarray | None,
    output_path: Path,
) -> None:
    """
    保存单张标定图的角点识别检查图。

    绿色/彩色角点应准确落在棋盘格内角点上；如果明显错位，应删掉该图或修正参数。
    """

    canvas = image.copy()
    cv2.drawChessboardCorners(canvas, board_size, corners, found)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _calibrate(
    images: list[Path],
    board_size: tuple[int, int],
    square_mm: float,
    debug_dir: Path,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    tuple[int, int],
    list[ImageDetection],
]:
    """
    执行完整内参标定。

    输入：标定图片、内角点数量、方格尺寸和调试图输出目录。
    输出：RMS、相机矩阵、畸变系数、每张图外参、图像尺寸和逐图检测信息。
    """

    object_template = _make_object_points(board_size, square_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    detections: list[ImageDetection] = []
    image_size: tuple[int, int] | None = None

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            detections.append(
                ImageDetection(
                    image_path=str(image_path),
                    found=False,
                    reason="OpenCV 无法读取该图片。",
                )
            )
            continue

        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            detections.append(
                ImageDetection(
                    image_path=str(image_path),
                    found=False,
                    reason=(
                        f"图片尺寸 {current_size} 与第一张图片尺寸 {image_size} 不一致。"
                    ),
                )
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = _find_corners(gray, board_size)
        _draw_detection_debug(
            image,
            board_size,
            found,
            corners,
            debug_dir / f"{image_path.stem}_corners.jpg",
        )

        if not found or corners is None:
            detections.append(
                ImageDetection(
                    image_path=str(image_path),
                    found=False,
                    reason="未找到完整棋盘格内角点。",
                )
            )
            continue

        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float32))
        detections.append(ImageDetection(image_path=str(image_path), found=True))

    if image_size is None:
        raise RuntimeError("没有任何图片能被 OpenCV 读取。")
    if len(image_points) < 3:
        raise RuntimeError(
            f"只有 {len(image_points)} 张图片成功识别角点，太少，无法标定。"
        )

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    for index, detection in enumerate([item for item in detections if item.found]):
        projected, _ = cv2.projectPoints(
            object_points[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            distortion,
        )
        error = cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
        detection.reprojection_error_px = float(error)

    return (
        float(rms),
        camera_matrix,
        distortion.reshape(-1),
        rvecs,
        tvecs,
        image_size,
        detections,
    )


def _write_outputs(
    output_dir: Path,
    image_dir: Path,
    board_size: tuple[int, int],
    square_mm: float,
    image_size: tuple[int, int],
    rms: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    detections: list[ImageDetection],
) -> None:
    """
    保存 JSON、config.py 片段和去畸变预览。

    输出文件都写入 output_dir，不会自动修改 config.py。
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "kind": "CAMERA_INTRINSICS_CALIBRATION",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_dir": str(image_dir),
        "image_size_px": {"width": image_size[0], "height": image_size[1]},
        "board_inner_corners": {"columns": board_size[0], "rows": board_size[1]},
        "square_mm": float(square_mm),
        "rms_reprojection_error_px": rms,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "successful_image_count": sum(1 for item in detections if item.found),
        "total_image_count": len(detections),
        "detections": [asdict(item) for item in detections],
    }
    (output_dir / "camera_intrinsics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    matrix_text = json.dumps(camera_matrix.tolist(), ensure_ascii=False, indent=4)
    distortion_text = json.dumps(distortion.tolist(), ensure_ascii=False, indent=4)
    snippet = (
        "# 将下面两段复制到 config.py 中对应位置。\n"
        "# 注意：如果改了分辨率、镜头焦距或对焦状态，需要重新标定。\n"
        f"CAMERA_MATRIX: list[list[float]] | None = {matrix_text}\n\n"
        f"DISTORTION_COEFFICIENTS: list[float] | None = {distortion_text}\n"
    )
    (output_dir / "config_snippet.txt").write_text(snippet, encoding="utf-8")

    first_success = next((Path(item.image_path) for item in detections if item.found), None)
    if first_success is not None:
        preview = cv2.imread(str(first_success), cv2.IMREAD_COLOR)
        if preview is not None:
            undistorted = cv2.undistort(preview, camera_matrix, distortion)
            cv2.imwrite(str(output_dir / "undistort_preview.jpg"), undistorted)


def _parse_arguments() -> argparse.Namespace:
    """解析命令行参数，让本脚本可以独立用于不同相机和不同棋盘格。"""

    parser = argparse.ArgumentParser(
        description="从棋盘格标定照片计算相机内参和畸变系数。"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=config.PROJECT_DIR / "calibration_images",
        help="标定图片文件夹，默认是项目下的 calibration_images。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件夹；默认写入 outputs/camera_calibration_时间戳。",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=int(config.CHECKERBOARD_INNER_CORNERS[0]),
        help="棋盘格内角点列数，不是方格列数。",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=int(config.CHECKERBOARD_INNER_CORNERS[1]),
        help="棋盘格内角点行数，不是方格行数。",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=float(config.CHECKER_SQUARE_MM),
        help="单个黑白方格边长，单位 mm。",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=12,
        help="建议的最少成功识别图片数；低于该值会报错。",
    )
    return parser.parse_args()


def main() -> Path:
    """
    脚本总入口。

    输入：命令行参数和 calibration_images 中的照片。
    输出：标定结果文件夹 Path。
    """

    args = _parse_arguments()
    image_dir = Path(args.images).expanduser()
    if not image_dir.is_absolute():
        image_dir = (config.PROJECT_DIR / image_dir).resolve()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.OUTPUT_ROOT / f"camera_calibration_{timestamp}"
    else:
        output_dir = Path(args.output).expanduser()
        if not output_dir.is_absolute():
            output_dir = (config.PROJECT_DIR / output_dir).resolve()

    board_size = (int(args.board_cols), int(args.board_rows))
    square_mm = float(args.square_mm)
    if board_size[0] < 2 or board_size[1] < 2:
        raise ValueError("棋盘格内角点列数和行数都必须至少为 2。")
    if square_mm <= 0:
        raise ValueError("square-mm 必须为正数。")

    images = _collect_images(image_dir)
    debug_dir = output_dir / "detection_debug"
    (
        rms,
        camera_matrix,
        distortion,
        _rvecs,
        _tvecs,
        image_size,
        detections,
    ) = _calibrate(images, board_size, square_mm, debug_dir)

    success_count = sum(1 for item in detections if item.found)
    if success_count < int(args.min_images):
        raise RuntimeError(
            f"成功识别 {success_count} 张，少于建议下限 {int(args.min_images)} 张。"
            "请补拍更多角度、位置不同且清晰的标定照片后重试。"
        )

    _write_outputs(
        output_dir,
        image_dir,
        board_size,
        square_mm,
        image_size,
        rms,
        camera_matrix,
        distortion,
        detections,
    )

    errors = [
        item.reprojection_error_px
        for item in detections
        if item.reprojection_error_px is not None
    ]
    print("[标定] 完成相机内参标定。")
    print(f"[标定] 成功图片：{success_count}/{len(detections)}")
    print(f"[标定] 图像尺寸：{image_size[0]}×{image_size[1]} px")
    print(f"[标定] RMS 重投影误差：{rms:.6f} px")
    if errors:
        print(
            f"[标定] 单图误差：平均 {float(np.mean(errors)):.6f} px，"
            f"最大 {float(np.max(errors)):.6f} px"
        )
    print(f"[标定] 结果文件夹：{output_dir}")
    print(f"[标定] 请检查：{output_dir / 'detection_debug'}")
    print(f"[标定] 可复制到 config.py：{output_dir / 'config_snippet.txt'}")
    return output_dir


if __name__ == "__main__":
    main()
