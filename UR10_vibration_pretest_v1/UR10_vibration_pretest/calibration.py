"""
像素坐标到空间坐标的可选转换。calibrate_camera_...是求内参，
而calibration是使用内参和固定深度平面或外参（只在config里预留了接口，默认是None）（固定深度平面和外参有一个就够了），把像素点解释为空间点

本文件只处理“坐标解释”，不参与圆点/棋盘格识别：
- 输入：视觉模块已经识别出的图像像素坐标；
- 输出：相机坐标系或机器人坐标系下的三维点；
- 前提：config.py 中已经填写相机内参，以及对应模式需要的平面/外参信息。

image_points_to_spatial()
总入口。把一组像素点转换成空间点，结果写入每帧 JSON 的 *_spatial 字段。

_normalized_rays()
使用内参和畸变参数，把像素点转换成相机坐标系里的归一化射线。

_camera_plane_points()
camera_plane 模式：把相机射线投到固定深度 Z=SPATIAL_CAMERA_PLANE_Z_M 的平面上。这种方式不需要外参。

_robot_plane_points()
robot_plane 模式：用相机到机器人外参，把射线投到机器人坐标系中的测量平面上。这种方式不需要固定深度平面。

_as_points()
整理和校验输入像素点，确保是 N×2 的有效数值数组。


这样设计的目的是把问题拆开：
如果像素坐标已经错了，就回到 camera.py 排查识别；
如果像素坐标对、空间坐标错，再回到这里检查标定和坐标系定义。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import config


def _as_points(points_px: np.ndarray | list[list[float]]) -> np.ndarray:
    """
    校验并整理视觉模块传来的像素点。

    输入：JSON 列表或 NumPy 数组形式的像素坐标。
    输出：N×2 的浮点数组，每行是一个图像点。
    实验作用：空间转换只应处理已经明确存在且数值有限的识别点，避免标定错误被空点或 NaN 掩盖。
    """

    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    if not np.all(np.isfinite(points)):
        raise ValueError("像素坐标中存在 NaN 或无穷值。")
    return points


def _normalized_rays(points_px: np.ndarray) -> np.ndarray:
    """
    把图像像素点转换成相机坐标系中的归一化射线。

    输入：整幅图像坐标系中的像素点，以及 config.py 里的相机内参/畸变。
    输出：每行形如 [x, y, 1] 的方向向量，表示从相机光心穿过该像素的射线。

    实验作用：
    像素坐标本身只是图像位置。要得到空间点，必须先把像素点变成相机光线，
    再让这条光线与已知测量平面求交。
    """

    if config.CAMERA_MATRIX is None:
        raise RuntimeError("空间坐标输出需要先填写 CAMERA_MATRIX。")

    # 本段准备相机标定参数。
    # 输入来自 config.py；畸变未填写时使用零畸变，保证 camera_plane 模式仍能只依赖内参运行。
    camera_matrix = np.asarray(config.CAMERA_MATRIX, dtype=np.float64)
    distortion = (
        np.zeros(5, dtype=np.float64)
        if config.DISTORTION_COEFFICIENTS is None
        else np.asarray(config.DISTORTION_COEFFICIENTS, dtype=np.float64)
    )

    # 本段执行“像素点 -> 去畸变归一化坐标”。
    # 输出不是三维点，只是从相机光心出发的方向；真正的三维位置由后面的平面求交决定。
    normalized = cv2.undistortPoints(
        points_px.reshape(-1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(-1, 2)
    return np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))


def _camera_plane_points(rays_camera: np.ndarray) -> tuple[np.ndarray, None]:
    """
    camera_plane 模式：把像素射线投到相机坐标系 Z=固定深度 的平面。

    输入：相机坐标系归一化射线，以及 config.SPATIAL_CAMERA_PLANE_Z_M。
    输出：相机坐标系下的三维点；robot_m 保持 None。

    实验作用：
    这是较简单的空间坐标模式，适合目标平面近似正对相机、且你已知平面深度的预实验。
    """

    if config.SPATIAL_CAMERA_PLANE_Z_M is None:
        raise RuntimeError("camera_plane 模式需要 SPATIAL_CAMERA_PLANE_Z_M。")

    # 本段用固定深度把方向向量变成空间点。
    # 因为归一化射线第三项为 1，所以乘以 Z 后就得到相机坐标系下的 [X, Y, Z]。
    z = float(config.SPATIAL_CAMERA_PLANE_Z_M)
    return rays_camera * z, None


def _robot_plane_points(rays_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    robot_plane 模式：用外参把相机射线投到机器人坐标系中的测量平面。

    输入：
    - 相机坐标系归一化射线；
    - 相机到机器人坐标系的 R/t；
    - 机器人坐标系中测量平面的一个点和法向。

    输出：
    - camera_m：点在相机坐标系中的三维坐标；
    - robot_m：同一点在机器人坐标系中的三维坐标。

    实验作用：
    这是更接近正式实验的空间坐标模式。它把视觉点直接放到机器人坐标系中，
    但前提是相机外参和平面定义都已经可信。
    """

    # 本段读取外参和平面参数。
    # 这些参数决定“相机看见的像素射线”如何落到机器人坐标系中的测量平面上。
    rotation = np.asarray(config.CAMERA_TO_ROBOT_ROTATION, dtype=np.float64)
    translation = np.asarray(config.CAMERA_TO_ROBOT_TRANSLATION_M, dtype=np.float64)
    plane_point = np.asarray(config.MEASUREMENT_PLANE_POINT_ROBOT_M, dtype=np.float64)
    plane_normal = np.asarray(config.MEASUREMENT_PLANE_NORMAL_ROBOT, dtype=np.float64)

    normal_norm = float(np.linalg.norm(plane_normal))
    if normal_norm <= 0:
        raise RuntimeError("MEASUREMENT_PLANE_NORMAL_ROBOT 不能是零向量。")
    plane_normal = plane_normal / normal_norm

    # 本段把相机射线方向转换到机器人坐标系，并与测量平面求交。
    # 输出的 distances 表示每条射线从相机光心走多远会碰到测量平面。
    directions_robot = rays_camera @ rotation.T
    denominator = directions_robot @ plane_normal
    numerator = float((plane_point - translation) @ plane_normal)

    if np.any(np.abs(denominator) < 1e-9):
        raise RuntimeError("存在几乎平行于测量平面的像素射线，无法求交。")

    distances = numerator / denominator
    if np.any(distances <= 0):
        raise RuntimeError("部分像素射线与测量平面的交点位于相机后方。")

    # 本段生成两个坐标系下的同一批空间点。
    # camera_points 便于检查相机侧标定，robot_points 便于和 UR TCP 或轨迹数据比较。
    camera_points = rays_camera * distances[:, None]
    robot_points = translation + directions_robot * distances[:, None]
    return camera_points, robot_points


def image_points_to_spatial(points_px: np.ndarray | list[list[float]]) -> dict[str, Any]:
    """
    将一组图像点转换为空间点，并整理成可直接写入 JSON 的普通列表。

    输入：一帧中某种视觉方法识别出的像素点。
    输出：固定结构的字典，供 camera.py 放进当前帧 result。

    实验作用：
    逐帧日志会同时保留像素坐标和空间坐标。这样排查时可以先看原始像素点，
    再看空间投影结果，避免把识别错误和标定错误混在一起。

    返回字段固定，方便后续检查：
    - enabled：是否开启空间坐标输出；
    - mode：使用的空间转换模式；
    - camera_m：相机坐标系三维点，未启用或无法计算时为 None；
    - robot_m：机器人坐标系三维点，仅 robot_plane 模式有值。
    """

    # 本段处理“空间坐标未启用”的正常情况。
    # 输出仍保持固定字段，让每帧日志结构稳定，同时明确告诉使用者当前没有做空间投影。
    if not config.SPATIAL_COORDINATES_ENABLED:
        return {
            "enabled": False,
            "mode": config.SPATIAL_COORDINATE_MODE,
            "camera_m": None,
            "robot_m": None,
        }

    # 本段执行空间转换主流程。
    # 输入是像素点；中间先变成相机射线；最后根据配置选择投到相机平面或机器人平面。
    points = _as_points(points_px)
    rays_camera = _normalized_rays(points)

    if config.SPATIAL_COORDINATE_MODE == "camera_plane":
        camera_points, robot_points = _camera_plane_points(rays_camera)
    elif config.SPATIAL_COORDINATE_MODE == "robot_plane":
        camera_points, robot_points = _robot_plane_points(rays_camera)
    else:
        raise RuntimeError(f"未知空间坐标模式：{config.SPATIAL_COORDINATE_MODE}")

    return {
        "enabled": True,
        "mode": config.SPATIAL_COORDINATE_MODE,
        "camera_m": camera_points.tolist(),
        "robot_m": None if robot_points is None else robot_points.tolist(),
    }
