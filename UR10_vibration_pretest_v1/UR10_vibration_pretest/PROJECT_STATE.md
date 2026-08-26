# PROJECT_STATE

更新时间：2026-08-26

## 当前目标

项目应保持在“有高速录像采集过程、有 RAW 离线分析过程、分析结果带时间戳字段”的中间版本。

当前重点不是继续大改，而是先恢复可运行状态：高速采集要保留弹窗预览，离线分析要能读取 `frames.raw` 和 `frame_timestamps.csv`，并用 `analysis_time_s` 做后续分析时间轴。

## 已完成内容

- 已确认 `UR10_vibration_pretest.rar` 不是该中间版本；RAR 内没有 `vision_capture`、`vision_offline`、`CAPTURE_*`、`frame_timestamps` 或 `frames.raw` 相关代码。
- 已在当前代码中恢复 `vision_capture` 和 `vision_offline` 两个运行模式。
- 已恢复 RAW 采集目录读取能力：`RawCaptureSource` 按 `capture_index` 从 `frames.raw` 读取真实保存的帧。
- 已恢复 `frame_timestamps.csv` 字段读取/写入：`capture_index`、`frame_id`、`frame_id_gap`、`missing_before`、`host_ns`、`camera_timestamp_raw`、`frame_time_s`、`camera_time_s`、`analysis_time_s`、`analysis_time_source`。
- 已恢复 `missing_frames.csv` 生成。
- 已确认现有采集目录 `outputs/vision_capture_20260826_111814` 可读，第一帧映射为 `capture_index=0`、`frame_id=669`、`analysis_time_s=0.0`。
- 已通过静态检查：`python -m py_compile config.py camera.py analyze.py main.py launcher_ui.py`。

## 当前代码架构

- `config.py`：包含 `vision_capture`、`vision_offline` 配置和校验。
- `main.py`：包含 `run_vision_capture_mode()`、`run_vision_offline_mode()`，并在 `main()` 中分发。
- `launcher_ui.py`：启动面板包含“高速采集”和“离线识别”按钮。
- `camera.py`：包含 `RawCaptureSource`、`run_vision_capture()`、`run_vision_offline()`，并在 `VisionProcessor.process_frame()` 中透传采集时间戳字段。
- `analyze.py`：`extract_vision_series()` 优先使用有效的 `analysis_time_s`，不再要求 `host_ns` 同时有效。

## 已确认的设备参数

- `EXPECTED_VISION_FPS = 132.23`。
- `CAPTURE_DURATION_S = 10.0`。
- `CAPTURE_STORAGE_MODE = "ram_then_raw"`。
- `CAPTURE_SHOW_PREVIEW = True`，高速采集默认保留预览弹窗。
- `CAPTURE_PREVIEW_FPS = 20.0`。
- `VISION_SOURCE = "hik_camera"`。
- `HIK_EXPOSURE_US = 6500.0`。
- `HIK_FRAME_TIMEOUT_MS = 1000` ms。
- 现有样例采集目录 metadata 显示：`width=1936`、`height=1464`、`dtype=uint8`、`actual_camera_fps≈132.300`、`frame_count=1377`。

## 不允许改变的约束

- 不要关闭或移除高速采集/视觉测试中的原有弹窗预览行为，除非用户明确要求。
- 不要重构棋盘格识别、圆点识别、机器人通信、启动器整体架构或正式实验流程。
- 不要把 `capture_index` 和 `frame_id` 混用。
- RAW 文件只保存真实取到的帧，不为缺失帧生成空白图、零图、复制帧或虚假识别结果。
- raw capture 离线频谱时间轴不能使用 `host_ns` 作为首选来源；应优先使用 `analysis_time_s`。
- 后续继续改动前应保持小补丁、先静态检查，再由用户实机验收。

## 最近修改内容

- 恢复 `vision_capture`、`vision_offline`、`RawCaptureSource`。
- 恢复 `frame_timestamps.csv` 和 `missing_frames.csv` 生成/读取逻辑。
- 恢复启动器中的“高速采集”和“离线识别”按钮。
- 将高速采集默认预览设为开启。
- 将 `EXPECTED_VISION_FPS` 设置为 132.23。

## 下一步任务

1. 用户运行“高速采集”，确认弹窗预览存在，采集结束后输出 `frames.raw`、`frame_timestamps.csv`、`missing_frames.csv`、`capture_summary.txt`。
2. 用户运行“离线识别”，确认生成 `vision_results.txt`，记录数等于 RAW 实际保存帧数。
3. 检查 `vision_results.txt` 每条记录是否包含 `capture_index`、`frame_id_gap`、`missing_before`、`frame_time_s`、`analysis_time_s`、`analysis_time_source`。
4. 再运行 `analyze`，确认缺帧不会压缩后续 `analysis_time_s`。
