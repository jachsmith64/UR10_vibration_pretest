# UR10 预实验工程 — 当前 `micro_closed_loop`

本包是 `UR10_vibration_pretest` 的源码交付包，供人工或其它模型复核本轮改动。
根目录扁平，全部源码在根下，单测在 `tests/`，**不含任何输出数据与图片**。

> 当前模式已经从旧的 5/20/50 μm `alt`/`seq` + 闭环结构改为
> **XY 多目标视觉闭环逼近实验**。本文后半部保留的 V1c 章节是历史实现记录，
> 不代表当前 `micro_closed_loop` 的主动运行路径。

## 当前模式概览

每个轴使用相邻目标步长 `+100, +150, +210, -100, -150, -210 μm`，
对应统一实验原点下的绝对位置：

```text
0 → 100 → 250 → 460 → 360 → 210 → 0 μm
```

X 完成后回到初始位姿附近，再执行 Y；本轮没有二维斜向目标。每个目标最多
12 次，控制律固定为 `command = error`，不加增益、PID 或学习补偿。正常目标状态为
`STABLE_REACHED`、`LIMIT_CYCLE`、`SMALL_COMMAND_STALL` 或 `MAX_ITER_REACHED`。

每次迭代在同一个活动 RAW 窗口内复制命令前最后 16 帧，计算当前误差后发送一次
MoveL 微动，再取稳定后的最后 16 个有效帧；不会把上一轮 `after` 沿用为下一轮
`before`。视觉 x/y 通过探针完整 2×2 矩阵的逆变换统一到机器人 X/Y 坐标。

原始 RAW 写入 `D:/UR10_micro_raw` 临时区。每次迭代的 CSV 行和必要证据保存后，
该次 RAW 立即删除；正常结束、用户停止和异常退出还有兜底清理。保留统一
`iterations.csv`、目标/最终 JSON、三张汇总 PNG 与必要证据帧。

---

## 1. 这个实验要回答什么

1. 多目标闭环最终能稳定逼近到多少微米；
2. 修正命令降到个位数/十几微米后，何时不响应、过冲或来回跳动；
3. X/Y 与正负方向是否有明显差异；
4. 当前系统对后续几十微米级抗振纠偏是否具备实际能力。

最终报告只给“经验上能产生明确同向响应的命令量级”和最终误差分布；数据不足时明确写
“当前数据不能确定单一最小运动阈值”，不会输出假的严格最小运动单元。

---

## 历史 V1c 实现记录（以下不再是当前主动路径）

开环与闭环测的是两件不同的事，字段与判读严格分开：

- **开环** = "给一个固定命令，实际走了多少" → 这才是执行能力的来源；
- **闭环** = "能不能用反复纠偏逼近一个目标" → 这是逼近能力，**最终残差不是最小微动**。

上一版把闭环残差当最小微动报出去，是概念错误。

---

## 2. 一次运行会做什么（全程无需人工干预）

用户要求的执行形态：**测量运动前位置 → 发一次微动 → 等短时间稳定 → 测量运动后
位置 → 立即算实际位移 → 记录 → 再走下一步**。**绝不是**"先把几十个动作跑完、
录一个大视频、最后统一处理"——那样既拿不到"运动前位置"，也无法在异常时及时停下。

```
A. UR10 通信检查
B. 相机检查与预热
C. ★ 轴向/符号探针（实测视觉↔机器人的对应关系、正负号与增益）
D. 静止基线 5 s（完整 RAW 永久保留）
E. X 开环连续微动   5 → 20 → 50 μm
F. X 开环来回微动   5 → 20 → 50 μm
G. Y 开环连续微动   5 → 20 → 50 μm
H. Y 开环来回微动   5 → 20 → 50 μm
I. X 视觉闭环       5 → 20 → 50 μm（每目标 +/− 两个方向）
J. Y 视觉闭环       5 → 20 → 50 μm
K. 汇总 final_report.json / .txt 并安全结束
```

探针排在静止基线之前，是因为**探针失败是具名中止**：与其先花 5 s 录基线再发现
符号定不出来，不如把无法继续的那一步提到最前面。

### 开环块：12 块 × 6 步

| 模式 | 序列 | 看什么 |
|---|---|---|
| `seq` 连续同向 | `+Δ +Δ +Δ −Δ −Δ −Δ` | 微小命令是否真的产生运动；连续同向能否累积；正负是否对称；有没有"前两步不动、第三步突然跳很大"的死区/积累现象 |
| `alt` 频繁换向 | `+Δ −Δ +Δ −Δ +Δ −Δ` | 换向死区、回差、摩擦与方向延迟；是否比连续同向明显更差 |

两种模式**净位移都 ≈0**，所以块与块之间不需要回位动作，也不会随块数累积走位。
正因如此，整个开环阶段**只建一次视觉零点**（链式相减在任何一步都成立）。

每个开环块有独立的 `block_id`（`X_open_seq_005um` … `Y_open_alt_050um`），
逐步结果写进该块的 `open_steps.csv`，**每走一步就落盘一次**；块结束再写
`block_summary.json`，并刷新一次 `open_loop_summary.json`。中途因异常收尾时，
已经跑完的块必须已经落盘，而不是等最后统一写。

### 闭环

每个目标：目标 → 测当前误差 → 发修正 → 重新测量 → 继续逼近，
直到下面之一：

| 状态 | 含义 |
|---|---|
| `CONVERGED` | 连续 3 次误差落在有效容差内，**且通过了方向性运动确认** |
| `CONVERGED_NOISE_LIMITED` | 收敛了，但残差与零在统计上不可分（信噪比 < 2） |
| **`MEASUREMENT_NOISE_LIMITED`** | **误差落在容差里，但没观察到朝目标方向的运动——这是"假成功"必须分开报** |
| `LIMIT_CYCLE` | 真极限环：反复换号，且峰峰值与中位幅值都不再收缩 |
| `STALLED_MIN_EFFECTIVE_MOTION` | **单边不收缩的平台**——控制器忽略了小于阈值的命令 |
| `DIVERGING` | 近期中位幅值 > 前窗 1.2 倍 |
| `MAX_ITER` / `MEASUREMENT_LOST` / `ABORTED` | 到顶 / 测不到 / 联锁中止 |

**每个目标最多 6 次迭代**（`MICRO_LOOP_MAX_ITER = 6`，落在用户要求的 5～8 内），
不做几十次反复磨。连续 2～3 次误差没有明显改善就判 `STALLED`；正负误差持续来回跳
判 `LIMIT_CYCLE`。**不为了强行达到 5 μm 结果而无限运行。**

### `MEASUREMENT_NOISE_LIMITED`：5 μm 档最危险的错误

不是"没收敛"，而是**完全没动却判成功**。当有效容差已经和幅值同量级
（`tol_eff ≥ A`，5 μm 档很可能如此）时，"误差落在容差内"几乎不构成证据——
机器人一步不走，误差恰好就是幅值本身，也满足 `|error| ≤ tol_eff`。

所以 `CONVERGED` 还必须额外通过 `directional_motion_confirmed()`：

- 取**净位移**（不是每步位移的绝对值之和——那个不区分方向，左走 10 μm 再右走
  10 μm 会被算成"走了 20 μm"）；
- 按 `direction` 归一（**目标在 −50 μm 时机器人合法地向 −X 走**，未归一化会把
  合法位移判成"方向错误"——这正是 sign=−1 最容易写错的地方）；
- 门槛 = `max(幅值的 50%, 本组噪声底 σ)`。抬到 σ 之上是因为比一个 σ 还小的"位移"
  与没动在统计上不可分。

不通过则报 `MEASUREMENT_NOISE_LIMITED`。那既不是假成功也不是失败，
而是"**这个幅值下视觉分不出来**"这一条真实结论。

### 收敛后的零命令复测

判 `CONVERGED` 后不发任何命令，再录一段静止片段比对：`|复测 − 残差| ≤ 2σ`
才算真不动点，否则重开闭环（最多 3 次，**不重置迭代预算**，所以不会拖长总时长）。
`converged_verified` 列写进汇总。

### 运行期联锁（防闭环发散）

- 连续 2 次指令与实测方向相反 → 中止（`SIGN_INVERTED`）
- 本组累计指令量超过 2000 μm → 中止（`WALKED_AWAY`）
- **每次修正后**跑一次全局漂移守卫 → 越限中止
- 连续 2 轮测量丢失 → 中止该目标

`WALKED_AWAY` 与漂移守卫是必要的：`robot.validate_trajectory` 的 ±0.20 m
相对工作区在**每次调用时都以当时位姿重新锚定**（`robot.py:215-220`），
结构上无法察觉缓慢的棘轮式走位。

漂移守卫的基准是**一条轴从头到尾的起点**，而不是每个目标各自的起点，也不按幅值
给允许量。**这是修掉的一个真实缺陷**：按幅值给的话，−A 目标的允许量只有 A，
而机器人合法地走到 −A 就已经用满了额度，于是**每一个负方向目标都会在第一次迭代
就被判"疑似棘轮式走位"而中止**。正确的不变量是"seq/alt 与闭环都净位移≈0"，
所以用轴向全局限值（警戒 300 μm / 中止 1500 μm），与瞬时幅值无关。
开环与闭环**共用同一份实现**（`axis_drift_note`）——两份拷贝迟早漂移成两个判据。

---

## 3. 时间与数据量（按实际代码算出来的，不是估的）

| | 正常 | 设计最坏 |
|---|---|---|
| 段数 | 144 | 168 |
| 时长 | **11.3 分钟** | 20.3 分钟 |
| 原始数据 | 48.5 GiB | 154.0 GiB |

闭环目标 **12 个**（2 轴 × 3 档 × 正负 2 方向），开环 **12 块 × 6 步 = 72 步**。
各部分的正常耗时：`static` 17.2 s ｜ `probe` 36.8 s ｜ 6 个组参考 65.0 s ｜
**`open_loop` 293.5 s（占 44%）** ｜ `closed_loop` 195.7 s ｜ 闭环验证 48.9 s ｜
启动开销 18.0 s。

### 时间耗在哪里

单步的机械等待只有约 0.4～0.6 s，**分析帧才是大头**：识别一帧全幅 1936×1464
实测 **163～168 ms**，本轮每次测量分析尾部 **16 帧**（不确定度 ≈ 0.32 μm），
于是一段窗口的在线分析约 2.6 s，机械等待只占约 4%。

所以在线只分析**窗口末尾**的 16 帧：闭环要的是运动停稳之后的位置，它就在窗口末尾；
窗口照录不误，前面的瞬态帧留在保留的 RAW 里供事后离线复查。

**11.3 分钟**落在操作者本轮明确接受的 10～12 分钟区间内。
已经逐项核对过没有可省而不损实验的地方：没有固定长睡眠（等待由
`_wait_until_position_stable` 提前放行）、没有重复扫同一段 RAW
（`prime()` 每组只跑一次，双扫是 medoid 锁参考所必需）、没有任何
FFT/PSD/频谱计算、组间只有一次 `RETURN_START` + `STABILIZE`。

### 设计最坏 20.3 分钟是什么，不是什么

它是一个**条件上界**，不是预报：要每一段窗口都同时吃满三个超时
（运动轮询 1.0 s + 停稳等待 0.8 s + 降级等待 0.5 s）才会到。真实机器若真这样，
那一步的数据本来也不可用，而且此时会先撞上测量丢失/漂移/方向联锁。

**它不再是截止时间。** 上一版在这里有一个"硬上限"，运行时外推总时长、
超预算就抛 `TimeBudgetExceeded` 收尾。**该策略已按操作者要求删除**（见 §13）：
机器人、相机、磁盘都正常时，不能因为"到 10 分钟了"就让 X 跑完之后 Y 的
闭环一个都不跑。

现在只剩下一条 `MICRO_LOOP_TIME_NOTICE_S = 900 s`（15 分钟）的**提示线**：
运行中投影超过它只打一条日志，操作者能看见"今天比平时慢"，
但**不会跳过任何目标**。取值 15 分钟而不是 10 分钟，是因为正常运行本来
就要 11.3 分钟——取 10 分钟会让每一轮都触发提示，提示就成了噪声。

### 时长不再中止实验，那什么才会中止

只有真实故障，以及单个目标自己的收尾：

| 类别 | 具体项 |
|---|---|
| 整轮中止 | UR10 通信异常、相机异常、磁盘不足、内存不足、超出工作空间、异常累计漂移、方向异常、操作者点"停止当前任务" |
| **只结束该目标** | 该目标的 `MAX_ITER` / `STALLED_MIN_EFFECTIVE_MOTION` / `LIMIT_CYCLE` / `DIVERGING` / `MEASUREMENT_LOST` / `ABORTED` |

第二类是**每目标独立**的：某一个目标跑满迭代或判停滞，只落盘该目标的结果，
然后继续下一个方向、下一档、下一条轴。12 个闭环目标全部跑完才算结束。

---

## 4. 输出目录

```
outputs/micro_motion_<YYYYmmdd_HHMMSS>/
├─ config.json              本次运行的全部生效参数
├─ final_report.json        核心问答（GPT 端直接读这个）
├─ final_report.txt         同样内容的纯文本版
├─ master_summary.csv       每个闭环目标一行
├─ open_loop_summary.json   12 个开环块的判读 + 最小可靠档位
├─ probe_gains.json         探针结果
├─ experiment_log.txt
├─ static/{static_frames.csv, static_summary.json, evidence/*.png}
├─ X_open_seq_005um/{open_steps.csv, block_summary.json}
├─ X_open_alt_005um/ … Y_open_alt_050um/         （共 12 个开环块）
└─ X_closed_005um/ … Y_closed_050um/             （每个闭环组）

D:/UR10_micro_raw/<run_id>/                        ← 运行期间的原始 RAW
├─ 00_static/
├─ 01_probe/
├─ 02_open_baseline/
├─ X_open_seq_005um/ … Y_closed_050um/
```

**工程目录中不存在任何完整视频**（没有 `.mkv` / `.avi` / `.raw`）。
临时 RAW 位于本次 D: 运行目录的 `_micro_temp`，用完即归档到对应实验块。
任务收尾时统一删除所有大体积视频载荷，只留下旁车时间戳、分析 CSV/JSON 和证据图。
启动时仍按**最坏**数据量检查磁盘并预留 15 GiB 余量，因为自动删除发生在收尾阶段，
不能掩盖单次完整实验本身可能写满磁盘的问题。

### `final_report.json` 回答什么

```json
{
  "static": {"static_sigma_um": …, "static_rms_um": …, "static_peak_to_peak_um": …},
  "open_loop": {"X": {"005um": {"seq": {…}, "alt": {…}}, …}},
  "closed_loop": {"X": {"005um": {"positive": {…}, "negative": {…}}, …}},
  "answers": {
    "X_5um": {"open_seq": "无法可靠动作", "open_alt": "无法可靠动作",
              "open_overall": "无法可靠动作",
              "closed_termination_positive": "MEASUREMENT_NOISE_LIMITED",
              "has_stalled": false, "has_limit_cycle": false, "has_noise_limited": true},
    …
  },
  "min_reliable_level_um": {"X_seq": 20.0, "X_alt": null, …},
  "not_run_targets": []
}
```

`min_reliable_level_um` 只能是"**已测的三个档位里最小且稳定的那一档**"，
推不出精确最小分辨率——要那个数就得扫更多档位，正是本轮刻意不做的。

### 关键交付列

`open_steps.csv`（开环每步一行）——用户第 12 条要求的字段全在这里：

```
block_id, axis, mode, target_level_um, step_index, direction,
commanded_increment_um,
vision_axis, vision_before_um, vision_after_um, measured_increment_um,
robot_tcp_before, robot_tcp_after, robot_achieved_um,
command_timestamp_ns, measurement_timestamp_ns,
settling_time_s, settle_ok, status, note
```

`*_iterations.csv`（闭环每轮一行）：

```
iteration_id, axis, direction, target_amplitude_um, target_um,
measured_position_um,      ← 视觉侧：停稳后测到的位置
error_before_um, command_um,
achieved_robot_um,         ← 机器人侧：编码器实测位移
actual_visual_displacement_um,
error_after_um, robot_tcp_before, robot_tcp_after, transverse_um,
g_obs,                     ← 有效环路增益 = Δ实测 / Δ指令
n_valid_frames, n_tail_frames, sigma_tail_um,
timestamp, step_status, termination_state, note
```

`achieved_robot_um` 与 `actual_visual_displacement_um` 并排，直接给出三分表：

| 机器人动了吗 | 视觉看到了吗 | 结论 |
|---|---|---|
| 没动 | 没看到 | 控制器忽略了命令 → **最小有效运动尺度** |
| 动了 | 没看到 | 机械柔性 / 夹具间隙吸收，或视觉噪声底 |
| 动了 | 看到但量不对 | 增益误差 / 交叉耦合 |

`master_summary.csv` 的列里有两列**本轮改过名**，因为旧名字是错的：

- `closed_loop_final_residual_um` = `target_um − final_measured_um`。
  上一版写的是 `abs(final_measured)`，那是**位置**而不是误差：目标在 −50 μm 时
  它报 50 μm 的"误差"，在 +50 μm 收敛良好时也报 50——整列数全是错的，
  而且错得很有规律（正好等于幅值），最容易被当成"没收敛"。
- `final_position_um` = 最终位置本身。残留说"离目标多远"，位置说"人到底在哪"。

---

## 5. 视觉链路：只改一处，且是结构性修复

### 视觉轴与符号的映射只有**一个**入口

```python
GroupVisionMeter.axis_measured_um(measurement, robot_axis, gains) -> float | None
```

它一次做完三件事：按探针结果**选视觉轴**、按探针结果**归一符号**、
测量不可用时返回 `None`（而不是 0——返回 0 会被上层当成"测到了位置 0"，
从而算出一个假的位移）。测量对象自己记着 `vision_axis`，与探针不一致时**硬抛
`ValueError`**：这不是容错场景，拿到的数看起来完全合理，但它描述的是另一条轴。

**为什么必须封成一处**：选轴（robot Y 走的是视觉 y 还是 x）与符号（视觉正向是否
等于机器人正向）是两个**必须一起用**的映射。上一版把"选轴"留给调用点、
测量层硬编码读了视觉 x，于是 **robot Y 的六个目标全部测的是视觉 x 的漂移**；
而符号没归一，让 sign=−1 的正常轴在闭环联锁里被判成"方向错误"直接中止。
两处各自抄一遍映射，迟早会漂移。

### 符号只转换一次：控制律**不接** sign

上一条把这个入口封好之后，符号就在 `axis_measured_um` / `axis_positions_um`
里用掉了，调用方拿到的 `measured` 已经是**机器人测试轴坐标系**。于是：

```
视觉位置 --(×sign, 唯一一次)--> measured(机器人轴系)
target(机器人轴系) − measured  =  error
command = Kp · error                      ← 这里**不再乘 sign**
```

**上一版在这里出过一个真实缺陷**：`command_for_error` 带了 `sign` 参数，
调用点传了 `gains.sign[axis]`，于是符号被用了两次。在 sign=−1 的相机安装
（机器人 +Y 在图像上是 −y）下，命令被重新翻反，误差按 `err_{k+1} = 2·err_k`
翻倍：50 → 100 → 200 → 300 → 400 → 500 μm，6 轮内机器人跑到 −550 μm 之外。

**它不会被方向联锁抓到。** 双符号下指令被翻反，视觉读数也被同一个翻反的
sign 归一回来，两者恒同号，`g_obs ≡ +1` —— 所以症状是 `DIVERGING`
（外加轴向漂移告警），看起来像"这台 UR10 连 50 μm 都逼近不了"，
而不像代码写错了。而且 sign=+1 时它完全不可见（乘 1 是恒等），
只在装反了相机的那台机器上发作。

**修法不是加判断，是把参数删掉**：`command_for_error` 现在不接受 `sign`，
"再乘一次"在类型层面就写不出来。对应回归用例
`test_the_command_law_does_not_take_a_sign`（断言传 sign 会 `TypeError`）
与 `test_double_applied_sign_would_diverge_which_is_why_it_is_gone`
（把旧律逐轮跑一遍，钉住它真实的后果是 `DIVERGING`）。

整条链路现在只有一处符号转换，逐段接起来测在 `SignChainTests`：
sign=+1 / −1 × 正/负目标 × 机器人已在目标正侧/负侧，断言一律写成
"指令方向 = (target − measured) 的方向"。

### 只用公开 API，复用已验证的亚像素识别流程

`GroupVisionMeter` 每帧走的是：`gray = frame`（Mono8 已是灰度）
→ `cv2.GaussianBlur(gray, (3,3), 0)` → `cv2.setRNGSeed(0)` → `tracker.process(gray)`。

- 与 `camera.preprocess_frame` 的**识别部分逐字等价**，只是不算那份显示用的画质统计
  （`MICRO_LOOP_COMPUTE_METRICS = False`，用户定"图像质量只显示、绝不参与控制"）。
- `setRNGSeed(0)` 的理由：`_estimate_rigid_motion` 用
  `estimateAffinePartial2D(method=RANSAC)`，内点集抖动会让 88 点均值位移跳动
  约 0.8 μm。固定种子让同一帧永远得到同一结果。
- **没有重新设计棋盘格识别算法**，`VISION_METHOD == "checkerboard"`、
  `CHECKERBOARD_INNER_CORNERS = (11, 8)`。

参考帧用 **medoid** 而不是"第一帧"：单帧误检若成为永久零点，整组报废。
每帧接受判据六项：`checker_is_valid` / `quality ≥ 0.5` /
`found_by == 参考帧的 found_by` / `|angle| ≤ 1°` / `|scale−1| ≤ 0.01` /
`位移 ≤ 2000 μm`。`found_by` 相等这一条是廉价且关键的一刀：
`findChessboardCornersSB` 与传统兜底 `findChessboardCorners` **不保证角点排序一致**。

连续性联锁比较的是同一机器人轴坐标系：当前视觉坐标先乘探针测得的 `sign`，再与
上一轮已归一的位置比较。不能把 `sign=-1` 时的视觉原坐标与机器人轴坐标直接相减；
否则累计位移约 150 μm 后会产生超过 300 μm 的假跳变并把正常帧全部拒绝。

为什么需要这些门禁：棋盘格**索引错位一格**会让所有点位移同一个向量（约 3 mm），
`estimateAffinePartial2D` 会拟合出一个**残差≈0、内点率 1.0、quality≈1.0、
angle≈0、scale≈1 的纯平移**——与"真的移动了 3 mm"在当前返回字典里无法区分。

### 静止基线为什么抽帧而不只看尾部

静止基线要的是"整段 5 s 里画面抖了多少"。只看尾部等于把前面 4.7 s 的记录扔掉，
噪声底会被系统性低估。所以用 `spanning_every_n()` 把 40 个分析帧**跨满整段**铺开；
运行期间被抽掉的帧仍在 RAW 里；收尾删除 RAW 后保留抽样帧的完整接受/拒绝记录。

### 为什么用未压缩 RAW，而不是 MJPG / FFV1

**绝对不退回 MJPG**（用户硬约束）。FFV1 也不行：OpenCV 的
`VideoWriter_fourcc('FFV1')` 写 Mono8 时会经 swscale 转成 `YUV420P`
（色度抽样 + 有限范围 16–235），**并非位精确**。本实验要的分辨率是
3 μm ≈ 0.044 px，这种系统性量化是实打实的损失，抽检帧比对查不出它。

**内存缓冲 → 未压缩 RAW 临时文件 → 处理 → 归档**。丢帧由架构隔离，
不靠压缩解决：全幅实时写盘每帧均值 2.27 ms（预算 7.56 ms）会丢 6.11%；
memcpy 进内存约 0.3 ms，零丢帧、余量 25×。

**不裁剪、不 resize**：实测同一批 20 帧，冻结一个以棋盘格为中心的原分辨率裁剪框
（720×722）后识别成功率从 20/20 掉到 7/20，耗时只快 19%。原因是
`CALIB_CB_NORMALIZE_IMAGE` 的归一化基准是整幅图，裁掉背景后对比度基准改变、
检测阈值随之漂移。**保持全幅 1936×1464。**

---

## 6. 命令接口（单点接缝）

```python
def send_cartesian_micro_correction(robot, axis, delta_um, *, safety_center_pose,
                                    stop_event, write_state, settle_timeout_s=None) -> dict
```

- **μm → m 只在这一处转换**（`delta_um * 1e-6`），别处禁止混用单位
- 构造 **2 点** `Waypoint`（`execute_trajectory` 对 ≤1 点轨迹**静默 no-op**）
- `validate_trajectory` → `verify_controller_safety_limits` → `execute_trajectory`
- **断言命令真的发下去了**：`execute_trajectory` 在成功与静默 no-op 两条路径上
  都返回 `None`，只能靠 `robot.motion_command_time` 是否推进来区分。
- 完成判据**不用** `motion_in_progress()`（带 200 ms 固定宽限，对 14 ms 量级的
  微动毫无信息量），改用基于位置与速度的稳定判据；超时**降级为记录**，
  由视觉片段自身的尾部标准差决定这次测量是否有效。

### 等待时间按"通常快、必要时才等更久"设计

```
前段稳定 ~0.15 s → 发指令 → 等位置稳定（0.8 s 内稳了就立刻走，最长 1.0 s）
→ 后段 ~0.25 s → 下一段
```

`MICRO_LOOP_STEP_TIMEOUT_S = 1.0` 而不是随手放大：100 μm 在 1 mm/s 下
约 0.35 s（含加减速），1.0 s 已是约 3 倍余量；而这个值同时是"最坏窗口"的
主要构成项，最坏窗口 × 段数决定了整轮的绝对上界。

**探针的超时必须与微动分开**：探针走 1000 μm，是微动步的 20～200 倍，
在 1 mm/s 下就要约 1.2 s，沿用 1.0 s 会让它**每次都超时**并退化成"靠固定
settle 猜"——而探针失败是具名中止。所以另设
`MICRO_LOOP_PROBE_TIMEOUT_S = 3.0`。两者的移动量差 20～200 倍，
共用一个超时值必然错一头。

### 死区 `MIN_COMMAND_UM = 2.0` 是必需的，不是调参

`robot.validate_trajectory` 拒绝 `length <= 1e-6 m`（"两点位置重合"）。而
`command = Kp · error`，收敛时 `|error|` 可能只有 1 μm → 命令 1 μm → 正好撞在
这条硬线上，**会在一个目标即将成功的时刻把整轮弄死**。所以
`|command| < 2 μm` 时不发运动，但仍照常录前后段、照常测量、照常判收敛。

---

## 7. 轴向 / 符号探针（必做）

视觉 +x/+y 与机器人 base X/Y 的对应关系（含符号）**无法从代码推出**，且符号反了
会让误差每步翻倍直冲 100 μm 限幅。所以启动时用一次 1000 μm 的探针测出来。

每次探针运动后的视觉测量若不合格，会保持当前位置、重新等待稳定并进行最多
**5 次无运动复测**；任一次复测合格即继续，初测与 5 次复测全部不合格才具名中止。
复测不降低“尾窗至少 12 帧、标准差 ≤ 2.0 μm”等原有门限。探针专用候选池为
末尾 64 帧 / 0.50 s，参考建立使用 32 帧；正式闭环仍为原来的 16 帧 / 0.20 s。

### 固定 20 cm 软件安全包络

`micro_closed_loop` 在 RTDE 连接并确认静止后，把当时的 TCP XYZ 固定为本次运行的
安全中心。每次运动前同时检查当前点与目标点，二者到该中心的三维欧氏距离均不得
超过 0.20 m；球体是凸集，因此两点间的直线 MoveL 也不会穿出球体。运动期间以
125 Hz 读取的实际 TCP 继续检查，空闲时也以 5 Hz 检查；一旦越界立即调用受控
`stopL` 并使实验失败。返回初始位姿同样执行此检查，中心不会随每一步重新锚定。

这是应用层的纵深保护，不能替代 UR 控制柜内的安全平面、减速模式、急停和现场
风险评估；这些控制器级措施不受 Python 进程卡死的影响，仍必须按现场配置启用。

增益门禁（任一不过则**具名中止**，不继续）：两个机器人轴必须映射到**不同**的视觉轴
（否则像平面近似侧视）；增益必须落在 `[0.3, 2.0]`；正反向增益差 < 20%；
交叉耦合 ≥ 0.3 时**记录并提示**但不中止。

**不做 `/g` 自标定**——不做自适应增益控制，命令律严格保持 `Kp · error`。

---

## 8. 本轮改动的文件

| 文件 | 改动 |
|---|---|
| `config.py` | 收紧等待与超时；新增探针超时、时间预算、方向性确认门槛；`MICRO_LOOP_AMPLITUDES_UM` 固定为 `(5, 20, 50)` |
| `micro_closed_loop.py` | 新增开环序列/判步/统计、块命名、`spanning_every_n`、`directional_motion_confirmed`、`MEASUREMENT_NOISE_LIMITED`、重写 `estimate_run_plan` |
| `main.py` | 开环阶段 `run_open_loop_phase` / `run_open_loop_block`；时间预算准入控制 `TimeBudget`；RAW 归档 `release()`；共用漂移守卫；`write_final_report` |
| `launcher_ui.py` | 按钮改为「**XY微动 + 视觉闭环快速测试**」；档位改为固定显示（**删除幅值输入框**）；确认弹窗给出完整流程与时间/数据量；使用说明第 8 条重写 |
| `tests/test_micro_closed_loop.py` | 新增 34 个用例（共 **174** 个） |
| `README.md` | 本文件 |

**没有修改 `robot.py` 与 `camera.py`**，也没有改动任何其它实验模式的逻辑。
也没有加入 APF / SFC / 轨迹规划 / MPC / PID。

### 有意绕过手电筒门禁

`batch_camera_worker` 的 `_FlashGate` 会硬抛"手电筒门禁未完成"。新的相机 worker
**不构造它**——闭环的时间对齐由全机共享的 `time.perf_counter_ns()`
（Windows `QueryPerformanceCounter`，跨进程一致）保证，不依赖操作者打手电筒。
**这是设计决定，不是安全回归**，在模块 docstring 里也写明了。

### 删除的东西

`micro_motion.py`、`micro_motion_export.py`、`tests/test_micro_motion_plan.py`，
以及 UI 里的幅值输入框与 `--micro-levels` 参数。**11 档扫描不再存在**：
配置里、UI 里、README 里、运行时间与数据量提示里都不会再出现它，
也不会再出现"2.7 小时 / 5 小时"这类数字。

---

## 9. 怎么跑

```
python launcher_ui.py        （或双击 run_launcher_ui.cmd）
→ A. 硬件检查 → 连接成功后才解锁运动按钮
→ E. XY 微动 + 视觉闭环快速测试 →「XY微动 + 视觉闭环快速测试」→ 二次确认
```

命令行等价形式：

```
python main.py --mode micro_closed_loop --ui-confirmed
```

### 机械臂完全断电的静止对照

在启动面板 A 区点击“机械臂断电静止测试”，或运行：

```
python main.py --mode offline_static_test
```

该模式只连接工业相机，绝不创建 RTDE、Dashboard 或机器人控制接口。默认连续做
3 次 × 5 s 全分辨率静止基线，每次使用与 `micro_closed_loop` 相同的棋盘格链路和
40 帧跨整段分析，结果落盘后立即删除该次 RAW。输出目录名为
`outputs/offline_static_时间戳/`，其中 `offline_static_summary.json` 是断电重复统计，
`comparison_with_latest_powered.json` 会自动与最近一次通电静止基线比较。

断电结果并不等于“纯算法误差”：它仍包含相机支架、桌面、地面和环境振动。只有在
相机、棋盘格、镜头、曝光和光照完全不变时，通电与断电结果的差值才有归因意义。

**前置条件**（任一不满足即在创建任何目录之前被拒）：
`ROBOT_RELATIVE_MOTION_ENABLED = True`、`CONTROL_MODE != "sfc"`、
`VISION_METHOD == "checkerboard"`、D: 剩余 ≥ 最大单组数据量 + 15 GiB、
可用物理内存 ≥ 1.5 GiB。

日志区只接收子进程 stdout 文本，`[状态]` 前缀的行会镜像到状态栏，
所以每步打一行（不是每帧）：

```
[状态] UR10 就绪，初始 TCP 位姿 (…, …, …) m
[状态] 工业相机就绪：1936×1464 ｜ 132.230 fps ｜ 缓冲可容纳 802 帧
[状态] 探针结果：机器人 X → 视觉 x ｜ 增益 +0.97 ｜ 符号 +1 ｜ 正向/反向差 3%
[状态] 静止噪声底：sigma 0.61 μm ｜ RMS 0.74 μm ｜ 完整 RAW 将在本次任务收尾后自动删除
[状态] 开环零点已建立：medoid 帧 41 ｜ 合格帧比例 99.2% ｜ 测量不确定度 0.32 μm
[状态]   X_open_seq_005um 步 1/6 ｜ 命令 +5.0 μm ｜ 实测 +0.41 μm ｜ 状态 ZERO_RESPONSE
[状态] 块结束 X_open_seq_005um：NO_RESPONSE ｜ 有效 0/6 ｜ 零响应 5 ｜ 部分 1 ｜ …
[状态]   X_closed_020um 迭代 2/6 ｜ 位置 +0.30 μm ｜ 误差 +19.7 μm ｜ 指令 −19.7 μm
[状态] 全部实验完成：闭环目标 12/12 个
[状态] 本次任务的大体积 RAW/视频文件已自动删除，CSV/JSON/证据图保留
```

---

## 10. 验证状态

| 项目 | 结果 |
|---|---|
| `python -m unittest discover -s tests` | **221 个用例全绿** |
| `ast.parse`（4 个改动文件） | 通过 |
| `main.py --help` | 出现 `micro_closed_loop`，`micro_motion_experiment` 已消失 |
| 无硬件干跑 | 4 个入口门禁逐一被拒，且**目录未被创建** |
| 探针门禁 | 6 种坏增益矩阵逐一具名中止（含"不可分辨"与"回差过大"） |
| 终止判据 | 全分支覆盖，**含两个必需反例**（早期单调收缩不得误判极限环；噪声底以下的换号不得判极限环） |
| 用户第 15 条验收清单 | **逐条对应用例**（见下）；本轮新增 21 个覆盖 sign / 目标数 / 时长策略 |
| 实机 | **尚未进行**（需要操作者在场） |

### 用户第 15 条验收清单 ↔ 用例

| # | 要求 | 用例 |
|---|---|---|
| 1 | 正式 level 只有 `[5, 20, 50]` | `test_formal_levels_are_only_5_20_50` |
| 2 | 顺序确实 5 → 20 → 50 | `test_level_order_is_ascending_and_fixed` |
| 3 | `seq` 生成 `+++−−−` | `test_seq_mode_is_three_positive_then_three_negative` |
| 4 | `alt` 生成 `+−+−+−` | `test_alt_mode_is_strict_alternation` |
| 5 | X/Y 都能执行两种模式 | `test_both_axes_run_both_modes_at_all_three_levels` |
| 6 | 闭环每目标最多 5～8 次迭代 | `test_closed_loop_iteration_cap_is_between_five_and_eight` |
| 7 | 5 μm 完全不动时不能判 `CONVERGED` | `test_a_target_that_never_moves_is_never_confirmed_as_converged` |
| 8 | Y 轴仍正确读取 probe 后的视觉轴 | `VisionAxisMappingTests`（13 个用例，含轴互换与「一次测量供两条轴」） |
| 9 | `sign=−1` 时不会误判方向错误 | `SignChainTests`（7 个）、`SignFlipInterlockTests`（5 个）、`test_sign_minus_one_flips_the_reading` |
| 10 | static RAW 不会自动删除 | `test_static_baseline_raw_is_retained_by_policy` |
| 11 | 运行时间估算不再显示 2.7 小时 / 5 小时 | `test_run_plan_never_quotes_hours_for_a_normal_run`、`test_budget_lines_are_minutes_not_hours` |
| **12** | **闭环目标预算 = 12**（本轮） | `ClosedTargetCountTests`（4 个） |
| **13** | **不存在「10 分钟自动强制收尾」**（本轮） | `NoTimeTruncationTests`（5 个） |

> 单测请用 `.venv/Scripts/python.exe`。系统 `C:\Python314` 没装 `cv2`，
> 会导致 `test_batch_camera_startup` / `test_batch_offline_vision` 报
> `ModuleNotFoundError`——那是环境问题，不是回归。

### 实机首次运行请重点确认

1. 探针符号是否正确（这是闭环会不会发散的分水岭）
2. 静止基线的 `static_sigma_um`（决定 5 μm 档到底能不能分辨）
3. 开环 5 μm 块的 `zero_response_steps` 与 `partial_response_steps`
   ——"完全没动"与"动了但只走一半"是两种不同的物理现象
4. `g_obs` 是否接近 1
5. 结束是否回到初始位姿
6. 实际耗时与 `mean_window_s`（`final_report.json` 的 `timing` 段）——
   **这里没有截止线**：时长长只说明慢，不代表实验被截断
7. 工程目录里是否**没有**任何残留视频，D: 下的块是否齐全

---

## 11. 已知局限

1. **本轮只有三个档位**，所以只能说"这一档稳不稳"，
   不能给出最小分辨率的精确值。
2. **正常时长 11.3 分钟**，原因是每步 16 帧的在线测量（已确认的取舍）加上
   闭环目标从 6 个修正为 12 个。本轮已明确接受 10～12 分钟，
   且**不存在强制截止**：宁可跑完，也不为压时间砍掉目标或降低稳定性判据。
3. **`GAUSSIAN_BLUR_KERNEL = 3` 在检测前就抹掉了换取亚像素精度所需的高频。**
4. **`CAMERA_MATRIX = None`，未做畸变校正。** 镜头畸变会让刚体平移**不**产生
   均匀像素平移，给均值带来系统性**增益**偏差（不是噪声，平均消不掉）。

第 3、4 项都会在探针增益与 `est_sigma_um` 上留下痕迹。**若探针测出的 `|g|`
偏离几何预测 > 10%，先跑一次内参标定**（仓库里已有 `calibrate_camera_intrinsics.py`）
再解读结果。本说明不主张已排除这个偏差。

5. `MICRO_LOOP_KP` 固定为 1.0（用户指定）。`g_obs` 每轮都记录，
   因此有效环路增益可以从数据反推，但**运行期不做任何自适应**。

---

## 12. V1 → V1b 改动清单

`V1` 已能通过全部验收项，但复核时发现一个**会让 Y 轴开环阶段整段跑不起来**的
缺陷，另有一处判据可能被读错。本版修掉并补测。

### 12.1 缺陷：Y 轴在开环零点就会抛错中止（严重）

**症状**：开环阶段在第一块 `X_open_seq_005um` 之后进入 Y 轴时，
`axis_measured_um()` 抛 `ValueError` 并中止整轮实验。

**根因**：开环零点需要在**一段**基线片段里同时拿到 X 与 Y 两个锚点。
`V1` 的实现对同一段片段调用两次 `axis_measured_um()`，而这个入口带一条硬校验——
「本次测量所用的视觉轴必须与探针给该机器人轴的映射一致」。Y 轴读到的那次测量
是用 X 的视觉轴做的，于是被自己的校验拒掉。

**修法**：新增 `GroupVisionMeter.axis_positions_um(measurement, gains, *, robot_axes)`，
一次测量同时给出两条轴的位置：主轴取 `measured_um`，另一条取 `transverse_um`；
各自仍乘 `gains.sign[axis]`。并**再拦一次**「两条机器人轴映射到同一条视觉轴」
这种退化情形（探针门禁本该拦住，但静默返回张冠李戴的位置远比抛错危险）。

**为什么不重扫两遍片段**：`measure_clip()` 一次识别本来就同时算出了两条视觉轴的
尾窗中位数，再扫一遍只是把识别开销翻倍，而且开环是正常时长的 53%（§3）。

**局限（已写在接口 docstring 上）**：`transverse_um` 没有独立的 `n_tail` /
`sigma_tail`，尾部是否静止由主轴的门槛统一把关；两条轴用的帧集完全相同，所以
这个共享的门是合理的，但这个数**适合当锚点，不适合单独当成一次带不确定度的
测量去下结论**。开环每步的 `measured_increment_um` 因此是「相对锚点的差值」，
其不确定度由主轴那一步的 `n_tail` / `sigma_tail_um` 给出。

### 12.2 澄清：静止基线那处 `vision_axis` 默认为 `x` 是无害的

`analyze_static_clip()` 直接从逐帧行的 `checker_dx_mm` / `checker_dy_mm`
各算一份统计，**不经过** `measured_um` / `transverse_um`，因此与 `vision_axis`
无关。`V1` 里这处调用没传 `vision_axis`（吃默认值 `x`），行为正确；本版补了
显式传参与注释，避免下一个人把它误读成同一类缺陷。

### 12.3 补测：`axis_positions_um` 的 8 条用例

`tests/test_micro_closed_loop.py::VisionAxisMappingTests` 新增：

| 用例 | 覆盖 |
|---|---|
| `test_one_measurement_yields_both_axes_without_any_second_pass` | 一次测量给两条轴 |
| `test_both_axes_follow_the_swapped_probe_mapping` | 探针 X→视觉 y / Y→视觉 x 的互换 |
| `test_each_axis_gets_its_own_sign` | `sign=-1` 只翻转自己那条轴 |
| `test_two_axes_asking_for_one_vision_axis_is_a_loud_error` | 退化映射必须抛 `ValueError` |
| `test_a_missing_mapping_is_a_loud_error_here_too` | 缺映射必须抛 `KeyError` |
| `test_unusable_measurement_gives_none_for_every_axis` | 不可用测量给 `None` 而不是 `0.0` |
| `test_a_missing_transverse_axis_nulls_only_that_axis` | 一条轴缺数不牵连另一条 |
| `test_axis_subset_is_honoured` | `robot_axes=("Y",)` 只取该轴 |

### 12.4 验收状态

**182 个用例全部通过**（`tests/test_micro_closed_loop.py` 1388 → 1468 行）。
四个改动文件（`config.py` / `micro_closed_loop.py` / `main.py` / `launcher_ui.py`）
均通过 `ast.parse` 语法检查，未引入未定义名。

用户第 15 条的 11 项验收在 §10 逐条对应到具名用例，其中第 8、9 条
（"Y 轴仍正确读取探针后的视觉轴"、「`sign=-1` 时不会误判方向」）在本版由
上表 8 条新用例加厚——`V1` 只有 5 条，且**没有覆盖"一次测量供两条轴"这条路径**，
那正是 12.1 缺陷溜过去的原因。

---

## 13. V1b → V1c 改动清单

三处**定向修正**，不动实验规模、不动档位、不动闭环算法、
不动 A→K 流程、不动 RAW 保留策略。

### 13.1 符号只转换一次（修掉一个真实缺陷）

**GPT 的判断是对的**，而且比"可能"更严重：缺陷确实存在。

**真实数据流（修改前）**

```
视觉位置(μm)                 {"x": …, "y": …}
  └─ 探针 vision_axis + sign  ← 唯一合法的转换位置
      └─ axis_measured_um()   ← 已经乘过一次 sign，返回"机器人轴系"标量
          ├─ target_um = direction × amplitude     （机器人轴系）
          └─ error = target_um − measured          （机器人轴系）
              └─ command_for_error(error, sign=sign)   ← ★ 又乘一次
                  └─ command_um (μm) → send_cartesian_micro_correction
                      └─ robot.execute_trajectory(tcp + command·axis)
```

`sign = +1` 时乘 1 是恒等，**看不出任何异常**；`sign = −1`
（机器人在图像上装反了一个轴）时命令被重新翻反。

**实际后果（逐轮推演 + 模拟验证，不是推测）**

以 Kp=1、sign=−1、目标 +50 μm 为例，误差按 `err_{k+1} = 2·err_k` 翻倍：

```
误差 : 50 → 100 → 200 → 300 → 400 → 500 μm
指令 : −50 → −100 → −100 → −100 → −100 → −100 μm
位置 : −50 → −150 → −250 → −350 → −450 → −550 μm
终止 : DIVERGING
```

**它不会被方向联锁抓到。** 这一点值得写下来：`SIGN_INVERTED` 比较的是
"指令"与"实测位移"，而双符号下指令被翻反、视觉读数也被同一个翻反的 sign
归一回来，两者**恒同号**，`g_obs ≡ +1`。所以症状是"闭环发散、残差巨大"
（外加轴向漂移告警），读起来像"这台 UR10 连 50 μm 都逼近不了"，
而不是像代码写错了。

**修法**：`command_for_error()` 删除 `sign` 参数（不是加判断，是让"再乘一次"
在类型层面写不出来）；`main.py` 的调用点与 `run_target` 里那个已经无用的
`sign` 局部量一并删掉；`g_obs` 处的注释改正（那里本来就是对的，但理由写错了）。

**低层接口不需要 sign**：`send_cartesian_micro_correction(robot, axis, delta_um)`
收的就是机器人基座坐标下的有符号 μm，直接 `current[axis_index] + delta_m`，
**不存在第二套坐标约定**，所以"方案 A"（统一在视觉测量入口转换）成立，
不需要为它保留任何例外。

### 13.2 闭环目标预算 6 → 12

`estimate_run_plan()` 里写的是

```python
closed_targets = len(axes) * len(levels)        # = 6  ← 漏了方向这一维
```

正式闭环是 **2 轴 × 3 档 × 2 方向 = 12** 个目标
（X: +5,−5,+20,−20,+50,−50 然后 Y 同样）。改为：

```python
closed_targets = len(build_targets(axes=axes, amplitudes=levels))   # = 12
```

用 `build_targets()` 而不是自己再乘一遍，是为了让"实际会跑多少个目标"只有
一个定义处。

**受影响并已同步的量**：`closed_steps_max`（36 → 72）、正常/最坏段数
（114/126 → 144/168）、正常/最坏时长、正常/最坏 RAW 数据量、磁盘预检、
`format_budget_lines()` 的 UI 文案、README §3。

**没有改错的一项**：`group_reference_windows` 仍然是 `2 × 3 = 6`，
因为参考是按 (轴, 档) 建的，组内 +A/−A **共用**同一个零点——这正是能测到
反向死区/回差的前提。改为 12 反而是错的。
（用 `test_group_references_are_still_six_not_twelve` 钉住。）

### 13.3 删除"到 10 分钟强制收尾"

`TimeBudget`（deadline 准入控制）+ `TimeBudgetExceeded` + 两处
`except TimeBudgetExceeded` + `budget_stopped` 全部删除。

**替换为** `TimeWatch`：只记录每段实测耗时、按实测均值外推总时长，
超过 `MICRO_LOOP_TIME_NOTICE_S` 时**打一条日志**，永远不阻止任何一段窗口开录。

- `MICRO_LOOP_TIME_BUDGET_S`（585 s）→ `MICRO_LOOP_TIME_NOTICE_S`（900 s）
- `MICRO_LOOP_TIME_BUDGET_SOFT_S` **删除**：核对后发现它在上一版里
  只被写进 `plan` 字典、**没有任何代码消费它**，是一处死配置，
  注释却声称"超过它就不再开可选的收敛验证窗口"——名不副实，故删。
- `final_report.json` 的 `budget` 段 → `timing` 段，
  `stopped_early` **恒为 False** 并保留该字段，让读报告的人一眼看到
  "没有因为时间跳过任何目标"。
- 开环/闭环的 `try/except` 一并删除。**单个目标的 `MAX_ITER` / `STALLED` /
  `LIMIT_CYCLE` 本来就只是该目标的收尾**，此前就是这样，这回把它写清楚。

**仍然会中止整轮的真实故障**（未改动）：UR10 通信异常、相机异常、
磁盘不足、内存不足、超出工作空间、异常累计漂移、方向异常、操作者停止。

### 13.4 为上述三点补的 21 个用例（182 → 204）

| 用例类 | 个数 | 覆盖 |
|---|---|---|
| `SignChainTests` | 7 | 视觉→probe→measured→target→error→command 整条链路；sign=±1 × 正负目标 × 机器人已在目标正/负侧 |
| `SignFlipInterlockTests` | 5 | sign=−1 的正常映射**不得**触发方向异常；真反了的仍要抓到 |
| `ClosedTargetCountTests` | 4 | 目标数 = 12；预算与 `build_targets()` 一致；组参考仍是 6 |
| `NoTimeTruncationTests` | 5 | 计划里不再有截止相关的键；提示线存在；文案明说"不中止"；正常时长允许 11～12 分钟 |
| `CommandLawTests` 增补 | 2 | `command_for_error` 传 `sign` 必须 `TypeError`；把旧律逐轮跑一遍钉住它真实的后果是 `DIVERGING` |

其中 `test_the_command_law_does_not_take_a_sign` 是**回归钉子**：
上一版正是一条 `command_for_error(10.0, sign=-1.0) == -10.0` 的用例
把错误行为"证明"成了正确的，缺陷才活得下来。

### 13.5 时长策略改动后的实测预算

| | 修改前（V1b） | 修改后（V1c） |
|---|---|---|
| 闭环目标 | 6（少算一半） | **12** |
| 正常段数 | 114 | 144 |
| 正常时长 | 9.21 分钟 | **11.25 分钟** |
| 最坏段数 | 126 | 168 |
| 设计最坏时长 | 15.49 分钟 | **20.35 分钟** |
| 正常 RAW | 39.02 GiB | **48.45 GiB** |
| 最坏 RAW | 114.42 GiB | **154.00 GiB** |

11.25 分钟落在本轮明确接受的 10～12 分钟区间内，**不再为压到 10 分钟以内
删任何步骤**。设计最坏 20.35 分钟是"每一段窗口都吃满三个超时"的条件上界，
不是预报，且**不再触发任何截断**；最坏 RAW 154 GiB 仍低于 D: 的 277 GiB 余量
（预检按最坏值 + 15 GiB 预留，不满足则在创建目录前拒绝启动）。

时长偏长的原因已逐项核对，**不是**不必要等待：无固定长睡眠
（`_wait_until_position_stable` 稳定即放行）、无重复扫同一段 RAW
（`prime()` 每组一次，双扫是 medoid 锁参考所必需）、无 FFT/PSD/频谱计算、
组间只有一次 `RETURN_START` + `STABILIZE`。时长的两个真实来源是
**16 帧/次在线测量**（已确认的取舍）与**目标数从 6 修正为 12**。
