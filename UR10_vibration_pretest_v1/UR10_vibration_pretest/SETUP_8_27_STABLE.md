# 8/27 稳定版换电脑拉取与环境搭建

本项目当前确认的 8/27 下午稳定版提交是：

```text
ba7ae40  8/27下午稳定版
```

注意：当前这台电脑的工作区里，`config.py` 和 `robot.py` 还有未提交修改。另一台电脑如果要严格使用截图里的稳定版，请以 `ba7ae40` 为准。

## 方案 A：使用远程仓库

如果你已经把本仓库推到了 GitHub、Gitee 或公司 Git 服务器，在另一台 Windows 电脑执行：

```powershell
git clone <你的远程仓库地址> UR10_vibration_pretest
cd UR10_vibration_pretest
git checkout ba7ae40
```

如果希望在稳定版基础上继续改代码，建议新建分支：

```powershell
git switch -c test-on-new-pc
```

## 方案 B：没有远程仓库时，使用 Git bundle

在原电脑生成 bundle：

```powershell
git bundle create UR10_8-27_stable_ba7ae40.bundle ba7ae40
```

把 `UR10_8-27_stable_ba7ae40.bundle` 拷到另一台电脑后执行：

```powershell
git clone UR10_8-27_stable_ba7ae40.bundle UR10_vibration_pretest
cd UR10_vibration_pretest
git checkout ba7ae40
git switch -c test-on-new-pc
```

## Windows 环境要求

- Windows 64-bit
- Python 3.12.x 64-bit，当前验证版本是 Python 3.12.10
- Git for Windows
- VS Code 可选，但推荐使用
- 海康 MVS SDK：真相机采集需要单独安装，不能通过 pip 安装
- UR 机器人网络：真机测试需要配置机器人 IP、电脑网卡、防火墙和控制器侧设置

## 一键创建 Python 环境

进入项目目录后执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

如果你是通过本地传输包拿到 `setup_windows.ps1`，请先把它复制到 clone 后的项目根目录，也就是和 `requirements-lock-win-py312.txt` 同一级的位置。

如果 pip 访问默认源很慢或失败，可以使用清华源：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1 -UseTsinghuaMirror
```

脚本会创建 `.venv`、安装 `requirements-lock-win-py312.txt` 中锁定的依赖，并运行 `pip check`。

## 手动环境步骤

如果不想运行脚本，也可以手动执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-py312.txt
.\.venv\Scripts\python.exe -m pip check
```

## 启动和测试

启动 UI：

```powershell
.\.venv\Scripts\python.exe launcher_ui.py
```

或者双击：

```text
run_launcher_ui.cmd
```

建议换电脑后的测试顺序：

1. 先启动 UI，确认 Python 环境和界面正常。
2. 先跑离线或 dry-run 流程，不连接机器人。
3. 安装海康 MVS SDK 后，再测试相机导入和采集。
4. 配置 UR 网络后，先用只读/低风险模式确认通信。
5. 只有现场安全确认后，再打开配置里的真机运动许可开关。

## VS Code 设置

打开项目目录后，选择解释器：

```text
${workspaceFolder}\.venv\Scripts\python.exe
```

仓库里的 `.vscode/settings.json` 已经默认指向这个解释器，并设置了 `PYTHONUTF8=1`，可以减少 Windows 控制台中文编码问题。
