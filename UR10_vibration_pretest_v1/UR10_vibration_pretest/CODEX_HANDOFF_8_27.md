# 给另一台电脑 Codex 的交接指令

请在另一台 Windows 电脑上帮助我恢复并测试这个项目的 8/27 下午稳定版。

稳定版提交：

```text
ba7ae4084051ddb4c6b4ed51128bff18b15585e6  8/27下午稳定版
```

我会把以下文件放到另一台电脑：

- `UR10_8-27_stable_ba7ae40.bundle`
- `SETUP_8_27_STABLE.md`
- `setup_windows.ps1`

请执行：

```powershell
git clone .\UR10_8-27_stable_ba7ae40.bundle UR10_vibration_pretest
cd .\UR10_vibration_pretest
git checkout ba7ae4084051ddb4c6b4ed51128bff18b15585e6
git switch -c test-on-new-pc
Copy-Item ..\setup_windows.ps1 .\
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

如果 pip 默认源失败，请改用：

```powershell
.\setup_windows.ps1 -UseTsinghuaMirror
```

环境搭好后请验证：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe launcher_ui.py
```

请注意：

- 不要复制旧电脑的 `.venv`，必须在新电脑重新创建。
- 海康 MVS SDK 不能靠 pip 安装，需要在新电脑单独安装。
- 真机 UR 测试前，需要确认机器人 IP、Windows 网卡、防火墙和控制器设置。
- 未做现场安全确认前，不要打开真机运动许可开关。
- 先测试 UI、离线分析或 dry-run，再测试相机，再测试 UR 通信。

完成后请告诉我：

- 当前 Git 提交是否为 `ba7ae40`
- Python 版本
- `pip check` 是否通过
- UI 是否能启动
- 海康 SDK 和 UR 通信是否还缺本机配置
