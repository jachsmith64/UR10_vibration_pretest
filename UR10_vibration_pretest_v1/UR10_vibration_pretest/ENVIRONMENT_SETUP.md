# Python Environment Setup

This project is intended to run on Windows 64-bit with Python 3.12.x.
The environment verified for this project is Python 3.12.10, 64-bit.

## Create The Virtual Environment

Create a project-local virtual environment named `.venv` from the project directory:

```powershell
py -3.12 -m venv .venv
```

If the Windows Python launcher does not detect Python 3.12, use the full path to a Python 3.12 64-bit interpreter instead:

```powershell
C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
```

Do not copy `.venv` from another computer. It contains machine-specific paths and must be recreated on each Windows PC.

## Install Dependencies

Use the virtual environment's Python explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-py312.txt
.\.venv\Scripts\python.exe -m pip check
```

If the default Python package index is unreachable, use a temporary mirror for the install command:

```powershell
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-lock-win-py312.txt
```

`requirements.txt` lists the direct project dependencies. `requirements-lock-win-py312.txt`, when present, records the exact package versions verified on Windows with Python 3.12.

## VS Code Interpreter

Open the Command Palette and choose:

```text
Python: Select Interpreter
```

Select:

```text
${workspaceFolder}\.venv\Scripts\python.exe
```

The workspace setting `python.defaultInterpreterPath` is configured to point VS Code to this interpreter.

Do not permanently change the global PowerShell execution policy just to activate the virtual environment. If activation scripts cannot run, always call:

```powershell
.\.venv\Scripts\python.exe
```

The VS Code workspace terminal sets `PYTHONUTF8=1` and `MPLCONFIGDIR` locally for this project. This avoids Windows GBK console encoding errors and keeps Matplotlib cache files inside the ignored `outputs` directory.

## External Hardware Dependencies

Hikvision MVS SDK cannot be installed through pip. Install the vendor SDK separately on each computer, then confirm the path that contains `MvCameraControl_class.py` before setting `HIK_MVS_IMPORT_PATH`.

UR robot network settings must also be configured separately on each computer and controller. Confirm the UR IP address, Windows network adapter settings, firewall rules, and controller-side configuration before running any real robot mode.
