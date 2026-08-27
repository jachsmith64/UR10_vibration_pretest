param(
    [switch]$UseTsinghuaMirror
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonLauncher) {
    throw "Windows Python Launcher was not found. Install Python 3.12.x 64-bit and enable Add python.exe to PATH."
}

Write-Host "Checking Python 3.12..."
py -3.12 -c "import platform, sys; print(sys.version); assert platform.architecture()[0] == '64bit'"

if (-not (Test-Path -LiteralPath ".venv")) {
    Write-Host "Creating .venv..."
    py -3.12 -m venv .venv
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python was not found: $python"
}

New-Item -ItemType Directory -Force -Path "outputs\matplotlib-cache" | Out-Null

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip

if ($UseTsinghuaMirror) {
    Write-Host "Installing dependencies from Tsinghua mirror..."
    & $python -m pip install -i "https://pypi.tuna.tsinghua.edu.cn/simple" -r requirements-lock-win-py312.txt
} else {
    Write-Host "Installing dependencies..."
    & $python -m pip install -r requirements-lock-win-py312.txt
}

Write-Host "Checking dependency consistency..."
& $python -m pip check

Write-Host ""
Write-Host "Environment ready."
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\python.exe launcher_ui.py"
Write-Host ""
Write-Host "Note: Hikvision MVS SDK and UR network/firewall/controller settings must be configured on each PC."
