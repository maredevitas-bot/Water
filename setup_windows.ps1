param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Test-Path -LiteralPath '.venv') {
        throw 'Incomplete .venv found. Rename it as a backup, then run setup again.'
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 from python.org.' }
        python -m venv .venv
    } else {
        throw 'Install Python 3.12 x64 from python.org, including the Python launcher.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 from python.org, including the Python launcher.' }
}
& '.\.venv\Scripts\python.exe' -c "import sys,struct; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) { throw 'This .venv must use Python 3.12 x64. Rename it as a backup and rerun setup.' }
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
if ($CpuOnly) {
    & '.\.venv\Scripts\python.exe' -m ripple.doctor
} else {
    & '.\.venv\Scripts\python.exe' -m ripple.doctor --gpu
}
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed. See the error above; training has not started.' }
Write-Host 'Setup complete. Run 02_prepare.cmd, then 03_train_gpu.cmd (or 03_train_cpu.cmd).'
