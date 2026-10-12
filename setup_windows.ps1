param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 x64 from python.org, including the Python launcher.' }
}
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($CpuOnly) {
    & '.\.venv\Scripts\python.exe' -m ripple.doctor
} else {
    & '.\.venv\Scripts\python.exe' -m ripple.doctor --gpu
}
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed. See the error above; training has not started.' }
Write-Host 'Setup complete. Run 02_prepare.cmd, then 03_train_gpu.cmd (or 03_train_cpu.cmd).'
