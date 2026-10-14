param(
    [ValidateSet('GPU', 'CPU')][string]$Device = 'GPU',
    [switch]$SkipInstall,
    [switch]$Offline,
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$waterLogDir = Join-Path $PSScriptRoot 'reports\logs'
New-Item -ItemType Directory -Force -Path $waterLogDir | Out-Null
$waterLog = Join-Path $waterLogDir ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ') + '.txt')
Start-Transcript -LiteralPath $waterLog | Out-Null
$waterExit = 0
try {
    if (-not ($SkipInstall -or $Offline)) {
        $waterSetup = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'setup_windows.ps1'))
        if ($Device -eq 'CPU') { $waterSetup += '-CpuOnly' }
        & powershell.exe @waterSetup
        if ($LASTEXITCODE -ne 0) { throw 'Setup failed. Training has not started.' }
    }
    $waterPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $waterPython)) {
        throw 'No installed environment. Run 01_setup.cmd online first.'
    }
    $waterArgs = @('-u', '-m', 'ripple.workflow', '--device', $Device)
    if ($Offline) { $waterArgs += '--offline' }
    if ($Smoke) { $waterArgs += '--smoke' }
    & $waterPython @waterArgs
    if ($LASTEXITCODE -ne 0) { throw 'Workflow failed. Check the failed step above. No CPU fallback was used.' }
    Write-Host 'Finished. Results and per-run provenance are in runs; SUMMARY.md explains the evaluation.'
} catch {
    $waterExit = 1
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
} finally {
    Write-Host ('Log: ' + $waterLog)
    Stop-Transcript | Out-Null
}
exit $waterExit
