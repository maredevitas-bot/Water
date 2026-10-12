@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 goto fail
echo Setup complete.
pause
exit /b 0
:fail
echo Setup failed. Read the error above.
pause
exit /b 1
