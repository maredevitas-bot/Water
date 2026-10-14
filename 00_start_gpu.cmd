@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_home_pc.ps1"
if errorlevel 1 goto fail
echo Complete. Open the latest runs folder and read SUMMARY.md.
pause
exit /b 0
:fail
echo Stopped. Read the error above and the log in reports\logs.
pause
exit /b 1
