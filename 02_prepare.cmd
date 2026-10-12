@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ripple.fetch
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ripple.prepare
if errorlevel 1 goto fail
echo Data preparation complete.
pause
exit /b 0
:fail
echo Preparation failed. Read the error above.
pause
exit /b 1
