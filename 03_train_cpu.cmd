@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ripple.train --device CPU
if errorlevel 1 goto fail
echo Training complete. See the runs folder.
pause
exit /b 0
:fail
echo Training failed. Read the error above.
pause
exit /b 1
