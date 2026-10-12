@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ripple.doctor --gpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ripple.train --device GPU
if errorlevel 1 goto fail
echo Training complete. See the runs folder.
pause
exit /b 0
:fail
echo Training failed. Read the error above. No CPU fallback was used.
pause
exit /b 1
