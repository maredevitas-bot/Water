@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ripple.train --device CPU --smoke
if errorlevel 1 goto fail
echo Verification complete. Smoke metrics are not research results.
pause
exit /b 0
:fail
echo Verification failed.
pause
exit /b 1
