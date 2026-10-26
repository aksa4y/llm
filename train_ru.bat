@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
 echo Install Python 3.12 x64 and enable Add Python to PATH.
 pause
 exit /b 1
)
if exist ".venv\Scripts\python.exe" goto :install
python -m venv .venv
if errorlevel 1 goto :fail
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m guppylm prepare
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m guppylm train
if errorlevel 1 goto :fail
pause
exit /b 0
:fail
echo Training failed. Please check the error message above.
pause
exit /b 1
