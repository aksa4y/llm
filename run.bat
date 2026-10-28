@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
 echo Python was not found. Install Python 3.12 x64 from python.org.
 echo Enable the Add Python to PATH option, then run this file again.
 pause
 exit /b 1
)
if exist ".venv-chat\Scripts\python.exe" goto :install
python -m venv .venv-chat
if errorlevel 1 goto :fail
:install
".venv-chat\Scripts\python.exe" -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements-chat.txt
if errorlevel 1 goto :fail
".venv-chat\Scripts\python.exe" chat_server.py %*
if errorlevel 1 goto :fail
pause
exit /b 0
:fail
echo Failed. Please check the error message above.
pause
exit /b 1
