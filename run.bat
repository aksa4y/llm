@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
 echo Установите 64-битный Python 3.12 с python.org и включите Add Python to PATH.
 pause
 exit /b 1
)
if not exist .venv-chat\Scripts\python.exe python -m venv .venv-chat
if errorlevel 1 goto :fail
.venv-chat\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements-chat.txt
if errorlevel 1 goto :fail
.venv-chat\Scripts\python.exe russian_chat.py %*
if errorlevel 1 goto :fail
pause
exit /b 0
:fail
echo Не удалось запустить помощника. Смотрите сообщение выше.
pause
exit /b 1
