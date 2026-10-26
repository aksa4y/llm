@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
 echo Установите Python 3.10 или новее и включите Add Python to PATH.
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv .venv
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m guppylm prepare
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m guppylm train
if errorlevel 1 goto :fail
pause
exit /b 0
:fail
echo Ошибка. Проверьте сообщение выше.
pause
exit /b 1
