@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :missing
if not exist "checkpoints\final_model.pt" goto :missing
".venv\Scripts\python.exe" -m guppylm chat --checkpoint checkpoints/final_model.pt
pause
exit /b 0
:missing
echo Legacy GuppyLM weights not found. Run train_ru.bat first.
echo For the ready Qwen assistant, use run.bat instead.
pause
exit /b 1
