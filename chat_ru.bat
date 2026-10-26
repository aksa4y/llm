@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto :missing
if not exist checkpoints\final_model.pt goto :missing
.venv\Scripts\python.exe -m guppylm chat --checkpoint checkpoints/final_model.pt
pause
exit /b 0
:missing
echo Сначала выполните train_ru.bat и дождитесь окончания обучения.
pause
exit /b 1
