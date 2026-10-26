@echo off
rem Compatibility launcher: run the ready Qwen model, not legacy GuppyLM.
call "%~dp0run.bat" %*
exit /b %errorlevel%
