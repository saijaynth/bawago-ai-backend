@echo off
title BawaGo Isolated AI Agent
echo ========================================================
echo   BawaGo Standalone AI Agent Prototype (Isolated)
echo ========================================================
echo.
echo Starting Isolated AI Agent on http://127.0.0.1:8080 ...
echo Press Ctrl+C in this window to stop anytime.
echo.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python agent_server.py 8080
pause

