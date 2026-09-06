@echo off
title HybridSense-X Unified System Launcher
echo ============================================================
echo      Starting HybridSense-X Integrated Platform
echo ============================================================
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
python run_system.py
pause
