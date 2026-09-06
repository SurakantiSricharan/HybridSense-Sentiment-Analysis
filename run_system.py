"""
HybridSense-X Unified System Master Launcher
Runs the complete enterprise full-stack platform with a single command:
1. FastAPI Backend & Web Portal (Port 8000)
2. Streamlit Data Science Studio (Port 8501)
3. Automatically opens browser to http://127.0.0.1:8000
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 65)
    print("    HYBRIDSENSE-X: UNIFIED ENTERPRISE SYSTEM LAUNCHER")
    print("=" * 65)
    print("Starting full-stack services...\n")

    python_exe = sys.executable

    # 1. Launch FastAPI Backend + Web Portal on port 8000
    print("[1/2] Starting FastAPI Backend & Web Portal on port 8000...")
    api_cmd = [
        python_exe, "-m", "uvicorn", "src.api:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    api_proc = subprocess.Popen(
        api_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 2. Launch Streamlit Data Science Studio on port 8501
    print("[2/2] Starting Streamlit Data Science Studio on port 8501...")
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "app/app.py",
        "--server.headless", "true",
        "--server.port", "8501"
    ]
    st_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait 3 seconds for boot
    time.sleep(3)

    print("\n" + "=" * 65)
    print("    ALL SERVICES ARE LIVE & CONNECTED!")
    print("=" * 65)
    print("  🌐 Main Web Application : http://127.0.0.1:8000")
    print("  🧪 Data Science Studio  : http://127.0.0.1:8501")
    print("  ⚡ REST API Docs (Swagger): http://127.0.0.1:8000/docs")
    print("  💾 Database             : data/hybridsense.db (SQLite)")
    print("=" * 65)
    print("Opening the Main Web Portal in your browser...")
    print("Press CTRL + C at any time to shut down all services cleanly.\n")

    webbrowser.open("http://127.0.0.1:8000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down HybridSense-X services gracefully...")
        api_proc.terminate()
        st_proc.terminate()
        api_proc.wait()
        st_proc.wait()
        print("All services stopped. Goodbye!")

if __name__ == "__main__":
    main()
