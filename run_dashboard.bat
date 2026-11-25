@echo off
REM Starts the Streamlit dashboard on http://localhost:8501 (run_server.bat must be running).
cd /d "%~dp0"
if not exist .venv (
    echo Run run_server.bat first to create the environment.
    pause
    exit /b 1
)
call .venv\Scripts\activate
streamlit run dashboard\app.py --server.port 8501
