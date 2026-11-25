@echo off
REM Starts the FastAPI sensor server on 0.0.0.0:8000 (creates .venv on first run).
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate
if not exist .venv\.setup_done (
    echo Installing requirements - first run only...
    python -m pip install --upgrade pip
    pip install -r server\requirements.txt || goto :error
    echo done> .venv\.setup_done
)
echo.
echo Starting server. Enter the ws:// URL printed below into the Android app.
python -m server.main
goto :eof
:error
echo Installation failed.
pause