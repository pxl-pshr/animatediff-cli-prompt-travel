@echo off
setlocal
rem AnimateDiff Prompt Travel - Web UI launcher.
rem First run creates ui\.venv (Gradio only); the generator keeps using venv\.
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist "venv\Scripts\python.exe" (
    echo [webui] The main venv was not found at venv\Scripts\python.exe
    echo [webui] Install animatediff first ^(see README "Installation"^), then run this again.
    pause
    exit /b 1
)

if not exist "ui\.venv\Scripts\python.exe" (
    echo [webui] Creating UI environment in ui\.venv ...
    py -3.10 -m venv ui\.venv 2>nul || python -m venv ui\.venv
    if errorlevel 1 (
        echo [webui] Could not create ui\.venv. Is Python 3.10+ installed?
        pause
        exit /b 1
    )
    "ui\.venv\Scripts\python.exe" -m pip install --upgrade pip
)

"ui\.venv\Scripts\python.exe" -c "import gradio" 2>nul
if errorlevel 1 (
    echo [webui] Installing UI requirements ...
    "ui\.venv\Scripts\python.exe" -m pip install -r ui\requirements.txt
    if errorlevel 1 (
        echo [webui] pip install failed.
        pause
        exit /b 1
    )
)

"ui\.venv\Scripts\python.exe" ui\app.py %*
if errorlevel 1 pause
endlocal
