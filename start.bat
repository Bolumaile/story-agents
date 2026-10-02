@echo off
rem ===================================================================
rem  Story Agents launcher - starts the web UI on port 8765.
rem  NOTE: this file is intentionally ASCII-only (no Chinese characters)
rem        and does NOT call "chcp"; both are known causes of .bat
rem        crashing / closing instantly on Chinese Windows.
rem ===================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "PY="
set "VENV_PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "%VENV_PY%" set "PY=%VENV_PY%"

if not defined PY (
  for /f "delims=" %%D in ('dir /b /ad /o-n "%USERPROFILE%\.workbuddy\binaries\python\versions" 2^>nul') do (
    if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\%%D\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\versions\%%D\python.exe"
  )
)

if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)

if not defined PY (
  echo.
  echo [Story Agents] ERROR: no Python interpreter found.
  echo   Looked in: %USERPROFILE%\.workbuddy\binaries\python\
  echo   and in PATH.
  echo.
  pause
  exit /b 1
)

echo [Story Agents] python  = %PY%
echo [Story Agents] workdir = %CD%

rem ---- dependency self-check -------------------------------------------------
"%PY%" -c "import fastapi, uvicorn, langgraph, openai" >nul 2>nul
if errorlevel 1 (
  echo [Story Agents] Dependencies missing - installing from requirements.txt ...
  "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [Story Agents] ERROR: dependency install failed. Check network and retry.
    echo.
    pause
    exit /b 1
  )
)

rem ---- already running? ------------------------------------------------------
set "RUNNING="
for /f "tokens=*" %%L in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do set "RUNNING=1"
if defined RUNNING (
  echo.
  echo [Story Agents] Port 8765 is ALREADY in use - a server is probably already running.
  echo.
  echo   ###  IF YOU EDITED ANY .py FILE, READ THIS FIRST  ###
  echo   The running server was started BEFORE your edit and still holds the
  echo   OLD code in memory - it runs WITHOUT --reload. Re-opening the page
  echo   will NOT pick up your changes. You would be testing old code.
  echo.
  echo   To load the new code:
  echo     1. Find the console window running uvicorn, press Ctrl+C there
  echo     2. Run start.bat again
  echo.
  echo   If you did NOT change any code, ignore this and use the page.
  echo.
  echo [Story Agents] Opening http://127.0.0.1:8765 ...
  start "" "http://127.0.0.1:8765"
  echo.
  echo   If your browser did not open, copy this address manually:
  echo       http://127.0.0.1:8765
  echo.
  pause
  exit /b 0
)

rem ---- start server (browser opens ~2s later) --------------------------------
echo.
echo [Story Agents] Starting web UI at http://127.0.0.1:8765
echo [Story Agents] Browser opens in 2 seconds. Keep this window open.
echo [Story Agents] Press Ctrl+C in this window to stop the server.
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"

"%PY%" -m uvicorn web.server:app --port 8765 --host 127.0.0.1

echo.
echo [Story Agents] Server stopped.
echo   If the browser did not open automatically, the address is:
echo       http://127.0.0.1:8765
echo.
pause
endlocal
