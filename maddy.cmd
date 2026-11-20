@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY_CMD=%~dp0.venv\Scripts\python.exe"

set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=cli"
if not "%~1"=="" shift

if /I "%COMMAND%"=="cli" goto cli
if /I "%COMMAND%"=="app" goto cli
if /I "%COMMAND%"=="run" goto cli
if /I "%COMMAND%"=="listen" goto voice
if /I "%COMMAND%"=="mic" goto voice
if /I "%COMMAND%"=="voice" goto voice
if /I "%COMMAND%"=="wakeword" goto voice
if /I "%COMMAND%"=="knock" goto knock
if /I "%COMMAND%"=="stop" goto stop
if /I "%COMMAND%"=="stop-capture" goto stop
if /I "%COMMAND%"=="capture-status" goto status
if /I "%COMMAND%"=="help" goto help
if /I "%COMMAND%"=="/?" goto help

goto cli_with_command

:cli
%PY_CMD% -m app.cli %*
goto end

:cli_with_command
%PY_CMD% -m app.cli %COMMAND% %*
goto end

:voice
%PY_CMD% -u voice_listener.py %*
goto end

:knock
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"
if not "%~1"=="" shift
%PY_CMD% core\knock_detector.py %ACTION% %*
goto end

:stop
%PY_CMD% -m session_capture.cli stop %*
goto end

:status
%PY_CMD% -m session_capture.cli status %*
goto end

:help
echo Usage: maddy.cmd ^<command^> [arguments]
echo.
echo   cli, app, run       Start text CLI
echo   listen, mic, voice Start voice listener
echo   knock [action]      Control knock detector
echo   stop                Stop session capture
echo   capture-status      Show capture status

goto end

:end
endlocal
