@echo off
rem Skinnerbek Studio - double-click to start.
rem First run installs a private copy of Python and the voice models (about 9 GB in total).
setlocal EnableExtensions DisableDelayedExpansion
title Skinnerbek Studio
cd /d "%~dp0"

if not exist "scripts\start.py" goto :not_unzipped
if /i "%PROCESSOR_ARCHITECTURE%"=="x86" if not defined PROCESSOR_ARCHITEW6432 goto :not_64bit

set "UV_VERSION=0.13.5"
set "UV_SHA256=a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
set "APP_HOME=%LOCALAPPDATA%\SkinnerbekStudio"
set "UV_DIR=%APP_HOME%\uv-%UV_VERSION%"
set "UV=%UV_DIR%\uv.exe"
set "VENV=%APP_HOME%\env"
set "PY=%VENV%\Scripts\python.exe"
set "UV_PYTHON_INSTALL_DIR=%APP_HOME%\python"
set "UV_CACHE_DIR=%APP_HOME%\cache"
set "UV_NO_CONFIG=1"
set "UV_PYTHON_PREFERENCE=only-managed"

echo.
echo   Skinnerbek Studio
echo   -----------------
echo.

rem A developer checkout with its own .venv runs as-is.
if exist ".venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" goto :run

if exist "%UV%" goto :python
echo Setting up for the first time. This needs an internet connection.
if not exist "%UV_DIR%" mkdir "%UV_DIR%"
if errorlevel 1 goto :fail_folder
set "UV_ZIP=%UV_DIR%\uv.zip"
echo Downloading the installer tool (uv %UV_VERSION%)...
curl.exe -fsSL --retry 3 -o "%UV_ZIP%" "%UV_URL%" 2>nul
if not errorlevel 1 goto :verify_uv
del "%UV_ZIP%" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri $env:UV_URL -OutFile $env:UV_ZIP"
:verify_uv
if not exist "%UV_ZIP%" goto :fail_download
powershell -NoProfile -ExecutionPolicy Bypass -Command "if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:UV_ZIP).Hash -ne $env:UV_SHA256) { exit 1 }"
if errorlevel 1 goto :fail_checksum
tar -xf "%UV_ZIP%" -C "%UV_DIR%"
if errorlevel 1 powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force -LiteralPath $env:UV_ZIP -DestinationPath $env:UV_DIR"
del "%UV_ZIP%" >nul 2>nul
if not exist "%UV%" goto :fail_download

:python
if exist "%PY%" goto :deps
echo Installing Python 3.12 for Skinnerbek Studio (does not touch any other Python)...
"%UV%" venv --quiet --python cpython-3.12-windows-x86_64-none "%VENV%"
if errorlevel 1 goto :fail_python

:deps
fc /b "requirements.txt" "%VENV%\installed-requirements.txt" >nul 2>nul
if errorlevel 1 goto :install
fc /b "constraints.txt" "%VENV%\installed-constraints.txt" >nul 2>nul
if not errorlevel 1 goto :run
:install
echo Installing the app's components (about 1.5 GB, first run only)...
"%UV%" pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto :fail_deps
copy /y "requirements.txt" "%VENV%\installed-requirements.txt" >nul
copy /y "constraints.txt" "%VENV%\installed-constraints.txt" >nul

:run
"%PY%" scripts\start.py
if errorlevel 1 goto :fail_app
exit /b 0

:not_unzipped
echo This folder is incomplete. If you opened start.bat from inside a .zip file,
echo right-click the .zip, choose "Extract All...", then run start.bat from the extracted folder.
goto :hold
:not_64bit
echo Skinnerbek Studio needs 64-bit Windows.
goto :hold
:fail_folder
echo Could not create "%APP_HOME%". Check that your user account can write there.
goto :hold
:fail_download
echo Download failed. Check your internet connection and run start.bat again.
goto :hold
:fail_checksum
del "%UV_ZIP%" >nul 2>nul
echo The downloaded installer did not match its published checksum, so it was deleted.
echo Run start.bat again. If this keeps happening, your network may be altering downloads.
goto :hold
:fail_python
echo Python could not be installed. Check your internet connection and run start.bat again.
goto :hold
:fail_deps
echo Installing components failed. Check your internet connection and free disk space,
echo then run start.bat again. It continues where it stopped.
goto :hold
:fail_app
echo.
echo Skinnerbek Studio stopped with an error. The message above says why.
:hold
echo.
pause
exit /b 1
