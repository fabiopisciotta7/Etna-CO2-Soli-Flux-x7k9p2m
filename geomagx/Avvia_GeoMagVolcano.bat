@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   GeoMagVolcano Monitor - avvio
echo ============================================================

set PYCMD=

python --version >nul 2>nul
if not errorlevel 1 (
    set PYCMD=python
) else (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
        set PYCMD=py -3
    )
)

if "%PYCMD%"=="" (
    echo [ERRORE] Python 3 non trovato sul PC.
    echo Installa Python da https://www.python.org/downloads/
    echo Durante l'installazione spunta la casella "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

echo Interprete Python trovato: %PYCMD%
echo.
echo Verifica/installazione dipendenze...
%PYCMD% -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo [AVVISO] Alcune dipendenze potrebbero non essersi installate correttamente.
)

echo.
echo Avvio dell'app nel browser predefinito...
%PYCMD% geomagx.py

echo.
echo App terminata. Premi un tasto per chiudere questa finestra.
pause >nul
