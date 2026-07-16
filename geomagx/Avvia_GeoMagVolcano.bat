@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   GeoMagVolcano Monitor - avvio
echo ============================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERRORE] Python non trovato nel PATH.
    echo Installa Python 3 da https://www.python.org/downloads/
    echo e riprova ^(assicurati di spuntare "Add Python to PATH"^).
    pause
    exit /b 1
)

echo Verifica/installazione dipendenze...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if %errorlevel% neq 0 (
    echo [AVVISO] Alcune dipendenze potrebbero non essersi installate correttamente.
)

echo Avvio dell'app nel browser predefinito...
python geomagx.py

echo.
echo App terminata. Premi un tasto per chiudere questa finestra.
pause >nul
