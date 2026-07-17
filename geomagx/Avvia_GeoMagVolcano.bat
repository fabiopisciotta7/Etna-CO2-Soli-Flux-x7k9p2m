@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   GeoMagVolcano Monitor - avvio
echo ============================================================

set PYCMD=

rem Preferisci il launcher "py" (installato da python.org): a differenza di
rem "python" non viene "oscurato" da altri programmi che portano con se'
rem un proprio python.exe senza pip (es. Inkscape, GIMP, ecc.) e che a volte
rem finiscono prima nel PATH di sistema.
py -3 -m pip --version >nul 2>nul
if not errorlevel 1 (
    set PYCMD=py -3
) else (
    python -m pip --version >nul 2>nul
    if not errorlevel 1 (
        set PYCMD=python
    )
)

if "%PYCMD%"=="" (
    echo [ERRORE] Non ho trovato un Python funzionante con pip.
    echo.
    echo Possibili cause:
    echo  - Python non e' installato: scaricalo da https://www.python.org/downloads/
    echo    ^(spunta "Add python.exe to PATH" durante l'installazione^)
    echo  - Un altro programma ^(es. Inkscape, GIMP^) ha aggiunto al PATH
    echo    un proprio python.exe senza pip, che viene trovato per primo.
    echo    In quel caso disinstalla/reinstalla Python da python.org e
    echo    assicurati che compaia il launcher "py".
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
