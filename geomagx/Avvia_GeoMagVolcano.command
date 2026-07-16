#!/usr/bin/env bash
# Doppio clic su questo file (macOS) per avviare GeoMagVolcano Monitor.
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  GeoMagVolcano Monitor - avvio"
echo "============================================================"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERRORE] python3 non trovato."
    echo "Installa Python 3 da https://www.python.org/downloads/ e riprova."
    read -n 1 -s -r -p "Premi un tasto per chiudere..."
    exit 1
fi

echo "Verifica/installazione dipendenze..."
"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check -r requirements.txt || \
    echo "[AVVISO] Alcune dipendenze potrebbero non essersi installate correttamente."

echo "Avvio dell'app nel browser predefinito..."
"$PYTHON_BIN" geomagx.py

echo ""
read -n 1 -s -r -p "App terminata. Premi un tasto per chiudere questa finestra..."
