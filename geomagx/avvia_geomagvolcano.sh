#!/usr/bin/env bash
# Esegui con: ./avvia_geomagvolcano.sh  (rendilo eseguibile una volta con: chmod +x avvia_geomagvolcano.sh)
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  GeoMagVolcano Monitor - avvio"
echo "============================================================"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERRORE] python3 non trovato. Installa Python 3 e riprova."
    exit 1
fi

# Evita che al primo avvio Streamlit resti in attesa dell'email di onboarding
# (il prompt "Welcome to Streamlit!" nel terminale).
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "Verifica/installazione dipendenze..."
"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check -r requirements.txt || \
    echo "[AVVISO] Alcune dipendenze potrebbero non essersi installate correttamente."

echo "Avvio dell'app nel browser predefinito..."
"$PYTHON_BIN" geomagx.py
