#!/usr/bin/env bash
# kdrive.py icin hcanakbas kimlik bilgilerini (~/.kaggle/kaggle.json yerine
# ~/.kaggle_hcanakbas/kaggle.json) enjekte eden ince sarmalayici.
set -euo pipefail
CREDS="$HOME/.kaggle_hcanakbas/kaggle.json"
export KAGGLE_USERNAME=$(python3 -c "import json;print(json.load(open('$CREDS'))['username'])")
export KAGGLE_KEY=$(python3 -c "import json;print(json.load(open('$CREDS'))['key'])")
exec python3 "$(dirname "$0")/kdrive.py" "$@"
