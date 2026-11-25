#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
[ -f .venv/.setup_done ] || { pip install -r server/requirements.txt && touch .venv/.setup_done; }
python -m server.main
