#!/usr/bin/env bash
# DocAudit web UI (macOS / Linux). Run from anywhere:  bash start_ui.sh
# Uses the Python on PATH, or the one in $PYTHON (e.g. PYTHON=.venv/bin/python).
cd "$(dirname "$0")"
exec "${PYTHON:-python}" -m src.ui.app "$@"
