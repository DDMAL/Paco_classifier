#!/bin/bash
# batch_training_gui.py's own tkinter process needs a Python with a working Tk build.
# The project's .venv bundles Apple's system Tcl/Tk 8.5.9, which fails to render
# most widgets on this machine — override PACO_GUI_PYTHON to point elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_PYTHON="${PACO_GUI_PYTHON:-/usr/local/bin/python3}"
exec "$GUI_PYTHON" "$SCRIPT_DIR/batch_training_gui.py"
