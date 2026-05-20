#!/usr/bin/env bash
# Launch the ViolenceLang Topology Builder (web app + generator backend).
#
# Usage:
#   ./graph_generator/run.sh            # serve on http://127.0.0.1:5000
#   VIOLENCE_PORT=8080 ./graph_generator/run.sh
#
# The script runs from the repository root so the package imports resolve and
# the bundled mal-toolbox is found automatically.
set -euo pipefail

# Resolve repo root (this script lives in graph_generator/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"

# Friendly dependency check (does not auto-install — your call).
"${PYTHON}" - <<'PY'
import importlib.util, sys  # importlib.util must be imported explicitly
missing = [m for m in ("flask", "python_jsonschema_objects")
           if importlib.util.find_spec(m) is None]
if missing:
    print("Missing Python packages: " + ", ".join(missing))
    print("Install with:")
    print("    pip install flask python-jsonschema-objects")
    sys.exit(1)
PY

exec "${PYTHON}" -m graph_generator.app
