# Sourced by the shell scripts (from training/): local paths, interpreter, import path.
if [ -f local.env ]; then set -a; . ./local.env; set +a; fi
# the repo's .venv if there is one, else whatever python is first on PATH
if [ -z "${PY:-}" ]; then
    if [ -x ../.venv/bin/python ]; then PY=$(cd .. && pwd)/.venv/bin/python; else PY=python3; fi
fi
export PY
export PYTHONPATH="$PWD:$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # sota_train (here) and sota (repo root)
