#!/bin/bash
# Primo argomento opzionale: lo script da eseguire (default main.py), es.
#   bash sbatch_script.sh paper_replica.py --parts whitebox
if [[ "${1:-}" == *.py ]]; then
    script="$1"; shift
else
    script=main.py
fi
python3.11 "$script" "$@"
