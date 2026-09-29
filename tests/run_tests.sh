#!/bin/bash
# Test della pipeline senza GPU e senza rete, con un modello Llama minuscolo
# creato al volo (make_tiny.py) e dataset finti. Uso, dalla cartella del repo
# (dentro il container o in un ambiente con le dipendenze di build/):
#     bash tests/run_tests.sh
# Ogni test stampa le verifiche superate; lo script si ferma al primo errore.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
export HF_HUB_OFFLINE=1
PY=${PY:-python3.11}
mkdir -p hfcache
[ -d tiny ] || $PY make_tiny.py
for t in test_batched_sampling test_integration test_costs test_paths test_orchestration test_split test_quant_section test_new_fixes; do
    echo "=== $t"
    $PY $t.py > $t.log 2>&1 || { echo "!!! $t FALLITO, vedi tests/$t.log"; tail -20 $t.log; exit 1; }
    grep -E "^OK|SUPERAT|OK$|OK:" $t.log | tail -12
done
# Replica del paper: scarica da Hugging Face i dataset di lm-polygraph (serve la
# rete, qualche centinaio di MB la prima volta) per verificare che le domande
# scelte siano le stesse della libreria. SKIP_NETWORK_TESTS=1 per saltarlo.
if [ "${SKIP_NETWORK_TESTS:-0}" != "1" ]; then
    echo "=== test_paper_replica"
    HF_HUB_OFFLINE=0 $PY test_paper_replica.py > test_paper_replica.log 2>&1 || { echo "!!! test_paper_replica FALLITO, vedi tests/test_paper_replica.log"; tail -20 test_paper_replica.log; exit 1; }
    grep -E "^OK|SUPERAT|OK$|OK:" test_paper_replica.log | tail -12
fi
echo "TUTTI I TEST SUPERATI"
