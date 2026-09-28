#!/bin/bash
# Controlli prima di lanciare un run lungo. Solo lettura: non modifica nulla.
# Uso, su OGNI nodo (faretra e moro232), dalla cartella del repository:
#   bash preflight.sh
# Esito finale: "TUTTO OK" oppure l'elenco dei problemi da sistemare.
set -u
REPO="${REPO:-/home/patrignani/mental_chatbot}"
IMAGE="mental-chatbot-image"
BRANCH="filo/benchmark-uq"
LLM_CACHE="${LLM_CACHE:-/llms}"
GATED=("google/medgemma-4b-it" "google/gemma-3-4b-it")
MODELS=("LiquidAI/LFM2-350M" "LiquidAI/LFM2-1.2B" "google/medgemma-4b-it" "google/gemma-3-4b-it" "mistralai/Mistral-7B-Instruct-v0.2")

PROBLEMI=()
ok()   { echo "  [ok]  $*"; }
warn() { echo "  [!!]  $*"; }
ko()   { echo "  [KO]  $*"; PROBLEMI+=("$*"); }

echo "== Nodo: $(hostname)"
cd "$REPO" 2>/dev/null || { echo "Cartella $REPO assente"; exit 1; }

echo "== 1. Codice"
git fetch --all -q 2>/dev/null || warn "git fetch fallito (rete?)"
CUR=$(git rev-parse --abbrev-ref HEAD)
[ "$CUR" = "$BRANCH" ] && ok "branch $CUR" || ko "branch attuale $CUR, atteso $BRANCH (git checkout $BRANCH)"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "@{u}" 2>/dev/null || echo "?")
echo "        commit locale: $(git log -1 --format='%h %s' | cut -c1-70)"
if [ "$REMOTE" = "?" ]; then warn "branch senza upstream: impossibile confrontare col remote"
elif [ "$LOCAL" = "$REMOTE" ]; then ok "allineato al remote ($(git rev-parse --short HEAD))"
else ko "non allineato al remote: locale $(git rev-parse --short HEAD), remote $(git rev-parse --short "$REMOTE") (git pull)"; fi
DIRTY=$(git status --porcelain --untracked-files=no)
[ -z "$DIRTY" ] && ok "nessuna modifica locale ai file versionati" || ko "modifiche locali non committate: $(echo $DIRTY | tr '\n' ' ')"
grep -q "def run_cell_chunked" main.py && grep -q '"--datasets"' main.py \
    && ok "main.py contiene le correzioni (blocchi, --datasets)" || ko "main.py e' una versione vecchia"
grep -q "lm-polygraph==0.7.0" build/requirements-benchmark.txt && ok "lm-polygraph fissato a 0.7.0" \
    || ko "requirements-benchmark.txt senza lm-polygraph==0.7.0"

echo "== 2. Token Hugging Face"
if [ -z "${HF_TOKEN:-}" ]; then
    ko "HF_TOKEN non impostato in questa shell (export HF_TOKEN=... prima di sbatch_script.sh)"
else
    WHO=$(curl -s -m 20 -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 | python3 -c "import sys,json;print(json.load(sys.stdin).get('name',''))" 2>/dev/null)
    [ -n "$WHO" ] && ok "token valido (utente: $WHO)" || ko "token HF non valido o HF non raggiungibile"
    for m in "${GATED[@]}"; do
        CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 20 -I -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$m/resolve/main/config.json")
        case "$CODE" in
            200|302|307) ok "accesso a $m";;
            401|403) ko "nessun accesso a $m (HTTP $CODE): accettare la licenza su huggingface.co/$m";;
            *) warn "$m: HTTP $CODE (rete?)";;
        esac
    done
fi

echo "== 3. Modelli gia' in cache ($LLM_CACHE)"
for m in "${MODELS[@]}"; do
    d="$LLM_CACHE/hub/models--${m//\//--}"
    [ -d "$d" ] && ok "$m ($(du -sh "$d" 2>/dev/null | cut -f1))" || warn "$m non in cache: verra' scaricato all'avvio"
done

echo "== 4. Immagine Docker"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    CREATED=$(docker image inspect -f '{{.Created}}' "$IMAGE")
    CREATED_TS=$(date -d "$CREATED" +%s 2>/dev/null || echo 0)
    REQ_TS=$(git log -1 --format=%ct -- build/)
    [ "$CREATED_TS" -ge "$REQ_TS" ] && ok "immagine costruita dopo l'ultima modifica a build/ ($CREATED)" \
        || ko "immagine piu' vecchia dell'ultima modifica a build/: bash create_docker_image.sh"
    VERS=$(docker run --rm "$IMAGE" python3.11 -c "import lm_polygraph, transformers, torch; from importlib.metadata import version; print(version('lm-polygraph'), transformers.__version__, torch.__version__)" 2>/dev/null)
    echo "        nell'immagine: lm-polygraph / transformers / torch = $VERS"
    [[ "$VERS" == 0.7.0* ]] && ok "lm-polygraph 0.7.0 nell'immagine" || ko "lm-polygraph nell'immagine non e' 0.7.0: ricostruire l'immagine"
else
    ko "immagine $IMAGE assente: bash create_docker_image.sh"
fi

echo "== 5. GPU, disco, cluster"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/        /'
else warn "nvidia-smi non disponibile su questo nodo"; fi
for p in "$REPO" "$LLM_CACHE"; do
    AV=$(df -Pk "$p" 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}')
    [ -n "$AV" ] && { [ "$AV" -ge 30 ] && ok "$p: ${AV} GB liberi" || ko "$p: solo ${AV} GB liberi (servono modelli + checkpoint)"; }
done
if command -v scontrol >/dev/null; then
    echo "        limite di tempo delle partizioni (un job oltre il limite viene ucciso; i blocchi lo rendono innocuo):"
    scontrol show partition 2>/dev/null | grep -oE "PartitionName=[^ ]+|MaxTime=[^ ]+" | paste - - | sed 's/^/          /'
    echo "        job in coda/in esecuzione:"; squeue -h -o "          %i %u %j %T %N" 2>/dev/null | head -10
fi

echo "== 6. Cartella risultati"
if [ -d results ] && [ -n "$(ls -A results 2>/dev/null)" ]; then
    if [ -d results/chunks ]; then ok "results/ contiene gia' checkpoint a blocchi: un nuovo lancio riprendera' da li'"
    else ko "results/ contiene risultati della versione vecchia: spostali (mv results results_13_09) prima del run"; fi
else ok "results/ vuota o assente"; fi

echo
if [ ${#PROBLEMI[@]} -eq 0 ]; then echo "TUTTO OK su $(hostname)"
else echo "DA SISTEMARE su $(hostname):"; for p in "${PROBLEMI[@]}"; do echo "  - $p"; done; exit 1; fi
