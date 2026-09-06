#!/bin/bash
# Allinea ~/mental_chatbot/results fra faretra e moro232.
#
# Perche' serve: i due nodi NON condividono la home (verificato: stesso path,
# dispositivi diversi, contenuti diversi). Un job SLURM scrive i risultati solo
# sul nodo su cui e' atterrato, quindi l'altra copia resta indietro in silenzio.
# E' cosi' che una run di settembre e' ripartita dai checkpoint di agosto,
# saltando quattro modelli su cinque e producendo figure "nuove" fatte di dati
# vecchi, senza un solo messaggio d'errore.
#
# Uso, sempre DA FARETRA:
#   bash sync_results.sh from-moro    # moro232 -> faretra  (dopo un job)
#   bash sync_results.sh to-moro      # faretra -> moro232  (prima di un job)
#   bash sync_results.sh check        # confronta e basta, non tocca nulla
#
# La copia precedente della destinazione viene rinominata in results.prev
# invece di essere cancellata: se la sincronizzazione va storta si torna
# indietro con un mv.
set -euo pipefail

REPO="/home/patrignani/mental_chatbot"
NODE="moro232"
TMP="/tmp/sync_results_$$.tgz"
trap 'rm -f "$TMP"' EXIT

# Impronta dell'albero: nomi + contenuti di tutti i file, in un solo hash.
FINGERPRINT_CMD='cd '"$REPO"' && if [ -d results ]; then find results -type f ! -name "*.prev" -exec md5sum {} + | sort -k2 | md5sum | cut -d" " -f1; else echo VUOTA; fi'

fingerprint_local() { bash -c "$FINGERPRINT_CMD"; }
fingerprint_remote() { srun -w "$NODE" bash -c "$FINGERPRINT_CMD"; }

check() {
    local a b
    echo "Calcolo impronte..."
    a=$(fingerprint_local)
    b=$(fingerprint_remote)
    echo "  faretra : $a"
    echo "  $NODE: $b"
    if [ "$a" = "$b" ]; then
        echo "OK: i due nodi sono allineati."
        return 0
    fi
    echo "DIVERSI: i due nodi hanno results/ differenti."
    return 1
}

swap_in() {  # $1 = directory appena estratta
    if [ -d "$REPO/results" ]; then
        rm -rf "$REPO/results.prev"
        mv "$REPO/results" "$REPO/results.prev"
        echo "Copia precedente conservata in results.prev"
    fi
    mv "$1" "$REPO/results"
}

case "${1:-}" in
  from-moro)
    echo "Copio results/ da $NODE a faretra..."
    srun -w "$NODE" tar czf - -C "$REPO" results > "$TMP"
    tar tzf "$TMP" >/dev/null || { echo "!!! archivio corrotto, non tocco nulla."; exit 1; }
    rm -rf "$REPO/results.incoming"
    mkdir -p "$REPO/results.incoming"
    tar xzf "$TMP" -C "$REPO/results.incoming"
    swap_in "$REPO/results.incoming/results"
    rmdir "$REPO/results.incoming" 2>/dev/null || true
    check
    ;;
  to-moro)
    echo "Copio results/ da faretra a $NODE..."
    [ -d "$REPO/results" ] || { echo "!!! non c'e' results/ su faretra."; exit 1; }
    tar czf "$TMP" -C "$REPO" results
    srun -w "$NODE" bash -c "
        set -e
        cd $REPO
        rm -rf results.incoming && mkdir results.incoming
        cat > /tmp/incoming_\$\$.tgz
        tar tzf /tmp/incoming_\$\$.tgz >/dev/null
        tar xzf /tmp/incoming_\$\$.tgz -C results.incoming
        if [ -d results ]; then rm -rf results.prev && mv results results.prev; fi
        mv results.incoming/results results
        rmdir results.incoming 2>/dev/null || true
        rm -f /tmp/incoming_\$\$.tgz
    " < "$TMP"
    check
    ;;
  check)
    check
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
