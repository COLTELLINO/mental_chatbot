# Benchmark di Uncertainty Quantification su LLM di piccola scala in ambito clinico

Questo repository contiene il codice degli esperimenti della tesi: un benchmark di
metodi di **Uncertainty Quantification (UQ)** per Large Language Model sotto i 4
miliardi di parametri, su dataset di dominio generale e clinico.

Il benchmark riproduce il protocollo di valutazione di Vashurin et al.,
*Benchmarking Uncertainty Quantification Methods for Large Language Models with
LM-Polygraph* (TACL 2025, arXiv:2406.15627), e lo estende in quattro direzioni:

| Domanda | Cosa si confronta |
|---|---|
| **Scala** | le conclusioni del paper (modelli 7-12B) valgono anche per modelli da 0.35B a 4B? |
| **Dominio** | il ranking dei metodi UQ cambia passando dal dominio generale a quello clinico? |
| **Severità** | l'UQ funziona allo stesso modo su domande dove un errore può essere letale? |
| **Costo** | quanto costa ogni metodo, e cosa resta dopo la quantizzazione a 4 bit? |

Ogni metodo UQ viene valutato con il **Prediction-Rejection Ratio (PRR)** del
paper: quanto del guadagno massimo possibile, rispetto a scartare risposte a caso,
si ottiene scartando quelle con l'incertezza più alta (0 = come il caso, 1 = come
un oracolo). La definizione delle metriche, i valori possibili e come leggerle
sono in [`METRICHE.md`](METRICHE.md).

> **Revisione del 29/09/2026.** Una revisione del codice ha trovato difetti che
> cambiano i risultati: il PRR riportato era l'area grezza e non il rapporto
> normalizzato del paper; i prompt erano in italiano; LFM2 non riceveva il suo
> chat template; i prompt con chat template avevano due token BOS; l'estrazione
> della lettera MCQ leggeva l'articolo "a" come risposta "A"; più problemi nelle
> analisi statistiche e nell'attribuzione dei costi. Tutti corretti; il dettaglio
> è nella sezione 10. **I risultati prodotti prima di questa data non sono
> confrontabili con quelli nuovi**: si possono rileggere con il PRR corretto
> (`recompute_stats.py`), ma i difetti nelle generazioni restano.

---

## Indice

1. [Contenuto del repository](#1-contenuto-del-repository)
2. [Requisiti](#2-requisiti)
3. [Installazione](#3-installazione)
4. [Come si esegue](#4-come-si-esegue)
5. [Argomenti di `main.py`](#5-argomenti-di-mainpy)
6. [Ricette: cosa lanciare per ogni esperimento](#6-ricette-cosa-lanciare-per-ogni-esperimento)
7. [Output prodotti](#7-output-prodotti)
8. [Script di analisi e di supporto](#8-script-di-analisi-e-di-supporto)
9. [Modelli, dataset e metodi](#9-modelli-dataset-e-metodi)
10. [Note metodologiche e limiti noti](#10-note-metodologiche-e-limiti-noti)

---

## 1. Contenuto del repository

| File | Cosa contiene |
|---|---|
| `main.py` | La pipeline: carica modelli e dataset, esegue i metodi UQ con lm-polygraph, calcola PRR, intervalli di confidenza, costi, e salva tutti i CSV. Contiene anche le sezioni extra (griglia di severità, metodi verbalized, confronto sulla quantizzazione). |
| `dataset_prep.py` | Caricamento e formattazione degli 8 dataset (prompt, risposte di riferimento, numero di istanze, stringhe di arresto) e le metriche di correttezza scritte per il progetto (`MCQAccuracyMetric`, `GSM8kAccuracyMetric`). |
| `analysis_lib.py` | Le metriche, in un'unica implementazione condivisa da `main.py` e dagli script offline: PRR (normalizzato e grezzo, pareggi in valore atteso), intervalli bootstrap, errori fra le risposte più confidenti, Kendall tau con intervallo, severità a parità di difficoltà, aggregazione fra dataset, famiglie di metodi. Dipende solo da numpy, pandas e scipy. |
| `batched_sampling.py` | Generatore dei K campioni in una sola chiamata, senza conservare stati nascosti e logit completi. Sostituisce quello di lm-polygraph (più lento e con molta più memoria); stesse statistiche. |
| `make_figures.py` | Rigenera **tutte le figure e le tabelle-immagine** dai CSV, in pochi secondi e senza GPU. |
| `paired_comparisons.py` | Test bootstrap appaiati: quali metodi sono statisticamente indistinguibili dal migliore. Senza GPU. |
| `recompute_stats.py` | Ricalcola le statistiche di una cartella di risultati già esistente con le metriche corrette il 29/09 (PRR normalizzato, nuove colonne, Kendall per dataset). Senza GPU. |
| `paper_replica.py` | Replica esatta di Vashurin et al. (Tabelle 7 e 10) con Mistral 7B v0.2 base e Instruct, sui dataset e con i prompt pubblicati da lm-polygraph. Vedi la ricetta "Replica del paper" nella sezione 6. |
| `paper_replica_configs.json` | Impostazioni ufficiali della replica (dataset e prompt, max_new_tokens, stringhe di arresto, stimatori, post-elaborazione delle risposte), estratte dai file di configurazione di lm-polygraph al commit 32cdf4a. |
| `paper_replica_processing.py` | Funzioni di post-elaborazione delle risposte del protocollo ufficiale, copiate senza modifiche da lm-polygraph. |
| `paper_reference_prr.csv` | PRR delle Tabelle 7 (white-box, Mistral 7B v0.2) e 10 (black-box, Mistral 7B v0.2 Instruct) del paper, con deviazione standard: il riferimento della replica. |
| `drop_models_from_checkpoints.py` | Toglie modelli o dataset dai checkpoint, per rieseguire solo quelli. |
| `tests/` | Test della pipeline senza GPU, con un modello minuscolo e dataset finti: `bash tests/run_tests.sh`. Solo il test della replica usa la rete (scarica i dataset di lm-polygraph). |
| `preflight.sh` | Controlli di sola lettura prima di un run lungo (codice, token, immagine Docker, GPU, disco). |
| `sbatch_script.sh`, `run_docker.sh`, `train.sh` | Catena di lancio su cluster SLURM: `sbatch` → container Docker → `main.py` (oppure lo script indicato come primo argomento, es. `paper_replica.py`). |
| `create_docker_image.sh`, `build/` | Costruzione dell'immagine Docker (`Dockerfile` e file dei requirements). |
| `sync_results.sh` | Allinea la cartella `results/` fra i due nodi del cluster (faretra e moro232). |
| `results_store.py` | Modulo per checkpoint con semantica di sovrascrittura per riga. **Non è usato** dalla pipeline attuale: è una base per un'eventuale riorganizzazione dei checkpoint. |
| `METRICHE.md` | Spiegazione di tutte le metriche: accuracy, AlignScore, PRR, pareggi, bootstrap, confronti appaiati, errori fra le risposte più confidenti, severità a parità di difficoltà, parse-failure rate, Kendall tau, costo marginale e pieno. |

---

## 2. Requisiti

- **GPU NVIDIA con 24 GB** (sviluppato su RTX 3090). I modelli da 4B girano in bf16
  (circa 9 GB), Mistral-7B in bf16 circa 15 GB; sulla stessa GPU girano anche il
  modello NLI (DeBERTa-large), il cross-encoder e AlignScore.
- **Token Hugging Face** con la licenza accettata per i modelli *gated*:
  [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) e
  [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it).
  Senza, quei due modelli falliscono al caricamento.
- **Spazio su disco**: circa 40 GB per i modelli, più alcuni GB per i dataset e i
  risultati.
- **Docker** con NVIDIA Container Toolkit (consigliato), oppure Python 3.11.

Versioni principali (fissate in `build/`): torch 2.6.0 (CUDA 12.4),
transformers 4.57.3, **lm-polygraph 0.7.0**, datasets 4.3.0. lm-polygraph è
fissato perché la pipeline dipende dal suo comportamento (stringhe di arresto,
gestione degli errori di `UEManager`); con altre versioni i risultati possono
cambiare.

---

## 3. Installazione

### Con Docker (consigliato)

```bash
git clone <url-del-repository> mental_chatbot
cd mental_chatbot
git checkout filo/benchmark-uq
bash create_docker_image.sh        # crea l'immagine "mental-chatbot-image"
```

`create_docker_image.sh` contiene il percorso del repository sul cluster
(`/home/patrignani/mental_chatbot`): su un'altra macchina va cambiato, oppure si
lancia direttamente `docker build -f build/Dockerfile -t mental-chatbot-image .`
dalla cartella del repository.

Il codice **non è copiato dentro l'immagine**: `run_docker.sh` monta la cartella
del repository in `/workspace`. Una modifica al codice non richiede quindi di
ricostruire l'immagine; serve solo se cambiano i file in `build/`.

I percorsi montati sono definiti in `run_docker.sh`:

| Sull'host | Nel container | Contenuto |
|---|---|---|
| `/home/patrignani/mental_chatbot` | `/workspace` | codice e risultati |
| `/llms` | `/llms` | cache dei modelli Hugging Face |

Su un'altra macchina vanno adattate le due variabili `PHYS_DIR` e `LLM_CACHE_DIR`.

### Senza Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r build/requirements.txt
pip install --no-deps -r build/requirements-nodeps.txt   # facoltativo: serve solo al fine-tuning
pip install -r build/requirements-benchmark.txt
python -m spacy download en_core_web_sm
```

### Token Hugging Face

Il token si passa come variabile d'ambiente e non va mai scritto nei file del
repository. Il modo più comodo è un file privato:

```bash
echo 'export HF_TOKEN="hf_..."' > ~/.uq_env
source ~/.uq_env          # da ripetere in ogni nuova shell, prima di lanciare
```

`sbatch` copia le variabili della shell al momento del lancio: il `source` va
fatto **prima** di `sbatch_script.sh`. Facoltativo: `WANDB_API_KEY`, per il
logging su Weights & Biases.

---

## 4. Come si esegue

### Prima di un run lungo: `preflight.sh`

```bash
source ~/.uq_env
bash preflight.sh
```

Controlla, in sola lettura: branch e commit allineati al remote, nessuna modifica
locale, token valido e accesso ai modelli gated, modelli già in cache, immagine
Docker aggiornata e con lm-polygraph 0.7.0, memoria GPU libera, spazio su disco,
limite di tempo delle partizioni SLURM, cartella `results/` senza dati di versioni
vecchie. Termina con `TUTTO OK` o con l'elenco preciso di cosa sistemare.

### Su cluster SLURM

```bash
source ~/.uq_env
SBATCH_NODE=faretra bash sbatch_script.sh [argomenti di main.py]
```

- Gli argomenti vengono passati tali e quali a `main.py` (vedi sezione 5).
- `SBATCH_NODE` fissa il nodo. **Va sempre fissato**: faretra e moro232 non
  condividono la home, quindi i checkpoint stanno sul disco del nodo su cui ha
  girato il job, e un job ripreso sull'altro nodo ripartirebbe da zero.
- Il job chiede una RTX 3090 (`--gpus=nvidia_geforce_rtx_3090:1`).
- Stato dei job: `squeue -u $USER`. Log: `slurm-<id>.out` nella cartella da cui si
  è lanciato.

Per lanciare e poter chiudere la connessione anche mentre si ricostruisce
l'immagine:

```bash
nohup bash -c "bash create_docker_image.sh && bash preflight.sh && \
  SBATCH_NODE=faretra bash sbatch_script.sh <argomenti>" > lancio.log 2>&1 &
```

Il run parte solo se la build riesce e il preflight stampa `TUTTO OK`.

### Direttamente, senza SLURM

Con Docker (sostituire `0` con la GPU da usare):

```bash
CUDA_VISIBLE_DEVICES=0 bash run_docker.sh [argomenti di main.py]
```

Senza Docker, dall'ambiente virtuale:

```bash
python3.11 main.py --results_dir ./results --cache_dir ~/.cache/huggingface \
                   --datasets_cache_dir ./hf_datasets_cache [altri argomenti]
```

### Ripresa dopo un'interruzione

Ogni cella modello × dataset viene eseguita a **blocchi di 100 istanze** (vedi
`--chunk_size`) e ogni blocco completato viene salvato in `<results_dir>/chunks/`.
Se il job si interrompe (limite di tempo, nodo perso, errore) basta **rilanciare lo
stesso identico comando**: le celle complete e i blocchi già calcolati vengono
ripresi dai checkpoint, e si perde al massimo il blocco in corso.

Al primo lancio viene salvata un'**impronta** del codice (`main.py`,
`dataset_prep.py`, `batched_sampling.py`, `analysis_lib.py`) e delle impostazioni
che cambiano i risultati (`run_fingerprint.json`). Una ripresa con un'impronta
diversa viene **rifiutata** con un messaggio che dice quali file o impostazioni
sono cambiati: riprendere mescolerebbe risultati di due versioni nelle stesse
figure. Le alternative: una `--results_dir` nuova, `--no_resume` (ricalcola
tutto), oppure `--force_resume` se la modifica non tocca generazioni e metriche
(ad esempio un commento). Le impostazioni che non cambiano i risultati
(`--models`, `--datasets`, le sezioni extra) si possono variare liberamente fra
un lancio e l'altro.

Due protezioni in più contro i risultati mescolati:

- `--no_resume` sposta **subito** tutti i risultati già presenti nella cartella
  (CSV, figure e `chunks/`) in `<results_dir>/_superati/<data-ora>/`, senza
  cancellare niente. Prima i checkpoint vecchi venivano scartati solo quando il
  run raggiungeva la loro cella: se il run si interrompeva, o girava solo su
  alcuni `--models`, il rilancio successivo riprendeva come validi i checkpoint
  della versione precedente per tutte le celle non ancora toccate.
- ogni cartella di blocchi porta l'impronta del codice che l'ha prodotta
  (`chunks/.../run_fingerprint.txt`): blocchi con un'impronta diversa o senza
  impronta vengono spostati in `_superati/` e la cella riparte da zero.

---

## 5. Argomenti di `main.py`

### Selezione del lavoro

| Argomento | Default | Cosa fa |
|---|---|---|
| `--models M [M ...]` | tutti | Esegue solo questi modelli: `LFM2-350M`, `LFM2-1.2B`, `MedGemma-4B-it`, `Gemma3-4B-it`, `Mistral-7B-it`. Vale per la pipeline principale e per tutte le sezioni extra. Le celle degli altri modelli già presenti nella cartella restano e finiscono nelle figure. |
| `--datasets D [D ...]` | tutti | Esegue solo questi dataset della pipeline principale: `CoQA`, `TriviaQA`, `MMLU`, `GSM8k`. Vale anche per le sezioni verbalized e quantizzazione, **non** per la griglia di severità. Serve a dividere il lavoro in più run nella stessa cartella. |
| `--n_test_samples N` | per dataset (sezione 9) | Sovrascrive il numero di istanze di **tutti** i dataset. Serve per i test rapidi. |
| `--run_severity_grid` | no | Esegue anche la griglia severità × formato: MedQAbstain-LT, MedQAbstain-Safe, MedicationQA, MedQuAD. |
| `--run_verbalized` | no | Esegue anche i metodi *verbalized* (il modello dichiara la propria confidenza, in forma numerica e verbale) su CoQA, TriviaQA e MMLU. |
| `--run_quant_comparison` | no | Esegue anche il confronto 4 bit contro bf16 sullo stesso modello, sui dataset principali. |
| `--quant_compare_model M` | `LFM2-1.2B` | Modello del confronto sulla quantizzazione. I due Gemma non sono utilizzabili: a 4 bit producono logit NaN. Con `Mistral-7B-it` si confronta l'ancora. |

### Esecuzione

| Argomento | Default | Cosa fa |
|---|---|---|
| `--batch_size N` | `1` | Batch di generazione. Con 1 la memoria usata è minima e non c'è padding fra sequenze di lunghezza diversa. Per i modelli pesanti il batch viene comunque ridotto in automatico. |
| `--chunk_size N` | `100` | Istanze per blocco con checkpoint dentro ogni cella. `0` disattiva i blocchi. I risultati sono identici con o senza blocchi. |
| `--no_resume` | no | Riesegue tutto da zero; i risultati già presenti vengono spostati in `<results_dir>/_superati/<data-ora>/`, non cancellati. |
| `--force_resume` | no | Riprende dai checkpoint anche se l'impronta del codice è cambiata (vedi "Ripresa dopo un'interruzione"). |
| `--anchor_precision {bf16,4bit}` | `bf16` | Precisione dell'ancora Mistral-7B. `bf16` è la precisione del paper; `4bit` riproduce le run fino al 29/09. |
| `--sampler {batched,library}` | `batched` | Generatore dei K campioni: `batched_sampling.py` (una chiamata, memoria costante) oppure quello di lm-polygraph, per confronto. |
| `--max_rejection F` | `0.5` | Frazione massima di risposte scartate nel calcolo del PRR, come nel paper. |
| `--n_bootstrap N` | `1000` | Ricampionamenti bootstrap per gli intervalli di confidenza. Costa solo CPU; abbassarlo accelera i test. |
| `--verbalized_max_new_tokens N` | `40` | Token generati nella sezione verbalized: devono bastare per la risposta **e** per la riga di confidenza. |

### Percorsi

| Argomento | Default | Cosa fa |
|---|---|---|
| `--results_dir DIR` | `$RESULTS_DIR` oppure `/workspace/results` | Dove scrivere risultati, figure e checkpoint. |
| `--cache_dir DIR` | `$HF_HOME` oppure `/llms` | Cache dei modelli Hugging Face. |
| `--datasets_cache_dir DIR` | `$HF_DATASETS_CACHE` oppure `/workspace/hf_datasets_cache` | Cache dei dataset. |

Parametri fissi nel codice (cambiarli significa cambiare il protocollo):
seed `3407` (`SEED` in `main.py`), prompt, numero di istanze, stringhe di arresto
e numero massimo di token per dataset (`DATASETS` e `SEVERITY_DATASETS` in
`dataset_prep.py`), modelli che non tollerano la quantizzazione
(`CANNOT_QUANTIZE_MODELS` in `main.py`), batch del modello NLI
(`DEBERTA_BATCH_SIZE_*`), fase di costo di ogni calcolatore (`PHASE_OF_CALCULATOR`).
Le condizioni effettive di ogni cella (precisione, attenzione, formato del prompt,
token, arresti, batch, campionatore, versioni delle librerie, GPU) vengono scritte
in `run_conditions.csv`.

---

## 6. Ricette: cosa lanciare per ogni esperimento

Negli esempi `SB` sta per `SBATCH_NODE=faretra bash sbatch_script.sh`.

**Test rapido dell'intera pipeline** (pochi minuti; da fare dopo ogni modifica al
codice o all'immagine, in una cartella separata):

```bash
SB --results_dir /workspace/results_smoke --n_test_samples 6 --chunk_size 3 \
   --run_severity_grid --run_verbalized --run_quant_comparison
```

Nel log controllare che non ci siano righe `!!!` e che compaia
`stringhe di arresto`; in `results_smoke/sample_generations.csv` e nella colonna
`greedy_text` di `per_instance_scores.csv` le risposte devono essere in inglese e
fermarsi alla fine della riga; in `run_conditions.csv` tutti i modelli devono avere
`prompt_format = chat_template`. Dopo le modifiche del 29/09 questo test va fatto
**prima** di qualunque run lungo, anche per verificare che Mistral in bf16 stia nei
24 GB.

**Esperimento completo, in un unico run:**

```bash
SB --run_severity_grid --run_verbalized --run_quant_comparison
```

**Esperimento completo, diviso in due run** (GSM8k da solo pesa circa due terzi
del tempo totale):

```bash
SB --datasets CoQA TriviaQA MMLU --run_severity_grid --run_verbalized --run_quant_comparison
# a run finito, sullo STESSO nodo e nella STESSA cartella:
SB --datasets GSM8k --run_quant_comparison
```

Il secondo run riprende dai checkpoint tutto ciò che il primo ha calcolato: i file
finali e le figure contengono tutti i dataset.

**Solo la pipeline principale** (replica delle Figure 2 e 3 del paper, Kendall tau,
costi):

```bash
SB
```

**Solo alcuni modelli**, per esempio per rifare un modello fallito:

```bash
SB --models Mistral-7B-it --run_severity_grid
```

**Solo la griglia clinica**, senza rifare la pipeline principale: non esiste un
flag dedicato; si lancia con `--run_severity_grid` nella cartella di un run
completo, e la pipeline principale viene ripresa interamente dai checkpoint.

**Rifare da zero un modello o un dataset** mantenendo il resto: vedi
`drop_models_from_checkpoints.py` nella sezione 8.

**Rileggere una run vecchia con le metriche corrette** (senza GPU):

```bash
python3.11 recompute_stats.py results_vecchi
python3.11 paired_comparisons.py results_vecchi
python3.11 paired_comparisons.py results_vecchi --per_instance_file results_severity_grid_per_instance.csv
python3.11 make_figures.py results_vecchi
```

**Misurare solo i costi** (bastano poche istanze):

```bash
SB --results_dir /workspace/results_costi --datasets CoQA TriviaQA MMLU --n_test_samples 100
```

**Figure e test appaiati** vengono prodotti in automatico alla fine di `main.py`,
che lancia nell'ordine `paired_comparisons.py` (sulla pipeline principale e, se c'è,
sulla griglia clinica) e `make_figures.py`. Per rigenerarli a mano, ad esempio dopo
aver copiato i risultati su un portatile con pandas e matplotlib (senza GPU):

```bash
python3.11 paired_comparisons.py results
python3.11 paired_comparisons.py results --per_instance_file results_severity_grid_per_instance.csv
python3.11 make_figures.py results
```

L'ordine conta: `fig_ties_vs_scale.png` legge i riepiloghi di `paired_comparisons.py`.

**Replica del paper** (Vashurin et al., Tabelle 7 e 10). Serve a verificare che la
pipeline riproduca i risultati del paper nel suo stesso setting, prima di usarla
sui modelli piccoli. Gira in una cartella separata (default
`/workspace/results_paper_replica`) e non tocca i risultati della pipeline principale.

```bash
SB paper_replica.py --parts whitebox                 # Tabella 7: Mistral 7B v0.2 base, 4 dataset
SB paper_replica.py --parts blackbox verbalized      # Tabella 10: Mistral 7B v0.2 Instruct, 3 dataset
SB paper_replica.py --parts whitebox --datasets GSM8k   # un solo dataset
SB paper_replica.py --parts whitebox --n_test_samples 20 --results_dir /workspace/results_replica_smoke  # prova breve
```

Cosa replica, e da dove viene ogni scelta:

- **modelli:** `mistral-community/Mistral-7B-v0.2`, l'unica copia su Hugging Face
  della v0.2 base (mistralai non l'ha pubblicata), per la parte white-box;
  `mistralai/Mistral-7B-Instruct-v0.2` per la black-box. Nessuno dei due è gated.
  Precisione bf16 (`--precision 4bit` solo in caso di memoria insufficiente);
- **domande:** i dataset pubblicati da lm-polygraph (`LM-Polygraph/coqa`,
  `triviaqa`, `mmlu`, `gsm8k`), con i prompt già costruiti. White-box:
  sottoinsieme `continuation` (5-shot per TriviaQA/MMLU/GSM8k, conversazione
  precedente per CoQA). Black-box: `empirical_baselines` per i metodi basati
  sui campioni, poi un sottoinsieme per ciascuno dei 6 metodi verbalized
  (`--verbalized_variants` per sceglierne alcuni);
- **numerosità:** come nel paper, 2.000 domande per dataset tranne MMLU (100 per
  materia, 5.700) e GSM8k (1.319, tutto il test set). Le 2.000 sono scelte come
  fa lm-polygraph (`np.random.seed(1)` + `np.random.choice`): il test verifica che
  siano identiche, una per una, a quelle della libreria;
- **generazione, stimatori, qualità:** max_new_tokens, stringhe di arresto,
  stimatori e loro parametri e post-elaborazione delle risposte vengono dai file
  di configurazione ufficiali (`paper_replica_configs.json`). Qualità: AlignScore
  per CoQA e TriviaQA (massimo sugli alias), accuracy per MMLU e GSM8k.

Alla fine `paper_replica_comparison.csv` confronta ogni metodo con il paper
(PRR, intervallo al 95%, valore e deviazione standard del paper, scarto), e
`paper_replica_summary.csv` riassume per dataset lo scarto medio, la quota di
metodi il cui valore del paper cade nel nostro intervallo e il tau di Kendall
fra le due classifiche. Figure: `fig_paper_replica_whitebox.png`,
`fig_paper_replica_blackbox.png`. Per rifare solo confronto e figure:
`python3.11 paper_replica.py --only_compare --results_dir <cartella>`.

Differenze note rispetto al paper: precisione bf16 (il paper non la dichiara),
campionatore in batch (stesse statistiche della libreria, vedi
`tests/test_batched_sampling.py`), metodi density-based esclusi in tutto il lavoro.
Il modello Mistral-7B-it della pipeline principale resta il riferimento di scala
per il Kendall tau dei modelli piccoli (stessi prompt dei modelli piccoli), ma
non è confrontabile direttamente con le tabelle del paper.

Durata stimata (da misurare con la prova breve): la parte white-box sono circa
11.000 domande con 10 campioni ciascuna, circa 1-2 giorni su una 3090; la parte
black-box 7 varianti di prompt su 9.700 domande, di cui 6 solo con generazione
greedy, circa un giorno. Le parti si possono lanciare su nodi diversi.

### Tempi indicativi (RTX 3090, 5 modelli)

| Parte | Istanze | Tempo con il codice fino al 29/09 |
|---|---|---|
| CoQA, TriviaQA, MMLU | 500, 1000, 1000 | circa 25 ore |
| GSM8k | 500 | circa 70 ore |
| Griglia di severità | 556 + 556 + tutte + 1000 | circa 1-2 giorni |
| Quantizzazione (LFM2-1.2B, 2 varianti) | come la pipeline principale | circa 10 ore |
| Verbalized | come la pipeline principale, senza GSM8k | alcune ore |

Stime dai run precedenti al 29/09. Con il campionatore in batch, i batch NLI più
grandi e l'attenzione SDPA sui Gemma i tempi dovrebbero scendere nettamente, ma
non sono ancora stati misurati; in senso opposto pesano Mistral in bf16, 256 token
su GSM8k (erano 200) e AlignScore su tutti gli alias di TriviaQA. Il test rapido
stampa i tempi per fase (`[fase ...]`) e li salva in `phase_timings.csv`: da lì si
stima la durata di un run completo.

---

## 7. Output prodotti

Tutto viene scritto in `--results_dir`. I CSV sono la fonte dei dati; le figure si
rigenerano da lì con `make_figures.py`.

### Pipeline principale

| File | Contenuto |
|---|---|
| `results_final.csv` | PRR di ogni metodo per ogni modello × dataset, nel formato di lm-polygraph (colonne `model, key, estimator, ue_metric, value, dataset`). `ue_metric = prr_0.5_normalized` è il PRR del paper (0 = casuale, 1 = oracolo), `prr_0.5` la sola area; entrambi ricalcolati dagli array per istanza con i pareggi in valore atteso. Il valore originale della libreria è in `value_lmpolygraph`. |
| `results_partial.csv` | Checkpoint di `results_final.csv`, aggiornato dopo ogni cella. |
| `results_paper_mapped.csv` | Righe del PRR del paper con le etichette dei metodi usate nel paper, intervalli di confidenza, accuracy, errori fra le risposte più confidenti. È la base delle Figure A e B. |
| `instance_level_stats.csv` | Per metodo × modello × dataset: `prr` (PRR del paper) con intervallo bootstrap al 95%, `prr_raw` (area grezza, per confronto), accuracy di base, numero di istanze, frazione di punteggi NaN, `distinct_fraction` (punteggi distinti), `error_rate_top10` ed `error_rate_overall`, il vecchio `silent_failure_rate`, parse-failure rate della risposta, istanze saltate. |
| `per_instance_scores.csv` | **Il dato grezzo**: per ogni domanda l'indice dell'istanza (`instance_index`, lo stesso per tutti i modelli), il testo generato (`greedy_text`), la qualità della risposta e il punteggio di incertezza di ogni metodo. Tutte le analisi offline ripartono da qui. |
| `accuracy_table.csv` | Accuracy (o AlignScore) per modello × dataset. |
| `table6_style_<modello>.csv` | Tabella nello stile della Tabella 6 del paper, per modello. |
| `rank_transfer_kendall_tau.csv` | Kendall tau fra la classifica dei metodi di ogni modello e quella di Mistral-7B, per dataset, con intervallo bootstrap appaiato sulle domande. |
| `estimator_timings.csv` | Tempi per metodo × modello × dataset: costo marginale (solo il calcolo del metodo), costo pieno standalone (tutti i calcolatori da cui dipende), fasi e modelli ausiliari usati, elenco dei calcolatori, memoria di picco della cella. |
| `phase_timings.csv` | Tempo per fase (generazione greedy, K campioni, NLI, cross-encoder, forward extra, metrica di qualità, ...) per modello × dataset. |
| `run_conditions.csv` | Scheda delle condizioni di esecuzione di ogni cella (vedi sezione 5). |
| `run_fingerprint.json` | Impronta del codice e delle impostazioni (vedi "Ripresa dopo un'interruzione"). |
| `estimator_timing_table.csv`, `estimator_cost_table.csv` | Riepiloghi dei tempi per metodo. |
| `excluded_methods.md` | Metodi del paper non inclusi e perché. |
| `sample_generations.csv` | Campione di generazioni grezze con la lettera estratta e quella attesa (task a scelta multipla): serve a controllare a occhio che le risposte siano estratte bene. |

### Sezioni extra

| File | Sezione |
|---|---|
| `results_severity_grid*.csv`, `accuracy_table_severity_grid.csv`, `error_rate_most_confident.csv`, `severity_matched_difficulty.csv` | Griglia di severità (stessa struttura dei file principali: `_instance_stats`, `_per_instance`, `_mapped`) e confronto degli strati a parità di difficoltà. |
| `results_verbalized_{numeric,linguistic}*.csv`, `accuracy_table_verbalized_*.csv`, `parse_failure_rate_*.csv` | Metodi verbalized. Il parse-failure rate è la frazione di risposte in cui la confidenza non è estraibile. |
| `results_quant_comparison*.csv`, `accuracy_table_quant_comparison.csv` | Confronto 4 bit contro bf16. |

### Checkpoint a blocchi

`chunks/<sezione>/<modello>__<dataset>__n<istanze>_c<blocco>/` contiene, per ogni
blocco, i file `chunkNNNN_{prr,timing,per_instance,meta}.csv` ed eventualmente
`chunkNNNN_skipped.csv`, con le istanze saltate perché fallite anche da sole. Si
possono cancellare a run concluso: servono solo alla ripresa.

### Figure (prodotte da `make_figures.py`)

| Figura | Cosa mostra |
|---|---|
| `fig_accuracy.png` | Accuracy per modello e dataset, con il livello del caso sui task a scelta multipla. |
| `fig_accuracy_vs_prr.png` | Area grezza e PRR del paper contro l'accuracy di base: mostra perché la prima misurava l'accuracy. |
| `fig_a_white_box.png`, `fig_b_reflexive.png` | Repliche delle Figure 2 e 3 del paper, con intervalli di confidenza. |
| `fig_prr_vs_scale_by_family.png` | PRR per famiglia di metodi al variare della scala. |
| `fig_rank_transfer.png` | Kendall tau della classifica dei metodi contro il modello da 7B, per dataset, con intervalli. |
| `fig_ties_vs_scale.png` | Quanti metodi sono indistinguibili dal migliore, per scala. Richiede `paired_comparisons.py`. |
| `fig_severity_grid.png` | Griglia 2×2 severità × formato. |
| `fig_severity_matched.png` | Gli strati di severità confrontati a parità di difficoltà delle domande. |
| `fig_phase_timings.png` | Dove va il tempo: fasi del calcolo per modello e dataset. |
| `fig_quant_comparison.png` | 4 bit contro bf16, per metodo, con intervalli di confidenza. |
| `fig_timing_full_cost.png`, `estimator_timing_chart.png` | Costo pieno contro costo marginale; costo marginale per modello. |
| `fig_pareto_cost_quality.png` | Frontiera di Pareto costo contro PRR, con intervalli di confidenza. |
| `fig_verbalized_accuracy_cost.png` | Quanto la richiesta di confidenza peggiora le risposte. |
| `fig_verbalized_{numeric,linguistic}.png` | PRR dei metodi verbalized per modello, con intervalli di confidenza e parse-failure rate. |
| `fig_paper_replica_{whitebox,blackbox}.png` | Solo nella cartella della replica: PRR ottenuto contro le Tabelle 7 e 10 del paper, metodo per metodo, con tau di Kendall e scarto medio per dataset. |
| `fig_tabella_*.png` | Tabelle in forma di immagine: accuracy, PRR affiancato all'accuracy (con l'intervallo del metodo migliore), costi, parse-failure rate, errori fra le risposte più confidenti per strato di severità. |

---

## 8. Script di analisi e di supporto

Tutti tranne `preflight.sh` e `sync_results.sh` girano senza GPU. Nel container si
lanciano con `python3.11`; fuori basta un ambiente con numpy, pandas, matplotlib
e scipy.

### `make_figures.py`

```bash
python3.11 make_figures.py <results_dir>
```

Rigenera tutte le figure della sezione 7 dai CSV. Se un CSV manca, la figura
corrispondente viene saltata con un messaggio. Ogni figura è disegnata in modo
isolato: se una fallisce (ad esempio per un CSV di una versione vecchia), le altre
vengono prodotte comunque e alla fine lo script elenca quelle mancanti ed esce con
codice 1. Un modello non presente in `MODEL_PARAMS_B` compare in coda nelle figure
invece di essere scartato. Ogni figura ha in fondo una nota su come leggerla e su
quando non fidarsi del dato.

### `paired_comparisons.py`

```bash
python3.11 paired_comparisons.py <results_dir> [--per_instance_file F] [--out F] \
                                 [--n_resamples 1000] [--max_rejection 0.5]
```

Per ogni modello × dataset ricampiona le domande e, su ogni ricampionamento,
ricalcola i PRR di tutti i metodi e guarda chi è il migliore **in quel
ricampionamento**. Un metodo è *peggiore* del migliore solo se risulta il migliore
in meno di 0.05/(m−1) dei ricampionamenti (Bonferroni sugli m−1 confronti),
altrimenti *indistinguibile*; *non testabile* se il PRR non è calcolabile nella
maggior parte dei ricampionamenti. Scrive `<input>_paired_comparisons.csv` e un
riepilogo `_paired_comparisons_summary.csv` (metodi equivalenti al migliore,
peggiori, non testabili). Prima di tutto esegue i controlli di sanità del PRR
(casuale ≈ 0, oracolo = 1). Per la griglia clinica:
`--per_instance_file results_severity_grid_per_instance.csv`.

### `recompute_stats.py`

```bash
python3.11 recompute_stats.py <results_dir> [--n_bootstrap 1000] [--n_resamples_kendall 200]
```

Per cartelle prodotte **prima** del 29/09: ricalcola dai punteggi per istanza le
statistiche di tutte le sezioni con le metriche corrette (PRR del paper con
intervalli, area grezza, punteggi distinti, errori fra le risposte più confidenti),
riscrive i file `*_mapped.csv` letti dalle figure e il Kendall tau per dataset. I
file originali vengono copiati una volta in `<results_dir>/_pre_ricalcolo/`. Dopo,
lanciare `paired_comparisons.py` e `make_figures.py`. I difetti nelle generazioni
(prompt in italiano, doppio BOS, LFM2 senza chat template, lettera "a" nelle MCQ,
un solo riferimento su TriviaQA) non si correggono così: servono run nuove.
Sostituisce `recompute_silent_failure.py`.

### `drop_models_from_checkpoints.py`

```bash
python3.11 drop_models_from_checkpoints.py <results_dir> Gemma3-4B-it                 # tutte le celle del modello
python3.11 drop_models_from_checkpoints.py <results_dir> --datasets MMLU MedQAbstain-LT  # tutte le celle dei dataset
python3.11 drop_models_from_checkpoints.py <results_dir> Gemma3-4B-it --datasets MMLU  # solo la combinazione
```

Toglie le celle indicate dai checkpoint (con copia `.bak` di ogni file) e sposta
i loro checkpoint a blocchi in `chunks_rimossi/`, così che il run successivo, lanciato
**senza** `--no_resume`, rifaccia solo quelle celle e riprenda tutte le altre.
Da usare quando cambia la configurazione di un singolo modello o la metrica di un
task.

### `tests/run_tests.sh`

```bash
bash tests/run_tests.sh            # nel container; fuori: PY=python bash tests/run_tests.sh
```

Esegue, in circa dieci minuti su CPU, i test della pipeline con un modello Llama
minuscolo creato al volo e dataset finti: stringhe di arresto, esecuzione a
blocchi e ripresa, errori di singola istanza e OOM, costi e dipendenze dei metodi,
campionatore in batch equivalente a quello di lm-polygraph, chat template e un
solo BOS, prompt inglesi dai template ufficiali, estrazione della lettera MCQ,
PRR normalizzato (casuale ≈ 0, oracolo = 1), confronti appaiati, impronta del
codice, orchestrazione completa di `main.py` con tutte le sezioni e le figure,
confronto quantizzazione con due modelli nella stessa cartella (sulla CPU la
variante 4-bit è caricata non quantizzata: si verifica la logica della sezione,
non l'effetto della quantizzazione).
Va rilanciato dopo ogni modifica al codice.

### `preflight.sh`

Vedi sezione 4. Percorsi personalizzabili con `REPO=... LLM_CACHE=... bash preflight.sh`.

### `sync_results.sh`

```bash
bash sync_results.sh check        # confronta le due copie, non tocca nulla
bash sync_results.sh from-moro    # moro232 -> faretra, dopo un job su moro232
bash sync_results.sh to-moro      # faretra -> moro232, prima di un job su moro232
```

Da lanciare sempre da faretra. Sincronizza solo la cartella `results/`; la copia
sostituita viene conservata in `results.prev/`.

---

## 9. Modelli, dataset e metodi

### Modelli

| Nome nel codice | Checkpoint | Parametri | Precisione |
|---|---|---|---|
| `LFM2-350M` | `LiquidAI/LFM2-350M` | 0.35B | 4 bit (NF4) |
| `LFM2-1.2B` | `LiquidAI/LFM2-1.2B` | 1.2B | 4 bit (NF4); anche bf16 nel confronto sulla quantizzazione |
| `Gemma3-4B-it` | `google/gemma-3-4b-it` | 4B | bf16 |
| `MedGemma-4B-it` | `google/medgemma-4b-it` | 4B | bf16 |
| `Mistral-7B-it` | `mistralai/Mistral-7B-Instruct-v0.2` | 7B | bf16 (`--anchor_precision 4bit` per NF4) |

I due Gemma girano in bf16 perché a 4 bit producono logit NaN. **Tutti** i modelli
ricevono il prompt dentro il proprio chat template (anche gli LFM2, che sono
modelli chat: fino al 29/09 ricevevano un prompt a completamento semplice), con un
solo token BOS. L'attenzione è SDPA per tutti; eager solo per un modello la cui
configurazione dichiara `attn_logit_softcapping` (Gemma 2, non Gemma 3).
Mistral-7B è l'**ancora di replica**: è uno dei modelli del paper, e serve a
verificare che la pipeline riproduca i risultati nel regime di scala originale.

### Dataset

| Dataset | Fonte HF | Tipo | Metrica | Istanze | Arresto della generazione |
|---|---|---|---|---|---|
| CoQA | `stanfordnlp/coqa` (validation) | QA conversazionale, risposta breve | AlignScore | 500 (tutte) | primo a capo |
| TriviaQA | `mandarjoshi/trivia_qa` rc.nocontext (validation) | QA, risposta breve, 5-shot | AlignScore, massimo sugli alias | 1000 | primo a capo |
| MMLU | `cais/mmlu` (test) | scelta multipla, 5-shot per subject | Accuracy | 1000, stratificate sui 57 subject | primo a capo |
| GSM8k | `openai/gsm8k` (test) | problemi aritmetici, 8-shot chain-of-thought | Accuracy sul numero dopo "The answer is" | 500 | inizio di un nuovo "Question:" |
| MedQAbstain-LT | `disi-unibo-nlp/MedQAbstain` (LT, fonte medqa_4opt) | scelta multipla, severità **alta** | Accuracy | 556 | primo a capo |
| MedQAbstain-Safe | `disi-unibo-nlp/MedQAbstain` (Safe, fonte medqa_4opt) | scelta multipla, severità **bassa** | Accuracy | 556 (tutte) | primo a capo |
| MedicationQA | `truehealth/medicationqa` | testo libero, severità **alta** | AlignScore | tutte dopo i filtri | inizio di una nuova "Question:" |
| MedQuAD | `lavita/MedQuAD` (solo domande informative) | testo libero, severità **bassa** | AlignScore | 1000 | inizio di una nuova "Question:" |

- I primi quattro sono i dataset del paper; gli altri quattro formano la griglia
  **severità × formato**, in cui le due celle di ogni colonna condividono formato e
  metrica, così che il confronto isoli la severità.
- MedicationQA e MedQuAD usano gli stessi filtri sulla lunghezza dei riferimenti
  (80-600 caratteri) e lo stesso numero massimo di token.
- Per MedQAbstain si usano 556 istanze per entrambe le celle, cioè tutte quelle
  della fonte a 4 opzioni disponibili nella cella Safe, così che le due celle
  abbiano la stessa dimensione.
- Il campionamento delle istanze è fissato dal seed 3407.
- **Prompt**: in inglese. Per i quattro dataset del paper sono i template ufficiali
  di lm-polygraph per i modelli instruct (sottoinsieme `simple_instruct` in
  `dataset_builders/builders/` del repository IINemo/lm-polygraph), trascritti
  carattere per carattere; le celle MCQ cliniche usano lo stesso template di MMLU,
  quelle a testo libero "Question: ... / Answer:".
- **Differenze dal protocollo del paper**, da dichiarare: CoQA con una sola
  domanda per conversazione (l'ultima); MMLU con 1000 domande stratificate invece
  di fino a 100 per subject; 12 token invece di 3 sui task a scelta multipla, con
  estrazione della lettera invece del confronto esatto; 256 token su GSM8k come
  nel protocollo, ma con arresto su un nuovo "Question:"; 5 esempi few-shot di
  TriviaQA scelti con il nostro seed.

### Metodi UQ

27 metodi di lm-polygraph, mappati sulle Figure 2 e 3 del paper (`PAPER_METHODS`
in `main.py`): information-based (MSP, Perplexity, Mean Token Entropy, CCP, PMI,
Conditional PMI, Fisher-Rao, Renyi, TokenSAR), basati sulla diversità dei campioni
(Semantic Entropy, SAR, SentenceSAR, Monte Carlo Sequence Entropy e versione
normalizzata, Lexical Similarity, EigValLaplacian, DegMat, Eccentricity, NumSet)
e riflessivi (P(True), BB P(True), Label Prob.), più BB Semantic Entropy (la versione
black-box, che stima la probabilità di ogni significato dalla frequenza dei
campioni; fino al 29/09 era per errore una copia di Semantic Entropy), più due
verbalized nella sezione dedicata. Nella replica del paper girano tutti i
metodi delle Tabelle 7 e 10, compresi i 6 verbalized con i prompt originali. I metodi density-based (Mahalanobis, RMD, RDE) sono esclusi per scelta:
richiedono le statistiche del training set al momento dell'inferenza, cosa
incompatibile con l'esecuzione sul dispositivo. HUQ-MD non esiste in lm-polygraph.
Il dettaglio è in `excluded_methods.md`, generato a ogni run.

---

## 10. Note metodologiche e limiti noti

- **Precisione non uniforme.** Nei confronti di scala cambiano insieme dimensione
  e precisione (4 bit per gli LFM2, bf16 per i Gemma e per Mistral). L'effetto della
  quantizzazione si legge in modo pulito solo nel confronto dedicato
  (`--run_quant_comparison`).
- **Poche risposte giuste o sbagliate.** Il PRR del paper non dipende
  meccanicamente dall'accuracy, ma con pochi casi nella classe minoritaria è
  rumoroso: lo mostrano gli intervalli di confidenza. Le celle cliniche a testo
  libero hanno AlignScore basso: lì gli intervalli vanno letti prima dei valori.
- **Severità e difficoltà.** Gli strati di severità contengono domande diverse;
  `fig_severity_matched.png` separa i due effetti in modo approssimato (la
  difficoltà è stimata dagli altri modelli).
- **Metodi indistinguibili.** Con poche centinaia di domande molti metodi non sono
  separabili: una classifica va letta con `paired_comparisons.py`.
- **Metodi verbalized.** lm-polygraph tratta una confidenza non estraibile come
  massima confidenza, quindi il PRR di un modello che non rispetta il formato va
  letto insieme al parse-failure rate. GSM8k è escluso da questa sezione perché
  40 token non bastano per il ragionamento e la confidenza insieme.
- **Costi** misurati su GPU, non su un dispositivo mobile: indicano i rapporti fra
  i metodi, non i tempi reali sul telefono. La memoria di picco è misurata per
  cella modello × dataset, non per singolo metodo.
- **Correzioni del 29/09/2026** (revisione del codice). Cambiano i risultati:
  PRR del paper (normalizzato) al posto dell'area grezza, con pareggi in valore
  atteso; prompt in inglese dai template ufficiali; chat template anche per gli
  LFM2; un solo BOS nei prompt con chat template (prima due); lettera MCQ non più
  letta dall'articolo "a"; TriviaQA con tutti gli alias; MMLU stratificato;
  Mistral in bf16. Cambiano le analisi: confronti appaiati con il migliore scelto
  in ogni ricampionamento, Bonferroni e stato "non testabile"; errori fra le
  risposte più confidenti al posto del silent failure rate; Kendall tau per
  dataset con intervallo bootstrap; severità a parità di difficoltà; fasi di costo
  dichiarate per ogni calcolatore e metrica di qualità cronometrata a parte.
  Cambiano velocità e memoria: K campioni in una sola chiamata senza stati
  nascosti, batch NLI 50/20 invece di 10/2, SDPA sui Gemma. Rendono il run più
  sicuro: impronta del codice che blocca le riprese miste, scheda delle
  condizioni di esecuzione, log con modello, dataset e blocco.
- **Run precedenti.** Le cartelle prodotte prima dei commit `8ccbd95` ed
  `e2dedbf` hanno anche altri difetti noti (TriviaQA valutato sullo split di test
  senza risposte, generazione senza stringhe di arresto, 100 istanze per cella).
  Nessuna run precedente al 29/09 va mescolata con i risultati nuovi.
