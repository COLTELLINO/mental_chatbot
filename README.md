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

Ogni metodo UQ viene valutato con il **Prediction-Rejection Ratio (PRR)**: quanto
migliora la qualità media delle risposte scartando quelle con l'incertezza più
alta. La definizione delle metriche, i valori possibili e come leggerle sono in
[`METRICHE.md`](METRICHE.md).

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
| `analysis_lib.py` | Funzioni condivise fra `main.py` e gli script offline: aggregazione fra dataset, famiglie di metodi, silent failure rate. Dipende solo da numpy e pandas. |
| `make_figures.py` | Rigenera **tutte le figure e le tabelle-immagine** dai CSV, in pochi secondi e senza GPU. |
| `paired_comparisons.py` | Test bootstrap appaiati: quali metodi sono statisticamente indistinguibili dal migliore. Senza GPU. |
| `recompute_silent_failure.py` | Ricalcola il silent failure rate di una cartella di risultati prodotta da versioni vecchie del codice. Senza GPU. |
| `drop_models_from_checkpoints.py` | Toglie modelli o dataset dai checkpoint, per rieseguire solo quelli. |
| `preflight.sh` | Controlli di sola lettura prima di un run lungo (codice, token, immagine Docker, GPU, disco). |
| `sbatch_script.sh`, `run_docker.sh`, `train.sh` | Catena di lancio su cluster SLURM: `sbatch` → container Docker → `main.py`. |
| `create_docker_image.sh`, `build/` | Costruzione dell'immagine Docker (`Dockerfile` e file dei requirements). |
| `sync_results.sh` | Allinea la cartella `results/` fra i due nodi del cluster (faretra e moro232). |
| `results_store.py` | Modulo per checkpoint con semantica di sovrascrittura per riga. **Non è usato** dalla pipeline attuale: è una base per un'eventuale riorganizzazione dei checkpoint. |
| `METRICHE.md` | Spiegazione di tutte le metriche: accuracy, AlignScore, PRR, bootstrap, silent failure rate, parse-failure rate, Kendall tau, costo marginale e pieno. |

---

## 2. Requisiti

- **GPU NVIDIA con 24 GB** (sviluppato su RTX 3090). I modelli da 4B girano in bf16
  e occupano circa 9 GB; sulla stessa GPU girano anche il modello NLI
  (DeBERTa-large) e AlignScore.
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

La ripresa riusa i risultati salvati **anche se nel frattempo il codice è
cambiato**. Dopo una modifica al codice che cambia i risultati, usare
`--no_resume` oppure una `--results_dir` nuova.

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
| `--quant_compare_model M` | `LFM2-1.2B` | Modello del confronto sulla quantizzazione. I due Gemma non sono utilizzabili: a 4 bit producono logit NaN. |

### Esecuzione

| Argomento | Default | Cosa fa |
|---|---|---|
| `--batch_size N` | `1` | Batch di generazione. Con 1 la memoria usata è minima e non c'è padding fra sequenze di lunghezza diversa. Per i modelli pesanti il batch viene comunque ridotto in automatico. |
| `--chunk_size N` | `100` | Istanze per blocco con checkpoint dentro ogni cella. `0` disattiva i blocchi. I risultati sono identici con o senza blocchi. |
| `--no_resume` | no | Ignora i checkpoint e riesegue tutto. Da usare dopo ogni modifica al codice che cambia i risultati. |
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
seed `3407` (`SEED` in `main.py`), numero di istanze, stringhe di arresto e numero
massimo di token per dataset (`DATASETS` e `SEVERITY_DATASETS` in
`dataset_prep.py`), precisione dei modelli (`NO_QUANT_MODELS` in `main.py`).

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
`stringhe di arresto`; in `results_smoke/sample_generations.csv` le risposte devono
fermarsi alla fine della riga.

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

### Tempi indicativi (RTX 3090, 5 modelli)

| Parte | Istanze | Tempo stimato |
|---|---|---|
| CoQA, TriviaQA, MMLU | 500, 1000, 1000 | circa 25 ore |
| GSM8k | 500 | circa 70 ore |
| Griglia di severità | 556 + 556 + tutte + 1000 | circa 1-2 giorni |
| Quantizzazione (LFM2-1.2B, 2 varianti) | come la pipeline principale | circa 10 ore |
| Verbalized | come la pipeline principale, senza GSM8k | alcune ore |

Stime ricavate dai tempi misurati nei run precedenti; dipendono dal carico del
nodo.

---

## 7. Output prodotti

Tutto viene scritto in `--results_dir`. I CSV sono la fonte dei dati; le figure si
rigenerano da lì con `make_figures.py`.

### Pipeline principale

| File | Contenuto |
|---|---|
| `results_final.csv` | PRR di ogni metodo per ogni modello × dataset, nel formato di lm-polygraph (colonne `model, key, estimator, ue_metric, value, dataset`). `ue_metric = prr_0.5` è l'area sotto la curva di rifiuto; `prr_0.5_normalized` è la stessa area normalizzata fra metodo casuale (0) e oracolo (1). |
| `results_partial.csv` | Checkpoint di `results_final.csv`, aggiornato dopo ogni cella. |
| `results_paper_mapped.csv` | Righe `prr_0.5` con le etichette dei metodi usate nel paper, intervalli di confidenza, accuracy, silent failure rate. È la base delle Figure A e B. |
| `instance_level_stats.csv` | Per metodo × modello × dataset: PRR, intervallo bootstrap al 95%, accuracy di base, numero di istanze, frazione di punteggi NaN, silent failure rate, parse-failure rate della risposta, istanze saltate. |
| `per_instance_scores.csv` | **Il dato grezzo**: per ogni domanda, la qualità della risposta e il punteggio di incertezza di ogni metodo. Tutte le analisi offline ripartono da qui. |
| `accuracy_table.csv` | Accuracy (o AlignScore) per modello × dataset. |
| `table6_style_<modello>.csv` | Tabella nello stile della Tabella 6 del paper, per modello. |
| `rank_transfer_kendall_tau.csv` | Kendall tau fra il ranking dei metodi di ogni modello e quello di Mistral-7B, con p-value. |
| `estimator_timings.csv` | Tempi per metodo × modello × dataset: costo marginale (solo il calcolo del metodo), costo pieno standalone (tutti i calcolatori da cui dipende), quali modelli ausiliari usa, elenco dei calcolatori, memoria di picco della cella. |
| `estimator_timing_table.csv`, `estimator_cost_table.csv` | Riepiloghi dei tempi per metodo. |
| `excluded_methods.md` | Metodi del paper non inclusi e perché. |
| `sample_generations.csv` | Campione di generazioni grezze con la lettera estratta e quella attesa (task a scelta multipla): serve a controllare a occhio che le risposte siano estratte bene. |

### Sezioni extra

| File | Sezione |
|---|---|
| `results_severity_grid*.csv`, `accuracy_table_severity_grid.csv`, `silent_failure_rate.csv` | Griglia di severità (stessa struttura dei file principali: `_instance_stats`, `_per_instance`, `_mapped`). |
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
| `fig_accuracy_vs_prr.png` | PRR contro accuracy di base, con le zone in cui il PRR non è interpretabile. |
| `fig_a_white_box.png`, `fig_b_reflexive.png` | Repliche delle Figure 2 e 3 del paper, con intervalli di confidenza. |
| `fig_prr_vs_scale_by_family.png` | PRR per famiglia di metodi al variare della scala. |
| `fig_rank_transfer.png` | Kendall tau del ranking dei metodi contro il modello da 7B. |
| `fig_ties_vs_scale.png` | Quanti metodi sono indistinguibili dal migliore, per scala. Richiede `paired_comparisons.py`. |
| `fig_anchor_replication.png` | Mistral-7B contro i valori del paper. |
| `fig_severity_grid.png` | Griglia 2×2 severità × formato. |
| `fig_quant_comparison.png` | 4 bit contro bf16, per metodo, con intervalli di confidenza. |
| `fig_timing_full_cost.png`, `estimator_timing_chart.png` | Costo pieno contro costo marginale; costo marginale per modello. |
| `fig_pareto_cost_quality.png` | Frontiera di Pareto costo contro PRR, con intervalli di confidenza. |
| `fig_verbalized_accuracy_cost.png` | Quanto la richiesta di confidenza peggiora le risposte. |
| `fig_verbalized_{numeric,linguistic}.png` | PRR dei metodi verbalized per modello, con intervalli di confidenza e parse-failure rate. |
| `fig_tabella_*.png` | Tabelle in forma di immagine: accuracy, PRR affiancato all'accuracy, costi, parse-failure rate, silent failure rate. |

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

Per ogni modello × dataset trova il metodo con il PRR più alto e confronta tutti
gli altri contro di lui con un bootstrap **appaiato** (stesse domande per i due
metodi). Scrive `<input>_paired_comparisons.csv` e un riepilogo
`_paired_comparisons_summary.csv` con il numero di metodi indistinguibili dal
migliore. Per la griglia clinica:
`--per_instance_file results_severity_grid_per_instance.csv`.

### `recompute_silent_failure.py`

```bash
python3.11 recompute_silent_failure.py <results_dir>
```

Serve solo per cartelle prodotte **prima** della correzione del silent failure
rate (commit `8ccbd95`). Ricalcola il valore dai punteggi per istanza e aggiorna
i CSV in place, dopo averne salvato una copia `*.pre_sfr_fix.bak`. Sui risultati
prodotti dal codice attuale non cambia nulla.

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
| `Mistral-7B-it` | `mistralai/Mistral-7B-Instruct-v0.2` | 7B | 4 bit (NF4) |

I due Gemma girano in bf16 perché a 4 bit producono logit NaN. Gemma, MedGemma e
Mistral usano il loro chat template; gli LFM2 un prompt a completamento semplice.
Mistral-7B è l'**ancora di replica**: è uno dei modelli del paper, e serve a
verificare che la pipeline riproduca i risultati nel regime di scala originale.

### Dataset

| Dataset | Fonte HF | Tipo | Metrica | Istanze | Arresto della generazione |
|---|---|---|---|---|---|
| CoQA | `stanfordnlp/coqa` (validation) | QA conversazionale, risposta breve | AlignScore | 500 (tutte) | primo a capo |
| TriviaQA | `mandarjoshi/trivia_qa` rc.nocontext (validation) | QA, risposta breve, 5-shot | AlignScore | 1000 | primo a capo |
| MMLU | `cais/mmlu` (test) | scelta multipla, 5-shot | Accuracy | 1000 | primo a capo |
| GSM8k | `openai/gsm8k` (test) | problemi aritmetici, 5-shot | Accuracy sul numero finale | 500 | inizio di un nuovo "Problema:" |
| MedQAbstain-LT | `disi-unibo-nlp/MedQAbstain` (LT, fonte medqa_4opt) | scelta multipla, severità **alta** | Accuracy | 556 | primo a capo |
| MedQAbstain-Safe | `disi-unibo-nlp/MedQAbstain` (Safe, fonte medqa_4opt) | scelta multipla, severità **bassa** | Accuracy | 556 (tutte) | primo a capo |
| MedicationQA | `truehealth/medicationqa` | testo libero, severità **alta** | AlignScore | tutte dopo i filtri | inizio di una nuova "Domanda:" |
| MedQuAD | `lavita/MedQuAD` (solo domande informative) | testo libero, severità **bassa** | AlignScore | 1000 | inizio di una nuova "Domanda:" |

- I primi quattro sono i dataset del paper; gli altri quattro formano la griglia
  **severità × formato**, in cui le due celle di ogni colonna condividono formato e
  metrica, così che il confronto isoli la severità.
- MedicationQA e MedQuAD usano gli stessi filtri sulla lunghezza dei riferimenti
  (80-600 caratteri) e lo stesso numero massimo di token.
- Per MedQAbstain si usano 556 istanze per entrambe le celle, cioè tutte quelle
  della fonte a 4 opzioni disponibili nella cella Safe, così che le due celle
  abbiano la stessa dimensione.
- Il campionamento delle istanze è fissato dal seed 3407.

### Metodi UQ

26 metodi di lm-polygraph, mappati sulle Figure 2 e 3 del paper (`PAPER_METHODS`
in `main.py`): information-based (MSP, Perplexity, Mean Token Entropy, CCP, PMI,
Conditional PMI, Fisher-Rao, Renyi, TokenSAR), basati sulla diversità dei campioni
(Semantic Entropy, SAR, SentenceSAR, Monte Carlo Sequence Entropy e versione
normalizzata, Lexical Similarity, EigValLaplacian, DegMat, Eccentricity, NumSet)
e riflessivi (P(True), BB P(True), Label Prob.), più due verbalized nella sezione
dedicata. I metodi density-based (Mahalanobis, RMD, RDE) sono esclusi per scelta:
richiedono le statistiche del training set al momento dell'inferenza, cosa
incompatibile con l'esecuzione sul dispositivo. HUQ-MD non esiste in lm-polygraph.
Il dettaglio è in `excluded_methods.md`, generato a ogni run.

---

## 10. Note metodologiche e limiti noti

- **Precisione non uniforme.** Nei confronti di scala cambiano insieme dimensione
  e precisione (4 bit per LFM2 e Mistral, bf16 per i Gemma). L'effetto della
  quantizzazione si legge in modo pulito solo nel confronto dedicato su
  LFM2-1.2B.
- **Prompt in italiano** su dataset in inglese; il few-shot è costruito nello
  stile della pipeline, non copiato dai template di lm-evaluation-harness usati
  nel paper.
- **PRR e accuracy.** Il PRR dipende fortemente dall'accuracy di base del modello
  (vedi `fig_accuracy_vs_prr.png` e `METRICHE.md`). Le celle cliniche a testo
  libero hanno AlignScore fra 0.05 e 0.25: in quel regime il PRR non è
  interpretabile.
- **Metodi verbalized.** lm-polygraph tratta una confidenza non estraibile come
  massima confidenza, quindi il PRR di un modello che non rispetta il formato va
  letto insieme al parse-failure rate. GSM8k è escluso da questa sezione perché
  40 token non bastano per il ragionamento e la confidenza insieme.
- **Costi** misurati su GPU, non su un dispositivo mobile: indicano i rapporti fra
  i metodi, non i tempi reali sul telefono. La memoria di picco è misurata per
  cella modello × dataset, non per singolo metodo.
- **Differenze rispetto alle run precedenti al 29/09/2026.** Le cartelle di
  risultati prodotte prima dei commit `8ccbd95` ed `e2dedbf` hanno difetti noti:
  TriviaQA valutato sullo split di test, che su Hugging Face non ha le risposte;
  generazione senza stringhe di arresto; silent failure rate errato con i
  pareggi; costi di alcuni metodi sottostimati; 100 istanze per cella. I dettagli
  sono nei messaggi di quei commit. Non vanno mescolate con i risultati nuovi.
