# CLAUDE.md: passaggio di consegne per il benchmark UQ

Questo file è il contesto per chi lavora sul repo (Claude Code compreso). Va letto
tutto prima di toccare il codice. Per i dettagli d'uso c'è `README.md`; per le
metriche, `METRICHE.md`.

## Il progetto

Tesi di laurea **triennale** di Filippo Patrignani (Ingegneria e Scienze
Informatiche, Unibo Cesena; relatore Prof. Gianluca Moro), scritta in inglese su
Overleaf. È un benchmark dei metodi di uncertainty quantification (UQ) su LLM
piccoli, adatti a girare su un telefono, nel dominio clinico. Replica e
riferimento: Vashurin et al., "Benchmarking Uncertainty Quantification Methods
for Large Language Models with LM-Polygraph", TACL 2025 (arXiv 2406.15627), con
**lm-polygraph 0.7.0** (versione fissata in `build/requirements-benchmark.txt`).

- **Modelli:**
  - LFM2-350M e LFM2-1.2B (4 bit nf4);
  - MedGemma-4B-it e Gemma3-4B-it (bf16: a 4 bit producono NaN);
  - Mistral-7B-Instruct-v0.2 (bf16), come riferimento di scala.
- **Dataset, pipeline principale:** CoQA, TriviaQA, MMLU, GSM8k.
- **Griglia clinica severità × formato:**
  - MedQAbstain-LT e MedQAbstain-Safe (scelta multipla);
  - MedicationQA e MedQuAD (risposta libera).
- **Sezioni extra:**
  - metodi verbalized;
  - confronto 4 bit contro bf16;
  - replica esatta del paper (`paper_replica.py`).
- **Domande della tesi:**
  - le conclusioni del paper (modelli da 7–12B) valgono anche a 0.35–4B?
  - quanto costa ogni metodo, pensando all'esecuzione sul telefono?
  - cosa cambia con la severità clinica?

## Regole di lavoro

1. **Prima di ogni commit:** `bash tests/run_tests.sh` (CPU, circa 15 minuti, modello
   minuscolo). Con `SKIP_NETWORK_TESTS=1` si salta il test della replica, che
   scarica i dataset da Hugging Face. Ogni bug corretto ha un test che lo
   riconosce: se un test fallisce, la modifica ha reintrodotto un bug, quindi va
   capito, non aggirato.
2. **Mai credenziali nei file del repo.** Il token Hugging Face sta in `~/.uq_env`
   sui nodi e si carica con `source ~/.uq_env` prima di `sbatch_script.sh`.
3. **Identità git:** `COLTELLINO <filippo.patrignani2@studio.unibo.it>`. Si
   pusha su `origin` (github.com/COLTELLINO/mental_chatbot), branch
   `filo/benchmark-uq`, sempre e subito dopo ogni commit (`git push origin
   filo/benchmark-uq`), senza chiedere conferma. `unibo` è il repo del gruppo:
   non si pusha lì senza che Filippo lo chieda.
4. **Commenti e messaggi in italiano.** Lo stile del repo: ogni scelta non ovvia ha
   un commento che spiega il perché; le correzioni sono marcate
   "BUG CORRETTO il <data>" con il sintomo che le ha fatte trovare.
5. **Figure:** si disegnano solo in `make_figures.py`, che `main.py` lancia
   alla fine, dopo `paired_comparisons.py`. Filippo vuole una figura per ogni
   analisi, non solo CSV. Ogni figura ha una nota che spiega come leggerla.
6. **Metriche:** sono implementate una sola volta, in `analysis_lib.py`. `main.py`,
   `paired_comparisons.py`, `recompute_stats.py` e `make_figures.py` la usano;
   non duplicare formule.
7. **I risultati vecchi non si mescolano ai nuovi.** L'impronta del codice
   (`run_fingerprint.json`) rifiuta le riprese con codice cambiato. Non usare
   `--force_resume` se la modifica tocca generazioni o metriche.

## Decisioni prese (non rimetterle in discussione senza Filippo)

- **PRR del paper, normalizzato:** (AUC − caso) / (oracolo − caso), con il 50%
  di risposte scartate al massimo. Caso e oracolo si ricalcolano in ogni
  ricampionamento bootstrap. L'area grezza (`prr_raw`) resta solo come colonna di
  confronto: era quasi uguale all'accuracy (r ≈ 0.98), e per questo è stata
  abbandonata.
- **Prompt in inglese**, dai template ufficiali `simple_instruct` di lm-polygraph,
  anche per i prompt clinici e quelli verbalized. La tesi è in inglese e si
  confronta con un paper che usa prompt in inglese.
- **Replica del paper** (`paper_replica.py`):
  - parte white-box: Mistral 7B v0.2 **base** (`mistral-community/Mistral-7B-v0.2`) con prompt
    "continuation", Tabella 7;
  - parte black-box: Mistral 7B v0.2 **Instruct** con "empirical baselines" e i 6
    prompt verbalized, Tabella 10;
  - dataset e prompt sono quelli pubblicati da `LM-Polygraph/*` su Hugging Face;
  - 2.000 domande scelte come `Dataset.subsample` (seed 1); MMLU 5.700, GSM8k 1.319;
  - le impostazioni stanno in `paper_replica_configs.json`, estratte dalle
    configurazioni ufficiali; i valori del paper in `paper_reference_prr.csv`.
  - **criterio di successo** (fissato il 1° ottobre 2026, prima del lancio; testo
    completo e motivazioni in `README.md`, sezione della replica): per ognuna
    delle 7 celle, mediana di |scarto dal paper| ≤ 0.05, τ di Kendall con la
    classifica del paper ≥ 0.6 (solo white-box), e ogni metodo con |scarto| > 0.15
    spiegato. Riuscita con 7 celle su 7; parziale con almeno 3 white-box su 4 e
    2 black-box su 3, con le celle fallite spiegate. Le soglie non si cambiano
    dopo aver visto i risultati.
- **Il Kendall τ dei modelli piccoli** si calcola contro il Mistral Instruct della
  pipeline principale, che usa gli stessi prompt dei modelli piccoli, e per dataset. La
  replica serve solo a validare il codice rispetto al paper.
- **Metodi density-based esclusi** (Mahalanobis, RDE, RMD): richiedono le
  statistiche del training set al momento dell'inferenza, cosa incompatibile con il telefono. HUQ-MD non esiste
  in lm-polygraph.
- **"Errori fra il 10% più confidente"**, sempre accanto all'error rate complessivo, al
  posto del silent failure rate. Quest'ultimo ha un tetto: con il 90% di errori il suo
  massimo è circa 0.11.
- **Confronti appaiati** (`paired_comparisons.py`): il migliore si sceglie dentro ogni
  ricampionamento, con soglia di Bonferroni e stato "non testabile". È una regola di
  selezione nello stile del Model Confidence Set, non un test formale, e nella tesi
  va descritta così.
- **GSM8k è escluso dalla sezione verbalized:** con 40 token il formato non ci sta.

## Bug già corretti: non reintrodurli

Ognuno è coperto da un test (indicato fra parentesi).

1. **PRR come area grezza** invece che normalizzato (`test_new_fixes` 1, `test_integration` T9).
2. **Lettera MCQ:**
   - il testo veniva messo in maiuscolo, e l'articolo "a" diventava la risposta "A";
   - anche "Both A and C" e "A 3-month-old…" venivano letti come risposte
   (`dataset_prep.self_test_mcq_metric`, che gira all'avvio, e `test_new_fixes` 9).
3. **Doppio BOS** con il chat template: il template mette già il BOS e il tokenizer ne aggiunge un altro (`test_new_fixes` 2).
4. **LFM2 senza chat template**, mentre è un modello chat (`test_new_fixes` 5).
5. **Mistral bloccato a 4 bit**, e il confronto sulla quantizzazione controllava la lista
   sbagliata (`CHAT_TEMPLATE_MODELS` al posto di `CANNOT_QUANTIZE_MODELS`) (`test_new_fixes` 5).
6. **Attenzione eager su Gemma 3**, che non ha soft-capping: si decide da
   `config.attn_logit_softcapping` (`test_new_fixes` 6).
7. **Campionatore in batch** (`batched_sampling.py`): un log p(EOS) spurio entrava in
   `sample_log_probs`. Succedeva:
   - per i campioni fermati da una stringa di arresto;
   - per i modelli con più token di fine (Gemma: `<end_of_turn>`).
   Il padding dopo la fine non va mai contato (`test_batched_sampling`).
8. **pad_token di generazione** diverso da EOS: il `<pad>` finiva nel testo generato
   (`test_integration`).
9. **Stringhe di arresto mancanti:** il modello continuava inventando nuove domande
   (`test_integration` T1–T3).
10. **Ripresa:**
    - checkpoint di codice vecchio ripresi come validi. Ora `--no_resume` sposta
      tutto in `_superati/` e ogni cartella di blocchi porta l'impronta del codice
      (`test_new_fixes` 10–11);
    - i file per-istanza venivano sovrascritti da job con `--models` diversi
      (`test_orchestration`);
    - `--no_resume` veniva ignorato dalle sezioni extra.
11. **Confronto quantizzazione** indicizzato solo per variante: cambiando modello
    venivano riusati i risultati del modello precedente (`test_quant_section`).
12. **Costi:**
    - attribuzione per nome di classe invece che dalle dipendenze reali;
    - ora c'è una tabella esplicita calcolatore → fase, con chiusura transitiva (`test_costs`).
13. **BB Semantic Entropy** era una copia di Semantic Entropy. Nel paper è
    `SemanticEntropy(class_probability_estimation="frequency")`.
14. **Punteggi NaN (verbalized):** si trattano come massima confidenza ovunque, come fa
    lm-polygraph, anche nell'error rate (`test_new_fixes` 13).
15. **TriviaQA:**
    - riferimenti `<unk>` (si usa lo split `validation`, e ora tutti gli alias, con il massimo);
    - nella replica non si tronca a 10.000 domande prima di sceglierne 2.000, perché
      lo script ufficiale non lo fa (`test_paper_replica` 1).
16. **`ignore_exceptions=True`** in UEManager faceva sparire in silenzio i metodi che
    fallivano. Deve restare `False`.
17. **Una figura che falliva bloccava tutte le altre**, e i modelli non presenti in
    `MODEL_PARAMS_B` sparivano dalle figure.
18. **Il commit `f447892` aveva cancellato `QUANT_COMPARE_MODEL_DEFAULT`**, e in quella versione
    `main.py` non partiva. Se si rimuove codice, rilanciare i test.

## Cluster

- Due nodi SLURM, **faretra** e **moro232**, con RTX 3090 da 24 GB. **Non condividono la
  home**: checkpoint e risultati stanno sul nodo dove gira il job. Il repo sta in
  `/home/patrignani/mental_chatbot` su entrambi.
- **Lancio:**
  ```
  source ~/.uq_env
  SBATCH_NODE=faretra bash sbatch_script.sh [script.py] [argomenti]
  ```
  La catena è `sbatch` → `run_docker.sh` (immagine `mental-chatbot-image`, repo montato
  in `/workspace`, cache modelli in `/llms`) → `train.sh` → `main.py`, oppure lo script
  passato come primo argomento.
- **Dopo un pull che tocca `build/`:** `bash create_docker_image.sh`.
- **Prima di un run lungo:** `bash preflight.sh`.
- **Comandi utili:** `squeue -u patrignani`; i log sono `slurm-<jobid>.out` nella cartella del repo.
- **Su moro232** a volte la GPU è parzialmente occupata da altri: il preflight lo segnala.

## Stato al 2 ottobre 2026 e prossimi passi

- Il job `16145101` (codice vecchio) è stato cancellato il 1/10. I suoi risultati sono in
  `results_run_28_09_codice_vecchio/` su faretra, con una copia in `Workspace/BACKUPS_RISULTATI/` sul PC di Filippo.
- Test nel container e prove brevi (`results_smoke`, `results_replica_smoke_wb/_bb`) passati il 2/10.
  Picco di memoria: Mistral bf16 su CoQA 22.7 GB su 24, quindi senza margine. Collo di
  bottiglia: il cross-encoder di SAR (45–70% del tempo), circa 50–65 s per domanda su GSM8k.
- **Run complete in coda il 2/10, tutte su faretra** (moro232 ha la GPU assegnata ad altri). Gli id
  sono in `lanci_02_10.txt` sul nodo. Catena in `results/`, ogni job parte solo se il
  precedente finisce bene (`afterok`):
  1. `--datasets CoQA TriviaQA MMLU` (stima circa 22 h);
  2. `--datasets GSM8k` (stima circa 38 h);
  3. griglia di severità, LFM2-350M, LFM2-1.2B, Mistral;
  4. griglia di severità, MedGemma e Gemma3;
  5. `--run_verbalized --run_quant_comparison`.
  Repliche in parallelo: `results_paper_replica_wb` (white-box) e `results_paper_replica_bb`
  (black-box e verbalized). A fine run copiare i file di una nelle cartella dell'altra e lanciare
  `paper_replica.py --only_compare`, poi applicare il criterio di successo.
- **Non modificare** `main.py`, `dataset_prep.py`, `batched_sampling.py` o `analysis_lib.py` finché la catena
  non è finita: cambierebbe l'impronta e i job successivi rifiuterebbero la ripresa.
- `sync_results.sh` usa `srun`: per operazioni brevi l'admin lo consente.
- **Rimandati a fine esperimenti, su richiesta di Filippo:**
  - limiti da dichiarare nella tesi: regola dei ties, 5 livelli di difficoltà nella severità,
    "I'd go with A because" non leggibile;
  - `<end_of_turn>` resta nei testi di Gemma3, sia greedy sia campioni, perché lm-polygraph decodifica
    senza togliere i token speciali. La qualità non cambia, ma le similarità lessicali tra i campioni
    di Gemma3 possono uscire un po' più alte.
- **Scadenza di caricamento della tesi:** 13 novembre 2026.

## Come lavorare con Filippo

- Scrive in italiano: rispondere in italiano, in modo diretto, senza riassunti dei
  passaggi fatti.
- Vuole che i bug vengano corretti subito e che gli si dica apertamente quando si è
  sbagliato qualcosa.
- I risultati vanno spiegati in modo che possa raccontarli al relatore.
- L'abstract e il testo della tesi li scrive lui: dare spunti, non testi già pronti.
- Quando una decisione cambia il protocollo sperimentale (prompt, modelli, numerosità,
  metriche), chiedere prima di farla.
