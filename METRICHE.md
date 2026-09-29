# Metriche del benchmark: cosa misurano, quali valori possono assumere, come si leggono

Il benchmark usa due livelli di misura che è importante non confondere.

Il **primo livello** giudica la *risposta* del modello: è corretta o no? Sono le
*generation metrics* (`MCQAccuracyMetric`, `AlignScore`, `GSM8kAccuracyMetric`), che
producono un punteggio di qualità per ogni singola domanda.

Il **secondo livello** giudica il *metodo di stima dell'incertezza*: quando dice
"non sono sicuro", ha ragione? È il **PRR** (Prediction Rejection Ratio), che
prende in input i punteggi di qualità del primo livello e i punteggi di
incertezza del metodo, e restituisce un unico numero per metodo.

> **Correzione del 29/09.** Fino a quella data il benchmark riportava l'area
> grezza sotto la curva prediction-rejection (colonna `prr_0.5` di lm-polygraph)
> invece del PRR definito nel paper. L'area grezza parte dal livello
> dell'accuracy del modello e ci aggiunge poco: nelle run di settembre valeva
> quasi esattamente accuracy + 0.045 in tutte le celle (correlazione 0.98), quindi
> le figure confrontavano l'accuracy dei modelli e non i metodi UQ. La nota che
> motivava la scelta ("il paper riporta il valore grezzo") era sbagliata: il paper
> riporta il rapporto normalizzato. L'equivoco nasce probabilmente dalla frase
> "PRR before and after normalization" del paper, dove però "normalization" si
> riferisce alla calibrazione dei punteggi di incertezza, non al PRR.
> Le run precedenti si possono rileggere con la metrica giusta, senza GPU, con
> `recompute_stats.py`.

---

## Livello 1 — Metriche di qualità della risposta

### `MCQAccuracyMetric` (MMLU, MedQAbstain-LT, MedQAbstain-Safe)

Estrae dalla risposta la lettera scelta (A–D) e la confronta con quella
corretta. L'estrazione prova, in ordine: risposta che è la sola lettera;
lettera dopo un marcatore ("Answer: C", "The answer is C"); lettera maiuscola
isolata che non sia l'articolo inglese "A" seguito da una parola ("A patient
with..."); qualunque lettera maiuscola isolata. Se nessuna regola trova una
lettera, l'istanza conta come sbagliata e viene conteggiata nel
**parse-failure rate della risposta**, riportato accanto all'accuracy.

Bug corretto il 29/09: prima il testo veniva messo in maiuscolo prima di
cercare la lettera, quindi "I think it is a C" veniva letto come "A". Il
self-test eseguito all'avvio di `main.py` contiene ora questi casi.

- **Valori per istanza**: 0.0 oppure 1.0 (binaria).
- **Media su un dataset**: l'accuracy, in [0, 1].
- **Baseline del caso**: 1/numero di opzioni, cioè 0.25 con 4 opzioni.

### `GSM8kAccuracyMetric` (GSM8k)

Legge il numero dopo "The answer is" (il formato chiesto dal template
ufficiale di lm-polygraph) e lo confronta con il riferimento con tolleranza
`1e-4`; se il formato manca, usa l'ultimo numero del testo.

- **Valori per istanza**: 0.0 oppure 1.0 (binaria).
- **Baseline del caso**: praticamente 0 (risposta aperta numerica).

### `AlignScore` (CoQA, TriviaQA, MedicationQA, MedQuAD)

Metrica di allineamento semantico (RoBERTa-large addestrato apposta, incluso in
lm-polygraph): misura quanto il contenuto della risposta di riferimento è
supportato dalla risposta generata. Serve per le risposte in linguaggio libero.
È addestrata sull'inglese: anche per questo, dal 29/09, i prompt sono in
inglese (prima erano in italiano e un modello piccolo poteva rispondere in
italiano, venendo giudicato sbagliato anche con la risposta giusta).

Su **TriviaQA** i riferimenti sono tutti gli alias accettati dal dataset e la
qualità è il **massimo** di AlignScore sugli alias (`MaxOverReferences`), come
nel protocollo ufficiale (`multiref: true`).

- **Valori per istanza**: continui, nominalmente in [0, 1].
- **Media su un dataset**: qualità media, **non** un'accuracy.

### Conseguenza sul confronto tra dataset

Una metrica **binaria** produce una separazione netta tra risposte giuste e
sbagliate, una **continua** un gradiente. Confrontare PRR **dentro** lo stesso
formato di risposta è lecito; **tra** formati diversi va fatto solo in modo
qualitativo. Per questo la griglia di severità è 2x2: le due celle MCQ
condividono la metrica, le due a risposta libera anche, e l'effetto della
severità si legge lungo le colonne. Le due celle a risposta libera usano anche
lo stesso `max_new_tokens` (100) e la stessa finestra di lunghezza sui
riferimenti (80–600 caratteri).

---

## Livello 2 — PRR

### Cosa calcola

Si ordinano le istanze dalla più confidente alla più incerta secondo il metodo;
si scartano progressivamente le più incerte (da 0 fino a `max_rejection` = 50%)
e si misura la qualità media di ciò che resta; la media di questi valori è
l'**area** della curva prediction-rejection (AUC). La qualità è riscalata
min-max in [0, 1] nella cella.

Il PRR confronta questa area con quella di due ordinamenti di riferimento:

```
PRR = (AUC_metodo − AUC_casuale) / (AUC_oracolo − AUC_casuale)
```

- **AUC casuale**: ordinamento a caso. Il suo valore atteso è esattamente la
  qualità media della cella (ogni sottoinsieme scelto a caso ha in media la
  qualità media), e il codice usa quello. lm-polygraph la stima mediando 1000
  permutazioni: stesso numero a meno del rumore di Monte Carlo.
- **AUC oracolo**: ordinamento perfetto (per qualità).

In breve: *"quanto del guadagno massimo possibile, rispetto a scartare a caso,
ottiene questo metodo?"*

### Valori possibili

- **1** = il metodo ordina come l'oracolo; **0** = come il caso; **negativo** =
  peggio del caso (tiene le risposte sbagliate e scarta quelle giuste).
- Il **`0.5` nel nome `prr_0.5`** è `max_rejection`, non un valore massimo.
- Nei CSV: `prr` (e `prr_ci_low`, `prr_ci_high`) è il PRR del paper; `prr_raw`
  è l'area grezza, conservata solo per confronto con le run precedenti. Nei file
  `results_*.csv` la riga con `ue_metric = prr_0.5_normalized` è il PRR, quella
  con `prr_0.5` l'area; le figure usano la prima.

### Pareggi

Molti metodi danno lo stesso punteggio a più istanze (NumSet conta le risposte
distinte fra i 10 campioni: un intero; ogni metodo a campionamento dà lo stesso
valore quando i campioni coincidono). lm-polygraph, a parità di punteggio,
ordina le istanze nell'ordine del file, quindi il PRR di quei metodi dipendeva
dall'ordine delle righe. Dal 29/09 il PRR è il suo **valore atteso su tutti gli
ordinamenti dei pareggi**: siccome il PRR è lineare nelle qualità ordinate, si
ottiene esattamente sostituendo ogni gruppo di pareggi con la sua qualità media
(nessuna permutazione casuale). Un metodo costante ottiene quindi esattamente 0.

Accanto al PRR si riporta `distinct_fraction`, la frazione di punteggi distinti:
vicina a 0 significa che il metodo non ordina quasi nulla.

### Rumore quando le risposte giuste o sbagliate sono poche

Il PRR normalizzato non dipende meccanicamente dall'accuracy, ma con pochissime
risposte giuste (o sbagliate) il denominatore è piccolo e la stima è rumorosa.
Non servono più soglie fisse di "accuracy interpretabile" (le vecchie bande
0.30–0.85 non avevano una giustificazione ed erano una conseguenza dell'area
grezza): lo dice l'ampiezza dell'intervallo di confidenza, e la tabella
`fig_tabella_prr_accuracy.png` riporta il numero di casi della classe
minoritaria.

### Casi degeneri

- **Qualità costante** (il modello sbaglia tutto o indovina tutto): il PRR non è
  definito e il codice restituisce `NaN`.
- **Punteggio di incertezza `NaN`**: lm-polygraph lo sostituisce con `-1e7`, cioè
  il valore **più basso** di incertezza: quelle istanze vengono trattate come le
  più confidenti in assoluto. Convenzione mantenuta per confrontabilità; ha
  conseguenze sui metodi verbalized (vedi sotto).

Il calcolo è in `analysis_lib.py` (`prr_normalized`, `prr_raw`), usato da tutti
gli script. `check_prr_implementation()` verifica ogni volta che punteggi casuali
diano ~0, l'oracolo 1, un punteggio invertito un valore negativo, che senza
pareggi l'area coincida con la formula di lm-polygraph e che con i pareggi il
valore sia la media sugli spareggi.

---

## Metriche aggiuntive prodotte dal benchmark

### Intervalli di confidenza bootstrap

Si ricampionano con reimmissione le domande (1000 volte, seed 3407) e si
ricalcola il PRR — area, oracolo e caso — su ogni ricampionamento; l'intervallo
è fra il 2.5° e il 97.5° percentile. Il ricampionamento è sugli indici delle
domande e si applica insieme a punteggi e qualità.

Sulle figure aggregate su più dataset l'intervallo viene **propagato**: ogni
intervallo diventa un errore standard (semi-ampiezza / 1.96), si combinano come
errori indipendenti e si torna a un intervallo al 95%. L'indipendenza fra
dataset (campioni disgiunti da fonti diverse) va dichiarata.

### Metodi indistinguibili dal migliore (`paired_comparisons.py`)

Per ogni cella si ricampionano le domande e su ogni ricampionamento si
ricalcolano i PRR di **tutti** i metodi, guardando chi è il migliore **in quel
ricampionamento**. Un metodo è dichiarato *peggiore* solo se risulta il migliore
in meno di α/(m−1) dei ricampionamenti (α = 0.05, m = metodi confrontati:
correzione di Bonferroni); altrimenti è *indistinguibile*. Non è un test
d'ipotesi formale ma una regola di selezione conservativa nello spirito del
Model Confidence Set (Hansen, Lunde e Nason, 2011), e nella tesi va descritta
così. Se meno di metà dei ricampionamenti è calcolabile, il verdetto è *non
testabile* (prima veniva contato come distinguibile): succede quando la qualità
è costante (accuracy 0 o 1) e vale per tutti i metodi della cella insieme. Un
punteggio costante invece ha PRR 0 (come scartare a caso) e viene confrontato
normalmente. Ogni cella ha il proprio generatore casuale, con seme ricavato da
modello e dataset, così il risultato non dipende dalle altre celle nel file.

Correzioni del 29/09: prima il migliore veniva scelto una volta sui dati
completi e poi testato sugli stessi dati (chi vinceva per fortuna sembrava
sistematicamente migliore), i 25 confronti non avevano correzione, e lo stato
"non testabile" mancava. Nel CSV restano, per confronto, la differenza dal
migliore sui dati completi con intervallo appaiato al 95% e con correzione di
Bonferroni.

### Errori fra le risposte date con più fiducia

`error_rate_top10`: fra il 10% di risposte su cui il metodo è più sicuro, la
frazione di risposte **sbagliate** (qualità < 0.5). Va letta accanto a
`error_rate_overall`, la frazione di risposte sbagliate in tutto: un metodo che
non sa nulla dà in media l'error rate complessivo, uno utile di meno. Si legge
come *"fra le risposte date con più sicurezza, quante sono sbagliate"*. I pareggi
sul bordo del 10% sono risolti in valore atteso. Un punteggio `NaN` (confidenza
verbalized non estraibile) conta come massima confidenza, come nel PRR: così le
due metriche e il PRR vedono le stesse istanze. Scartarli, come nella prima
versione, faceva sembrare utile un metodo i cui parse failure cadevano tutti
sulle risposte sbagliate. Nella griglia clinica il riepilogo è in
`error_rate_most_confident.csv`, con l'error rate complessivo nell'ultima riga.

Sostituisce il **silent failure rate** ("quale frazione degli errori finisce nel
10% più confidente", ancora nei CSV come `silent_failure_rate`), che ha un
tetto: con il 90% di risposte sbagliate al massimo un decimo degli errori può
stare nel 10% più confidente, quindi il suo massimo è ~0.11 e il valore del caso
0.10, e non distingueva nulla proprio nelle celle cliniche a risposta libera.

- **Soglia di correttezza**: 0.5 sul punteggio di qualità. Per le metriche
  binarie è indifferente; per AlignScore è una scelta convenzionale, da
  dichiarare.

### Severità a parità di difficoltà

Le due celle di severità contengono domande diverse: se quelle pericolose sono
anche più difficili, un calo del PRR non è attribuibile alla severità.
`fig_severity_matched.png` (e `severity_matched_difficulty.csv`) confronta gli
strati a parità di difficoltà: la difficoltà di una domanda è il numero di
**altri** modelli che la azzeccano (escluso quello valutato, per non rendere
l'analisi circolare); per ogni livello si tengono tante domande quante ne ha lo
strato più povero (sottocampionamento ripetuto 50 volte) e si ricalcolano i PRR.
Serve che almeno tre modelli abbiano girato su entrambi gli strati.

### Parse-failure rate (metodi verbalized)

Frazione di istanze in cui la confidenza dichiarata **non è estraibile** dal testo
generato. **Va letto sempre insieme al PRR**: un punteggio `NaN` diventa `-1e7`,
cioè massima confidenza, quindi un modello che non produce il formato richiesto
non viene penalizzato dal PRR.

### Kendall tau (trasferimento della classifica)

Quanto la classifica dei metodi (per PRR) di un modello coincide con quella del
modello-ancora a 7B (Mistral-7B-Instruct-v0.2). Range [−1, 1].

Dal 29/09 il tau si calcola **per dataset**, non sulla media fra dataset (che
mescolava scale diverse), e al posto del p-value di scipy — che assume i 26
metodi indipendenti, mentre molti condividono gli stessi dati — si riporta un
**intervallo bootstrap appaiato sulle domande**: le due classifiche sono
calcolate sulle stesse domande, si ricampionano le domande e si ricalcolano
entrambe. Le figure non collegano con linee modelli di famiglie diverse.

### Costo: marginale vs pieno standalone

- **Costo marginale**: solo la chiamata dello stimatore su statistiche già
  calcolate.
- **Costo pieno standalone**: la chiamata dello stimatore più il tempo misurato di
  **tutti** i calcolatori di statistiche da cui dipende, anche indirettamente
  (chiusura transitiva delle dipendenze dichiarate). È il numero rilevante per il
  deployment on-device.

Ogni calcolatore ha una fase dichiarata esplicitamente (`PHASE_OF_CALCULATOR` in
`main.py`: generazione greedy, K campioni, NLI sui campioni, NLI sulle alternative
di CCP, cross-encoder di SAR/SentenceSAR/TokenSAR, forward extra di PMI/CPMI e
P(True)); prima la fase veniva indovinata dal nome della classe, e ad esempio
l'NLI di CCP finiva in "generazione greedy". BB P(True) genera da sé le proprie
risposte dentro la chiamata dello stimatore: il suo tempo è classificato come
generazione. Il calcolo della metrica di qualità (AlignScore) è cronometrato come
fase a parte (`metrica_qualita`) e non entra nel costo dei metodi. I tempi per
fase di ogni cella sono in `phase_timings.csv` e in `fig_phase_timings.png`.

Nota su TokenSAR: in lm-polygraph 0.7.0 la sua statistica `token_similarity` è
prodotta dallo stesso calcolatore che confronta i K campioni, quindi il suo costo
pieno include il campionamento. È come la libreria lo calcola; un'implementazione
minima potrebbe farne a meno.

Tutte le misure di tempo usano `time.perf_counter()` con `torch.cuda.synchronize()`
prima di ogni lettura del cronometro.
