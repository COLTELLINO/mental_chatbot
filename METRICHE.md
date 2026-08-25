# Metriche del benchmark: cosa misurano, quali valori possono assumere, come si leggono

Il benchmark usa due livelli di misura che è importante non confondere.

Il **primo livello** giudica la *risposta* del modello: è corretta o no? Sono le
*generation metrics* (`AccuracyMetric`, `AlignScore`, `GSM8kAccuracyMetric`), che
producono un punteggio di qualità per ogni singola domanda.

Il **secondo livello** giudica il *metodo di stima dell'incertezza*: quando dice
"non sono sicuro", ha ragione? È la *UE metric* (`PredictionRejectionArea`), che
prende in input i punteggi di qualità del primo livello e i punteggi di
incertezza del metodo, e restituisce un unico numero per metodo.

Nessuna delle due misura la stessa cosa dell'altra, e il PRR non è interpretabile
senza sapere in che regime di accuracy è stato calcolato — è il motivo per cui
ogni figura è accompagnata da `accuracy_table.csv`.

---

## Livello 1 — Metriche di qualità della risposta

### `AccuracyMetric` (MMLU, MedQAbstain-LT, MedQAbstain-Safe)

Confronto esatto di stringhe tra la risposta generata e quella di riferimento,
dopo aver applicato una regex che scarta tutto ciò che segue la lettera iniziale
(`(?<=[ABCDabcd])[\s\S]*`), perché i modelli tendono ad aggiungere punteggiatura
o spiegazioni dopo la lettera.

- **Valori per istanza**: 0.0 oppure 1.0 (binaria).
- **Media su un dataset**: l'accuracy, in [0, 1].
- **Baseline del caso**: 1/numero di opzioni, cioè 0.25 con 4 opzioni.

### `GSM8kAccuracyMetric` (GSM8k)

Implementata nel progetto perché lm-polygraph non ha una metrica per risposte
numeriche. Estrae l'**ultimo** numero presente nel testo generato — scelta
robusta al fatto che il modello non segua il formato `Risposta finale: X`
richiesto nel prompt — e lo confronta col riferimento con tolleranza `1e-4`.

- **Valori per istanza**: 0.0 oppure 1.0 (binaria).
- **Media su un dataset**: l'accuracy, in [0, 1].
- **Baseline del caso**: praticamente 0 (risposta aperta numerica).

### `AlignScore` (CoQA, TriviaQA, MedicationQA, MedQuAD)

Metrica di allineamento semantico (modello RoBERTa-large addestrato apposta,
incluso in lm-polygraph): misura quanto il contenuto della risposta generata è
supportato dal testo di riferimento. Serve per le risposte in linguaggio libero,
dove il confronto esatto classificherebbe come sbagliata quasi ogni risposta
corretta ma formulata diversamente.

- **Valori per istanza**: continui, nominalmente in [0, 1].
- **Media su un dataset**: qualità media, **non** un'accuracy.
- **Baseline del caso**: non definita in modo netto; una risposta generica e
  vagamente pertinente ottiene già un punteggio intermedio.

### Conseguenza sul confronto tra dataset

Una metrica **binaria** produce una separazione netta tra risposte giuste e
sbagliate. Una metrica **continua** produce un gradiente. Il PRR calcolato sopra
le due non ha lo stesso tetto raggiungibile, quindi:

- confrontare PRR **dentro** lo stesso formato di risposta (stessa metrica) è
  lecito;
- confrontare PRR **tra** formati diversi (Accuracy vs AlignScore) va fatto solo
  in modo qualitativo, dichiarando il limite.

È esattamente il motivo per cui la griglia di severità è costruita 2x2: le due
celle MCQ condividono `AccuracyMetric`, le due celle a risposta libera
condividono `AlignScore`, e l'effetto della severità si legge lungo le colonne, a
parità di metrica. Per la stessa ragione le due celle a risposta libera usano
anche lo stesso `max_new_tokens` (100) e la stessa finestra di lunghezza sui
riferimenti (80–600 caratteri): senza questi vincoli, una differenza di PRR
potrebbe derivare dalla lunghezza dei testi invece che dal rischio clinico.

---

## Livello 2 — `PredictionRejectionArea` (PRR)

### Cosa calcola

Ordina le istanze dalla più confidente alla più incerta secondo il metodo in
esame; poi, scartando progressivamente le più incerte (dallo 0% fino a
`max_rejection`), misura la qualità media di ciò che resta; infine media questi
valori. Prima di tutto ciò, i punteggi di qualità vengono **normalizzati min-max
in [0, 1]** all'interno del batch di valutazione.

In breve: *"se scarto le risposte di cui il modello è meno sicuro, quanto migliora
la qualità di quelle che tengo?"*

### Valori possibili

- **Range**: [0, 1] per costruzione, grazie alla normalizzazione min-max della
  qualità.
- Il **`0.5` nel nome `prr_0.5`** è il parametro `max_rejection`, cioè il tetto
  della frazione di risposte scartabile (50%). **Non** è un valore massimo della
  metrica né una soglia: un PRR di 0.62 non significa "oltre il massimo".
- Un valore **alto** significa che l'incertezza ordina bene; un valore vicino a
  quello che si otterrebbe ordinando a caso significa che non porta segnale.

Nel benchmark viene usato `prr_0.5` **grezzo**. lm-polygraph calcola anche una
variante `prr_0.5_normalized`, riscalata tra il punteggio di un ordinamento
casuale e quello di un oracolo; quella variante viene scartata perché il paper di
riferimento (Vashurin et al.) riporta il valore grezzo.

### Il regime di accuracy: perché il PRR da solo non basta

Il PRR ha bisogno che nel campione ci siano **sia** risposte corrette **sia**
risposte sbagliate: è un ordinamento tra le due categorie.

| Accuracy del modello | Cosa succede al PRR |
|---|---|
| ~ baseline del caso (≈25% su MCQ a 4 opzioni) | Il modello tira a indovinare: giusto/sbagliato dipende dalla fortuna, nessuno stato interno può prevederlo. Il PRR crolla per **tutti** i metodi insieme, indipendentemente dalla loro qualità. |
| ~ 40–70% | Molti errori e molte risposte corrette da ordinare: è qui che il PRR misura davvero il metodo. |
| ~ 100% | Quasi nessun errore da trovare. Con 3 errori su 100 domande il risultato dipende da dove capitano quei 3 punti: stima dominata dal rumore. |

Confrontare il PRR tra modelli con accuracy molto diverse significa confrontare
misure prese in regimi diversi. Per questo `accuracy_table.csv` va citato nella
didascalia di ogni figura: *"PRR 0.31 con accuracy 54%"* è un'informazione,
*"PRR 0.31"* da solo non lo è.

### Casi degeneri gestiti nel codice

- **Qualità costante** (il modello sbaglia tutto o indovina tutto): il
  denominatore della normalizzazione min-max è zero. Il codice restituisce `NaN`
  invece di un numero privo di significato.
- **Punteggio di incertezza `NaN`**: lm-polygraph lo sostituisce con `-1e7`, cioè
  il valore **più basso** possibile di incertezza. Quelle istanze vengono quindi
  trattate come le più confidenti in assoluto e non vengono mai scartate. È un
  dettaglio con conseguenze concrete sui metodi verbalized (vedi sotto).

---

## Metriche aggiuntive prodotte dal benchmark

### Intervalli di confidenza bootstrap

Il benchmark gira su un campione di domande, non sul dataset intero: ripetendolo
con altre domande ogni PRR verrebbe un po' diverso. Il bootstrap stima quanto,
senza nuove esecuzioni su GPU — ricampiona con reimmissione le istanze già
calcolate (1000 volte, seed 3407), ricalcola il PRR ogni volta, e prende il
2.5° e 97.5° percentile dei valori ottenuti.

Il ricampionamento avviene sugli **indici delle istanze** e viene applicato
insieme a punteggio e qualità: l'unità che varia tra un esperimento e l'altro è
la domanda, e separare i due array distruggerebbe l'accoppiamento sottostimando
l'incertezza.

Per affermare *"il metodo A batte il metodo B"* non basta guardare se le due
barre d'errore si sovrappongono — è un criterio troppo conservativo, perché A e B
sono valutati sulle stesse domande e i loro punteggi sono correlati. Si usa
invece il **bootstrap appaiato della differenza**: a ogni ricampionamento si
calcola PRR(A) − PRR(B) sullo stesso campione, e si guarda se l'intervallo delle
differenze contiene lo zero.

Sulle figure aggregate su più dataset l'intervallo non viene ricalcolato ma
**propagato**: ogni intervallo diventa un errore standard (semi-ampiezza / 1.96),
si combinano come errori indipendenti, e si torna a un intervallo al 95%.
L'ipotesi di indipendenza tra dataset è ragionevole (campioni disgiunti da fonti
diverse) ma va dichiarata.

### Silent failure rate

Frazione delle risposte **sbagliate** che finisce nel decile **più confidente**
del metodo.

- **Range**: [0, 1]. Più basso è meglio.
- **Soglia di correttezza**: 0.5 sul punteggio di qualità. Per le metriche binarie
  è indifferente; per AlignScore è una scelta convenzionale, da dichiarare.
- **Perché serve**: il PRR misura la capacità di ordinamento *in media*. Il silent
  failure rate misura una cosa diversa e clinicamente più rilevante: quanti errori
  passano del tutto inosservati, presentati con la massima sicurezza. Un metodo
  può avere un PRR discreto e comunque lasciar passare errori pericolosi nel
  decile in cui l'utente si fida di più.

### Parse-failure rate (solo metodi verbalized)

Frazione di istanze in cui la confidenza dichiarata **non è estraibile** dal testo
generato: la regex di `Verbalized1S` non trova un numero, o nessuna delle
espressioni di `Linguistic1S` compare nella risposta.

- **Range**: [0, 1].
- **Va letto sempre insieme al PRR**, per il motivo tecnico spiegato sopra: un
  punteggio `NaN` viene convertito da lm-polygraph in `-1e7`, cioè in massima
  confidenza. Un modello che non riesce nemmeno a produrre il formato richiesto
  **non viene penalizzato** dal PRR — anzi, quelle istanze non vengono mai
  scartate. Senza il parse-failure rate accanto, il suo risultato sembrerebbe
  migliore di quanto sia.
- È esso stesso un risultato: per i modelli più piccoli il dato interessante
  potrebbe essere proprio *"non emette nemmeno il formato richiesto"*.

### Kendall tau (trasferimento del ranking)

Misura quanto la classifica dei metodi UQ di un modello coincide con quella del
modello-ancora a 7B (Mistral-7B-Instruct-v0.2, nel regime di scala del paper di
riferimento).

- **Range**: [−1, 1]. 1 = ranking identico, 0 = scorrelato, −1 = invertito.
- **A cosa serve**: se il tau cala scendendo di scala, le conclusioni di un
  benchmark costruito su modelli da 7–12B non si trasferiscono ai modelli
  piccoli — che è la giustificazione stessa di un benchmark dedicato.

### Costo: marginale vs pieno standalone

Due numeri diversi, entrambi riportati.

- **Costo marginale**: solo l'aritmetica dello stimatore su statistiche già
  calcolate. Risponde a *"quanto costa aggiungere questa tecnica se sto già
  calcolando tutto il resto?"*.
- **Costo pieno standalone**: generazione greedy + K generazioni campionate (se la
  tecnica le richiede) + forward del modello NLI (se li richiede) + aritmetica.
  Risponde a *"quanto costa questa tecnica se è l'unica che faccio girare sul
  telefono?"* — ed è il numero rilevante per il deployment on-device.

La distinzione non è accademica: guardando solo il costo marginale, i metodi
basati su diversità campionaria sembrano quasi gratuiti (millisecondi di algebra
su una matrice già pronta) mentre il loro costo vero — le K generazioni più i
forward NLI — risulta ammortizzato nel magazzino condiviso e non attribuito a
nessuno. La **frontiera di Pareto** (`fig_pareto_cost_quality.png`) usa il costo
pieno: una tecnica è *dominata* se ne esiste un'altra insieme più economica e più
affidabile, e le non dominate formano la curva dei migliori compromessi possibili
a ogni budget di calcolo.

Tutte le misure di tempo usano `time.perf_counter()` con `torch.cuda.synchronize()`
prima di ogni lettura del cronometro: le operazioni GPU sono asincrone, e senza
sincronizzazione si misurerebbe l'**accodamento** dell'operazione invece della sua
esecuzione, sottostimando i tempi anche di ordini di grandezza.
