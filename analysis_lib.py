"""Funzioni condivise fra il calcolo (main.py) e il disegno (make_figures.py).

Sta qui quello che serve a entrambi e che non deve esistere in due copie:
l'aggregazione fra dataset, che decide i valori delle Figure A e B, e la
lettura della mappa dei metodi del paper.

Dipende solo da numpy, pandas e la libreria standard: make_figures.py deve
poterlo importare senza tirarsi dietro torch e lm-polygraph, che senza driver
NVIDIA vanno in segmentation fault.
"""
import ast
import os

import numpy as np
import pandas as pd


def aggregate_across_datasets(raw_df):
    """Media del PRR di ogni metodo sui dataset, con l'intervallo di confidenza
    propagato da quelli dei singoli dataset.

    Le figure aggregate del paper mediano il PRR su tutti i task; l'intervallo
    su quella media non puo' essere preso da un singolo dataset. Qui si
    converte ogni intervallo bootstrap in un errore standard (semi-ampiezza /
    1.96), si combinano come errori indipendenti -- se(media) = sqrt(somma dei
    quadrati) / k -- e si torna a un intervallo al 95%.

    L'ipotesi di indipendenza tra dataset e' ragionevole (sono campioni
    disgiunti da fonti diverse) ma va dichiarata: se i dataset fossero
    correlati, l'intervallo risultante sarebbe leggermente ottimistico."""
    if "prr_ci_low" not in raw_df.columns:
        return raw_df.groupby(["model", "paper_label"], as_index=False)["value"].mean()

    tmp = raw_df.copy()
    se = (tmp["prr_ci_high"] - tmp["prr_ci_low"]) / (2 * 1.96)
    tmp["_se2"] = se ** 2

    agg = tmp.groupby(["model", "paper_label"], as_index=False).agg(
        value=("value", "mean"),
        n_datasets=("value", "size"),
        _se2_sum=("_se2", "sum"),
        _se_count=("_se2", "count"),
    )
    denom = agg["_se_count"].replace(0, np.nan)
    se_comb = np.sqrt(agg["_se2_sum"]) / denom
    agg["prr_ci_low"] = agg["value"] - 1.96 * se_comb
    agg["prr_ci_high"] = agg["value"] + 1.96 * se_comb
    return agg.drop(columns=["_se2_sum", "_se_count"])


# Metodi che nel nostro benchmark girano con un prompt e un max_new_tokens
# diversi dagli altri (sezione --run_verbalized), e i cui valori quindi NON
# sono confrontabili barra a barra con il resto: nelle figure vanno separati.
VERBALIZED_LABELS = {
    "Verbalized 1S top-k", "Verbalized 1S top-1", "Verbalized 2S top-k",
    "Verbalized 2S CoT", "Verbalized 2S top-1", "Linguistic 1S",
}


# Famiglie di metodi UQ.
#
# ATTENZIONE, DA VERIFICARE: questa e' la ripartizione standard della
# letteratura (information-based / sample-diversity / riflessivi /
# density-based), non trascritta dal paper -- la pagina arXiv espone solo
# l'abstract. Prima di citarla in tesi va confrontata con la Sezione 3 di
# Vashurin et al., in particolare per le entropie Monte Carlo, che a seconda
# della presentazione stanno fra le information-based (sono entropie di
# sequenza) o fra quelle a diversita' campionaria (richiedono i K campioni).
METHOD_FAMILIES = {
    "Maximum Sequence Probability": "information-based",
    "Perplexity": "information-based",
    "Mean Token Entropy": "information-based",
    "Pointwise Mutual Information": "information-based",
    "Conditional Pointwise Mutual Information": "information-based",
    "Renyi Divergence": "information-based",
    "Fisher-Rao": "information-based",
    "CCP": "information-based",
    "TokenSAR": "information-based",
    "Monte Carlo Sequence Entropy": "information-based",
    "Monte Carlo Normalized Sequence Entropy": "information-based",
    "Semantic Entropy": "diversita' campionaria",
    "BB Semantic Entropy": "diversita' campionaria",
    "SAR": "diversita' campionaria",
    "SentenceSAR": "diversita' campionaria",
    "DegMat NLI Score Entail.": "diversita' campionaria",
    "DegMat Jaccard Score": "diversita' campionaria",
    "EigValLaplacian NLI Score Entail.": "diversita' campionaria",
    "EigValLaplacian Jaccard Score": "diversita' campionaria",
    "Eccentricity NLI Score Entail.": "diversita' campionaria",
    "Eccentricity Jaccard Score": "diversita' campionaria",
    "Lexical Similarity Rouge-L": "diversita' campionaria",
    "Lexical Similarity BLEU": "diversita' campionaria",
    "NumSet": "diversita' campionaria",
    "P(True)": "riflessivi",
    "BB P(True)": "riflessivi",
    "Label Prob.": "riflessivi",
    "Verbalized 1S top-k": "riflessivi",
    "Verbalized 1S top-1": "riflessivi",
    "Verbalized 2S top-k": "riflessivi",
    "Verbalized 2S CoT": "riflessivi",
    "Verbalized 2S top-1": "riflessivi",
    "Linguistic 1S": "riflessivi",
    "Mahalanobis Distance - Decoder": "density-based",
    "RDE - Decoder": "density-based",
    "Relative Mahalanobis Distance - Decoder": "density-based",
    "HUQ-MD - Decoder": "density-based",
}

FAMILY_ORDER = ["information-based", "diversita' campionaria", "riflessivi", "density-based"]


def family_of(paper_label):
    return METHOD_FAMILIES.get(paper_label, "non classificato")


def load_paper_methods(source_path):
    """Legge la mappa dei metodi del paper dal SORGENTE di main.py, senza
    importarlo.

    Perche' cosi': `PAPER_METHODS` e' la fonte unica di verita' su quali metodi
    vanno in Figura A, quali in Figura B e quali sono esclusi con che
    motivazione. Le figure hanno bisogno di quella mappa, ma importare main.py
    significa caricare torch e lm-polygraph, che senza GPU crashano. Copiare la
    lista qui creerebbe una seconda verita' destinata a divergere alla prima
    modifica. Leggerla dall'AST del sorgente la tiene sincronizzata per
    costruzione.

    Ritorna un DataFrame con: paper_label, figure, status
    ("incluso" | "alias" | "escluso"), alias_of, reason.
    """
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    node = None
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and stmt.targets
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "PAPER_METHODS"):
            node = stmt.value
            break
    if node is None or not isinstance(node, ast.List):
        raise ValueError(f"PAPER_METHODS non trovata in {source_path}")

    righe = []
    for elem in node.elts:
        voce = {}
        for chiave, valore in zip(elem.keys, elem.values):
            nome = chiave.value
            if isinstance(valore, ast.Constant):
                voce[nome] = valore.value
            elif isinstance(valore, ast.Lambda):
                voce[nome] = "__lambda__"
            else:
                voce[nome] = None

        factory = voce.get("factory")
        if factory == "__lambda__":
            status, alias_of = "incluso", None
        elif isinstance(factory, str) and factory.startswith("alias:"):
            status, alias_of = "alias", factory.split("alias:", 1)[1]
        else:
            status, alias_of = "escluso", None

        righe.append({
            "paper_label": voce.get("paper_label"),
            "figure": voce.get("figure"),
            "status": status,
            "alias_of": alias_of,
            "reason": voce.get("reason", ""),
        })
    return pd.DataFrame(righe)


def labels_for_figure(paper_methods, figure_letter):
    """Etichette che il paper mostra in una certa figura, nell'ordine in cui
    compaiono in PAPER_METHODS. Include anche gli esclusi: una figura che
    replica quella del paper deve avere le stesse righe, altrimenti non e'
    confrontabile riga per riga con l'originale."""
    mask = paper_methods["figure"].isin([figure_letter, "AB"])
    return paper_methods[mask].copy()


def find_main_py(results_dir):
    """main.py sta accanto a questo modulo; il fallback copre il caso in cui i
    risultati siano stati copiati altrove senza il codice."""
    accanto = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    if os.path.exists(accanto):
        return accanto
    vicino = os.path.join(os.path.dirname(os.path.abspath(results_dir)), "main.py")
    return vicino if os.path.exists(vicino) else None


# ---------------------------------------------------------------------------
# PRR: definizione del paper, con pareggi risolti in valore atteso.
#
# Vashurin et al. definiscono PRR = (AUC_unc - AUC_rnd) / (AUC_oracle - AUC_rnd):
# 0 per un punteggio casuale, 1 per l'oracolo. Fino al 29/09 il benchmark
# riportava invece il solo AUC_unc (la colonna "prr_0.5" di lm-polygraph), che
# parte dal livello dell'accuracy del modello e ci aggiunge poco: la mediana
# valeva quasi esattamente accuracy + 0.045 in tutte le celle, quindi le figure
# misuravano l'accuracy e non i metodi UQ. lm-polygraph calcola la versione del
# paper con il nome "prr_0.5_normalized"; qui la si ricalcola dagli array
# per-istanza con due differenze documentate:
#
# 1. Pareggi. lm-polygraph ordina le istanze con np.argsort(ue), che a parita'
#    di punteggio usa l'ordine del file: per i metodi a valori discreti
#    (NumSet, o qualunque metodo quando i K campioni coincidono) il PRR
#    dipendeva quindi da come erano disposte le righe. Qui il PRR e' il suo
#    valore atteso su tutti gli ordinamenti possibili dei pareggi. Siccome il
#    PRR e' lineare nel vettore delle qualita' ordinate, quel valore atteso si
#    ottiene esattamente sostituendo ogni gruppo di pareggi con la sua qualita'
#    media: nessuna permutazione casuale, nessun seed.
# 2. AUC casuale. lm-polygraph la stima mediando 1000 permutazioni casuali. Il
#    suo valore atteso e' esattamente la qualita' media (ogni sottoinsieme
#    trattenuto a caso ha in media la qualita' media), e qui si usa quello:
#    stesso numero senza rumore di Monte Carlo, e abbastanza veloce da stare
#    dentro un bootstrap.
#
# Senza pareggi e con molte istanze i due calcoli coincidono con quelli della
# libreria (verificato da check_prr_implementation).
# ---------------------------------------------------------------------------

UE_NAN_FILL = -1e7  # convenzione di lm-polygraph (_delete_nans): NaN = massima confidenza


def _prepare(ue, quality):
    """Stesso preprocessing di lm-polygraph: si scartano le istanze con qualita'
    NaN; i NaN del punteggio diventano -1e7 (massima confidenza, vedi la nota
    sui metodi verbalized in main._prr_from_arrays); qualita' riscalata min-max."""
    ue = np.nan_to_num(np.asarray(ue, dtype=float), nan=UE_NAN_FILL,
                       neginf=UE_NAN_FILL, posinf=-UE_NAN_FILL)
    q = np.asarray(quality, dtype=float)
    keep = ~np.isnan(q)
    ue, q = ue[keep], q[keep]
    if len(q) == 0:
        return None, None
    qmin, qmax = q.min(), q.max()
    if qmax == qmin:
        return None, None
    return ue, (q - qmin) / (qmax - qmin)


def _auc_from_sorted(q_sorted, max_rejection):
    """Area della curva prediction-rejection, dato il vettore delle qualita'
    gia' ordinato per incertezza crescente. Stesso calcolo di
    PredictionRejectionArea: media delle qualita' trattenute rifiutando
    0, 1, ..., int(n * max_rejection) istanze."""
    n = len(q_sorted)
    n_rej = int(max_rejection * n)
    if n_rej == 0:
        return float(np.mean(q_sorted))
    csum = np.cumsum(q_sorted)
    # lm-polygraph somma i punteggi da n - n_rej + 1 a n istanze trattenute
    # (non include il caso "n - n_rej"): replicato identico.
    kept = np.arange(n - n_rej + 1, n + 1)
    return float(np.mean(csum[kept - 1] / kept))


def _tie_averaged(ue, q):
    """Qualita' ordinate per incertezza crescente, con ogni gruppo di punteggi
    uguali sostituito dalla sua media (valore atteso sugli spareggi)."""
    order = np.argsort(ue, kind="stable")
    u_s, q_s = ue[order], q[order]
    new_group = np.empty(len(u_s), dtype=bool)
    new_group[0] = True
    new_group[1:] = u_s[1:] != u_s[:-1]
    gid = np.cumsum(new_group) - 1
    sums = np.bincount(gid, weights=q_s)
    counts = np.bincount(gid)
    return (sums / counts)[gid]


def prr_components(ue, quality, max_rejection=0.5):
    """(AUC del metodo, AUC casuale, AUC dell'oracolo) su una cella, oppure
    tre NaN se la qualita' e' costante (non c'e' nulla da ordinare)."""
    ue, q = _prepare(ue, quality)
    if q is None:
        return np.nan, np.nan, np.nan
    auc = _auc_from_sorted(_tie_averaged(ue, q), max_rejection)
    oracle = _auc_from_sorted(_tie_averaged(-q, q), max_rejection)
    random = float(np.mean(q))
    return auc, random, oracle


def prr_normalized(ue, quality, max_rejection=0.5):
    """PRR come definito nel paper: 0 = casuale, 1 = oracolo, negativo =
    peggio del caso."""
    auc, random, oracle = prr_components(ue, quality, max_rejection)
    if not np.isfinite(auc) or oracle == random:
        return np.nan
    return (auc - random) / (oracle - random)


def prr_raw(ue, quality, max_rejection=0.5):
    """Solo l'area (la vecchia colonna 'prr_0.5'), con pareggi in valore
    atteso. Conservata per confronto con le run precedenti."""
    return prr_components(ue, quality, max_rejection)[0]


def distinct_fraction(ue):
    """Frazione di punteggi distinti sulle istanze. Vicina a 0 = il metodo da'
    quasi sempre lo stesso voto e quindi non ordina nulla: il suo PRR va letto
    con cautela anche se il numero sembra buono."""
    u = np.asarray(ue, dtype=float)
    u = u[np.isfinite(u)]
    if len(u) == 0:
        return np.nan
    return len(np.unique(u)) / len(u)


def bootstrap_prr_ci(ue, quality, max_rejection=0.5, n_resamples=1000, seed=3407,
                     alpha=0.05, min_valid_fraction=0.5):
    """Intervallo bootstrap percentile sul PRR normalizzato.

    Si ricampionano gli indici delle domande e si ricalcolano AUC, oracolo e
    caso su ogni ricampionamento: normalizzare con i valori della cella intera
    sottostimerebbe l'incertezza. Se meno di meta' dei ricampionamenti e'
    calcolabile (qualita' costante) l'intervallo e' NaN."""
    ue = np.asarray(ue, dtype=float)
    quality = np.asarray(quality, dtype=float)
    n = len(quality)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = prr_normalized(ue[idx], quality[idx], max_rejection)
        if np.isfinite(v):
            values.append(v)
    if len(values) < n_resamples * min_valid_fraction:
        return np.nan, np.nan
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


def check_prr_implementation(n_checks=200, seed=3407):
    """Controlli di sanita' sul PRR, da eseguire prima di fidarsi dei numeri.

    1. Punteggio casuale -> PRR vicino a 0; oracolo (ue = -qualita') -> 1;
       punteggio invertito (ue = qualita') -> negativo.
    2. Senza pareggi l'area coincide con la formula di lm-polygraph riscritta
       riga per riga (reference_auc).
    3. Il valore con i pareggi e' la media delle permutazioni degli spareggi.
    Solleva AssertionError se qualcosa non torna."""
    rng = np.random.default_rng(seed)

    # 1. casuale / oracolo / invertito
    q = (rng.random(4000) < 0.4).astype(float)
    randoms = [prr_normalized(rng.random(4000), q) for _ in range(20)]
    assert abs(np.mean(randoms)) < 0.02, f"PRR di un punteggio casuale = {np.mean(randoms):.3f}, atteso ~0"
    assert np.isclose(prr_normalized(-q, q), 1.0), "PRR dell'oracolo diverso da 1"
    assert prr_normalized(q, q) < 0, "PRR di un punteggio invertito non negativo"
    qc = rng.random(500)
    assert np.isclose(prr_normalized(-qc, qc), 1.0), "PRR dell'oracolo (qualita' continua) diverso da 1"

    # 2. equivalenza con la formula della libreria, senza pareggi
    def reference_auc(ue, quality, max_rejection=0.5):
        ue = np.nan_to_num(np.asarray(ue, float), nan=UE_NAN_FILL)
        quality = np.asarray(quality, float)
        keep = ~np.isnan(quality)
        ue, t = ue[keep], quality[keep]
        t = (t - t.min()) / (t.max() - t.min())
        num_obs = len(ue)
        num_rej = int(max_rejection * num_obs)
        sorted_metrics = t[np.argsort(ue)]
        cumsum = np.cumsum(sorted_metrics)[-num_rej:]
        scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
        return np.sum(scores) / num_rej

    worst = 0.0
    for t in range(n_checks):
        n = int(rng.integers(20, 300))
        qual = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(float)
        if qual.min() == qual.max():
            continue
        ue = rng.normal(size=n)  # continuo: niente pareggi
        worst = max(worst, abs(prr_raw(ue, qual) - reference_auc(ue, qual)))
    assert worst < 1e-12, f"area diversa dalla formula di lm-polygraph (scarto {worst:.2e})"

    # 3. pareggi = media sugli spareggi
    qual = (rng.random(60) < 0.5).astype(float)
    ue = rng.integers(0, 4, size=60).astype(float)  # molti pareggi
    perms = []
    for _ in range(4000):
        jitter = rng.random(60) * 1e-6
        perms.append(prr_raw(ue + jitter, qual))
    assert abs(np.mean(perms) - prr_raw(ue, qual)) < 2e-3, \
        "il PRR con pareggi non e' la media sugli spareggi"
    return worst


def error_rate_most_confident(ue, quality, correctness_threshold=0.5, quantile=0.10):
    """Frazione di risposte SBAGLIATE fra il `quantile` di risposte a cui il
    metodo da' piu' fiducia (incertezza piu' bassa).

    Sostituisce il vecchio silent failure rate ("quale frazione degli errori
    finisce nel decile piu' confidente"), che ha un tetto: con il 90% di
    risposte sbagliate al massimo un decimo degli errori puo' stare nel 10%
    piu' confidente, quindi il valore massimo e' ~0.11 e quello del caso 0.10,
    e la metrica non distingue nulla proprio nelle celle cliniche. Questa si
    confronta con l'error rate complessivo (error_rate_overall): se il metodo
    funziona e' piu' bassa. Si legge come "fra le risposte date con piu'
    sicurezza, quante sono sbagliate".

    Pareggi sul bordo in valore atteso (ogni istanza a pari merito pesa
    posti_rimasti / istanze_a_pari_merito). NaN se il punteggio e' costante.

    Punteggi NaN (confidenza verbalized non estraibile): stessa convenzione del
    PRR, cioe' massima confidenza (UE_NAN_FILL). Prima venivano scartati, e
    l'error rate del 10% piu' confidente era calcolato su un insieme diverso da
    quello di error_rate_overall: se i parse failure cadevano sulle risposte
    sbagliate, il metodo sembrava funzionare (0.17 contro 0.50) mentre il PRR
    dello stesso metodo era negativo. Cosi' le due metriche e il PRR vedono le
    stesse istanze; la frazione di NaN e' riportata a parte (nan_rate)."""
    ue = np.nan_to_num(np.asarray(ue, dtype=float), nan=UE_NAN_FILL,
                       neginf=UE_NAN_FILL, posinf=-UE_NAN_FILL)
    quality = np.asarray(quality, dtype=float)
    keep = ~np.isnan(quality)
    if keep.sum() == 0:
        return np.nan
    u, q = ue[keep], quality[keep]
    if np.all(u == u[0]):
        return np.nan
    wrong = (q < correctness_threshold).astype(float)
    n = len(u)
    k = int(np.ceil(quantile * n))
    cutoff = np.sort(u)[k - 1]
    below = u < cutoff
    tied = u == cutoff
    weight = below.astype(float)
    weight[tied] = (k - below.sum()) / tied.sum()
    return float((weight * wrong).sum() / k)


def error_rate_overall(quality, correctness_threshold=0.5):
    q = np.asarray(quality, dtype=float)
    q = q[~np.isnan(q)]
    return float(np.mean(q < correctness_threshold)) if len(q) else np.nan


def silent_failure_rate(ue, quality, correctness_threshold=0.5, quantile=0.10):
    """VECCHIA METRICA, conservata solo per confronto con le run precedenti:
    nelle figure si usa error_rate_most_confident (vedi sopra il motivo).

    Frazione delle risposte SBAGLIATE che finisce nel decile piu'
    confidente del metodo (incertezza piu' bassa: tutti gli stimatori di
    lm-polygraph restituiscono incertezza, valore alto = piu' incerto).

    Il decile e' definito come le k = ceil(quantile * n) istanze piu'
    confidenti, non come "tutte le istanze con incertezza <= 10mo percentile".
    La differenza conta quando il metodo assegna lo stesso punteggio a molte
    istanze (es. metodi basati sui campioni quando i K campioni sono identici):
    con la soglia sul percentile finivano "nel decile" anche il 90-100% delle
    istanze, e il silent failure rate saliva artificialmente verso 1.

    I pareggi sul bordo del decile vengono risolti in valore atteso: se servono
    m posti e ci sono t istanze a pari merito, ciascuna conta m/t. E'
    equivalente alla media su tutti gli spareggi casuali possibili, quindi il
    risultato e' deterministico e non dipende da un seed.

    Ritorna NaN se il metodo e' costante su tutte le istanze (non ordina nulla,
    quindi "il suo decile piu' confidente" non esiste) o se non ci sono errori."""
    ue = np.asarray(ue, dtype=float)
    quality = np.asarray(quality, dtype=float)
    keep = ~np.isnan(quality) & ~np.isnan(ue)
    if keep.sum() == 0:
        return np.nan
    u, q = ue[keep], quality[keep]
    wrong = q < correctness_threshold
    if wrong.sum() == 0:
        return np.nan
    if np.all(u == u[0]):
        return np.nan
    n = len(u)
    k = int(np.ceil(quantile * n))
    cutoff = np.sort(u)[k - 1]
    below = u < cutoff
    tied = u == cutoff
    slots = k - below.sum()
    weight = below.astype(float)
    weight[tied] = slots / tied.sum()
    return float((weight * wrong).sum() / wrong.sum())


# ---------------------------------------------------------------------------
# Trasferimento della classifica dei metodi (Kendall tau) fra due modelli.
#
# Fino al 29/09 il tau era calcolato sulla MEDIA fra dataset del PRR grezzo,
# con il p-value di scipy. Due problemi: (1) la media fra dataset di un PRR non
# normalizzato mescola scale diverse (un dataset facile pesa piu' di uno
# difficile); (2) il p-value assume che i 26 metodi siano osservazioni
# indipendenti, mentre molti condividono gli stessi dati (le stesse K
# generazioni, la stessa matrice NLI), quindi e' troppo ottimista. Qui il tau
# si calcola PER DATASET, e la sua incertezza con un bootstrap appaiato sulle
# domande: le due classifiche sono ottenute sulle stesse domande (stesso seed
# di campionamento per tutti i modelli), si ricampionano le domande e si
# ricalcolano entrambe. L'intervallo che ne esce tiene conto della
# dipendenza fra i metodi senza doverla modellare.
# ---------------------------------------------------------------------------

def _kendall(a, b):
    from scipy.stats import kendalltau
    tau, _ = kendalltau(a, b)
    return tau


def kendall_tau_with_ci(cell_a, cell_b, methods, max_rejection=0.5, n_resamples=200,
                        seed=3407, alpha=0.05):
    """tau fra le classifiche per PRR dei `methods` in due celle (stesso
    dataset, modelli diversi) e intervallo bootstrap appaiato sulle domande.

    cell_a, cell_b: DataFrame per-istanza con colonne quality, instance_index
    (se manca, si assume che le righe siano gia' nello stesso ordine) e una
    colonna per metodo."""
    if "instance_index" in cell_a.columns and "instance_index" in cell_b.columns:
        common = np.intersect1d(cell_a["instance_index"], cell_b["instance_index"])
        a = cell_a.set_index("instance_index").loc[common]
        b = cell_b.set_index("instance_index").loc[common]
    else:
        if len(cell_a) != len(cell_b):
            return np.nan, np.nan, np.nan, 0
        a, b = cell_a.reset_index(drop=True), cell_b.reset_index(drop=True)
    qa = a["quality"].to_numpy(float)
    qb = b["quality"].to_numpy(float)
    ua = {m: a[m].to_numpy(float) for m in methods}
    ub = {m: b[m].to_numpy(float) for m in methods}

    def tau_on(idx):
        pa = [prr_normalized(ua[m][idx], qa[idx], max_rejection) for m in methods]
        pb = [prr_normalized(ub[m][idx], qb[idx], max_rejection) for m in methods]
        pa, pb = np.array(pa), np.array(pb)
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 3:
            return np.nan
        return _kendall(pa[ok], pb[ok])

    n = len(qa)
    point = tau_on(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [tau_on(rng.integers(0, n, size=n)) for _ in range(n_resamples)]
    boots = np.array([t for t in boots if np.isfinite(t)])
    if len(boots) < n_resamples // 2:
        return point, np.nan, np.nan, n
    return (point, float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))), n)


# ---------------------------------------------------------------------------
# Severita' a parita' di difficolta'.
#
# Le due celle di severita' (es. MedQAbstain-LT e MedQAbstain-Safe) contengono
# domande DIVERSE: se le domande pericolose sono anche piu' difficili, un calo
# del PRR fra le due non si puo' attribuire alla severita'. Qui la difficolta'
# di una domanda e' il numero di ALTRI modelli (tutti tranne quello valutato)
# che la azzeccano: usare anche il modello valutato renderebbe l'analisi
# circolare. Per ogni livello di difficolta' si tengono tante domande quante
# ne ha il piu' povero dei due strati (sottocampionamento casuale, ripetuto
# n_rep volte e mediato), cosi' che i due strati abbiano la stessa
# distribuzione di difficolta'; poi si ricalcola il PRR di ogni metodo.
# ---------------------------------------------------------------------------

def _aligned_correctness(per, dataset, models, threshold):
    """Matrice (domande x modelli) di correttezza 0/1 per un dataset, con le
    domande allineate fra modelli (instance_index se c'e', altrimenti ordine)."""
    cols = {}
    for m in models:
        cell = per[(per["model"] == m) & (per["dataset"] == dataset)]
        if cell.empty:
            return None, None
        key = cell["instance_index"].to_numpy() if "instance_index" in cell.columns \
            else np.arange(len(cell))
        cols[m] = pd.Series((cell["quality"].to_numpy(float) >= threshold).astype(float), index=key)
    frame = pd.DataFrame(cols).dropna()
    return frame, frame.index.to_numpy()


def severity_matched(per, dataset_a, dataset_b, max_rejection=0.5, threshold=0.5,
                     n_rep=50, seed=3407):
    """PRR di ogni metodo e modello sui due strati, completo e a parita' di
    difficolta'. Ritorna un DataFrame (vuoto se servono dati che mancano:
    almeno tre modelli presenti su entrambi i dataset)."""
    models = sorted(set(per[per["dataset"] == dataset_a]["model"])
                    & set(per[per["dataset"] == dataset_b]["model"]))
    if len(models) < 3:
        return pd.DataFrame()
    corr_a, keys_a = _aligned_correctness(per, dataset_a, models, threshold)
    corr_b, keys_b = _aligned_correctness(per, dataset_b, models, threshold)
    if corr_a is None or corr_b is None:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    meta = {"model", "dataset", "instance_index", "quality", "greedy_text"}
    for m in models:
        others = [o for o in models if o != m]
        diff_a = corr_a[others].sum(axis=1).to_numpy()
        diff_b = corr_b[others].sum(axis=1).to_numpy()
        cell_a = per[(per["model"] == m) & (per["dataset"] == dataset_a)]
        cell_b = per[(per["model"] == m) & (per["dataset"] == dataset_b)]
        if "instance_index" in cell_a.columns:
            cell_a = cell_a.set_index("instance_index").loc[keys_a]
            cell_b = cell_b.set_index("instance_index").loc[keys_b]
        else:
            cell_a = cell_a.reset_index(drop=True).loc[keys_a]
            cell_b = cell_b.reset_index(drop=True).loc[keys_b]
        methods = [c for c in cell_a.columns if c not in meta and c in cell_b.columns
                   and cell_a[c].notna().any() and cell_b[c].notna().any()]
        qa, qb = cell_a["quality"].to_numpy(float), cell_b["quality"].to_numpy(float)
        levels = sorted(set(diff_a) | set(diff_b))
        per_level = {d: (np.flatnonzero(diff_a == d), np.flatnonzero(diff_b == d)) for d in levels}
        n_matched = sum(min(len(ia), len(ib)) for ia, ib in per_level.values())
        if n_matched < 20:
            continue
        samples = []
        for _ in range(n_rep):
            sa, sb = [], []
            for ia, ib in per_level.values():
                k = min(len(ia), len(ib))
                if k:
                    sa.append(rng.choice(ia, k, replace=False))
                    sb.append(rng.choice(ib, k, replace=False))
            samples.append((np.concatenate(sa), np.concatenate(sb)))
        for meth in methods:
            ua, ub = cell_a[meth].to_numpy(float), cell_b[meth].to_numpy(float)
            pa = [prr_normalized(ua[i], qa[i], max_rejection) for i, _ in samples]
            pb = [prr_normalized(ub[j], qb[j], max_rejection) for _, j in samples]
            rows.append({
                "model": m, "dataset_a": dataset_a, "dataset_b": dataset_b, "method": meth,
                "prr_a_full": prr_normalized(ua, qa, max_rejection),
                "prr_b_full": prr_normalized(ub, qb, max_rejection),
                "prr_a_matched": float(np.nanmean(pa)), "prr_b_matched": float(np.nanmean(pb)),
                "acc_a_full": float(np.nanmean(qa)), "acc_b_full": float(np.nanmean(qb)),
                "acc_a_matched": float(np.mean([np.nanmean(qa[i]) for i, _ in samples])),
                "acc_b_matched": float(np.mean([np.nanmean(qb[j]) for _, j in samples])),
                "n_a": len(qa), "n_b": len(qb), "n_matched_per_stratum": n_matched,
            })
    return pd.DataFrame(rows)
