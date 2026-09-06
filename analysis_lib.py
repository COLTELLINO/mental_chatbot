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
