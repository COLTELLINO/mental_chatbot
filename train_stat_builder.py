"""Builder custom per TrainingStatisticExtractionCalculator, richiesto da
Mahalanobis Distance / RDE / Relative Mahalanobis Distance. lm-polygraph NON
registra questo calculator di default per i modelli Whitebox (verificato nel
sorgente di register_default_stat_calculators: EmbeddingsCalculator e
TrainingStatisticExtractionCalculator compaiono solo nel proprio builder
ufficiale lm_polygraph.defaults.stat_calculator_builders.
default_TrainingStatisticExtractionCalculator, mai nel ramo "Whitebox" di
register_default_stat_calculators stesso), quindi va aggiunto a mano.

lm-polygraph costruisce ogni stat calculator "custom" importando
dinamicamente un modulo con import_module(nome_modulo) e chiamandone
load_stat_calculator(cfg, environment) (vedi
lm_polygraph.utils.factory_stat_calculator.create_stat_calculator). Questo
modulo e' esattamente quel punto di ingresso per il nostro caso: costruisce
un train_dataset (dallo stesso dataset/loader usato per il test, ma un
campione indipendente) e un background_train_dataset (WikiText, generico e
fuori dominio, usato solo da Relative Mahalanobis Distance) e li passa a
TrainingStatisticExtractionCalculator, seguendo lo stesso pattern del
builder ufficiale (verificato via sorgente GitHub), adattato per riusare i
nostri loader invece di lm_polygraph.utils.dataset.Dataset.load().

Caveat noto (non risolvibile senza modificare la libreria): il codice di
TrainingStatisticExtractionCalculator confronta erroneamente la LISTA
["train_", "background_train_"] anziche' l'elemento corrente del loop
contro la stringa "train_" (vedi statistic_extraction.py, riga con
`train_max_new_tokens = (max_new_tokens if datasets_name == "train_" ...`),
quindi il ramo True non scatta mai: sia il train set in-domain sia il
background set vengono generati con max_new_tokens=100 (il default
dell'argomento background_train_dataset_max_new_tokens), MAI col
max_new_tokens del dataset corrente (es. 200 per GSM8k). Non e' un nostro
bug, e' nella libreria; accettato come limitazione documentata."""

from lm_polygraph.utils.dataset import Dataset as PolygraphDataset
from lm_polygraph.stat_calculators.statistic_extraction import (
    TrainingStatisticExtractionCalculator,
)

from dataset_prep import (
    DATASETS,
    SAFETY_DATASETS,
    prepare_wikitext_background,
    format_prompt,
    format_chat_prompt,
)

# dataset_name puo' arrivare sia dalla pipeline principale (CoQA/TriviaQA/
# MMLU/GSM8k, in DATASETS) sia dal confronto safety (MedQA/MedicationQA, in
# SAFETY_DATASETS) -- stesso identico loader/plain_suffix in entrambi i
# casi, quindi basta cercare in entrambi i dizionari.
_ALL_DATASET_CFGS = {**DATASETS, **SAFETY_DATASETS}


def _build_polygraph_dataset(examples, cfg, use_chat_template, tokenizer):
    if use_chat_template:
        x = [format_chat_prompt(tokenizer, ex["content"]) for ex in examples]
    else:
        x = [format_prompt(ex["content"], cfg.plain_suffix) for ex in examples]
    y = [ex["reference"] for ex in examples]
    return PolygraphDataset(x, y, batch_size=cfg.batch_size)


def load_stat_calculator(cfg, environment):
    model = environment.model
    tokenizer = model.tokenizer

    # Campione INDIPENDENTE dal test set (seed diverso), stesso loader e
    # stesso stile di prompt del dataset corrente. Nota: non e' uno split
    # train/test rigoroso -- per dataset piccoli (es. CoQA, 500 righe di
    # validation) puo' esserci una sovrapposizione parziale per puro caso
    # con le 100 righe usate per il test. Caveat da citare in tesi.
    train_loader = _ALL_DATASET_CFGS[cfg.dataset_name]["loader"]
    train_examples = train_loader(cfg.n_train, cfg.seed + 1, cache_dir=cfg.cache_dir)
    train_dataset = _build_polygraph_dataset(train_examples, cfg, cfg.use_chat_template, tokenizer)

    # Background set generico (WikiText) per Relative Mahalanobis Distance.
    # Stesso identico per ogni dataset (seed fisso, non cfg.seed+1) --
    # rappresenta "testo generico" indipendentemente dal task in corso.
    bg_examples = prepare_wikitext_background(cfg.n_background, cfg.seed, cache_dir=cfg.cache_dir)
    background_dataset = _build_polygraph_dataset(bg_examples, cfg, cfg.use_chat_template, tokenizer)

    return TrainingStatisticExtractionCalculator(
        train_dataset=train_dataset,
        background_train_dataset=background_dataset,
    )
