import warnings; warnings.filterwarnings("ignore")
exec(open("test_integration.py").read().split("# T1:")[0])
import shutil
model.tokenizer.chat_template = ("{% for m in messages %}<|user|>{{ m['content'] }}{% endfor %}"
                                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")
shutil.rmtree("rp", ignore_errors=True)
r = M.run_model_on_dataset(model, "tiny", "DS", examples, cfg, make_args("rp"), {}, use_chat_template=True,
                           estimators_factory=estimators)
assert r[0] is not None
rc = pd.read_csv("rp/run_conditions.csv"); assert rc["prompt_format"].iloc[0] == "chat_template"
M.SAMPLER = "library"
r2 = M.run_model_on_dataset(model, "tiny", "DS", examples, cfg, make_args("rp"), {}, use_chat_template=True,
                            estimators_factory=estimators)
assert r2[0] is not None and "SamplingGenerationCalculator" in ";".join(r2[1]["calculators"].dropna())
print("PATHS OK: chat template + campionatore della libreria")
