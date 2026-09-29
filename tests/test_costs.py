import warnings; warnings.filterwarnings("ignore")
exec(open("test_integration.py").read().split("# T1:")[0])
from lm_polygraph.estimators import MeanPointwiseMutualInformation, PTrue
def est():
    return [MaximumSequenceProbability(), MonteCarloSequenceEntropy(), MeanPointwiseMutualInformation()]
import shutil; shutil.rmtree("rc", ignore_errors=True)
prr, tim, st, pi = M.run_model_on_dataset(model, "tiny", "DS", examples, cfg, make_args("rc"), {},
                                          use_chat_template=False, estimators_factory=est)
t = tim.set_index("estimator")
print(t[["seconds_full_standalone","needs_sampling","needs_nli","needs_cross_encoder","needs_extra_forward","calculators"]].to_string())
assert t.loc["MonteCarloSequenceEntropy","needs_sampling"] and not t.loc["MaximumSequenceProbability","needs_sampling"]
assert t.loc["MeanPointwiseMutualInformation","needs_extra_forward"]
assert t.loc["MonteCarloSequenceEntropy","seconds_full_standalone"] > t.loc["MaximumSequenceProbability","seconds_full_standalone"]
assert t.loc["MeanPointwiseMutualInformation","seconds_full_standalone"] > t.loc["MaximumSequenceProbability","seconds_full_standalone"]
# chiusure dei 26 metodi reali
conts = M.default_stat_calculators("hfcache")
for m in M.PAPER_METHODS:
    if callable(m["factory"]):
        c = M.estimator_calculator_closure(m["factory"](), conts)
        if m["paper_label"] in ("CCP","TokenSAR","Label Prob."):
            print(m["paper_label"], sorted(c))
print("COSTI OK")
