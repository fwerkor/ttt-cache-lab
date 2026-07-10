import subprocess
import sys

import pytest

from ttt_cache_lab.data.synthetic import SyntheticTaskFactory
from ttt_cache_lab.models.toy import ToyBackend


@pytest.mark.parametrize("task", ["passkey", "key_value", "multi_needle", "variable_tracking"])
def test_toy_backend_scores_prepared_synthetic_samples(task: str) -> None:
    sample = SyntheticTaskFactory(7).build(
        task,
        num_samples=1,
        context_length=1024,
        answer_length=4,
    )[0]
    backend = ToyBackend(num_layers=4, hidden_size=8, vocab_size=256, seed=7)
    prepared = backend.prepare_sample(sample, context_length=1024)
    output = backend.prefill(prepared.prompt)
    assert backend.score_answer(prepared, output) == 1.0


def test_toy_update_drift_is_stable_across_processes() -> None:
    code = """
from ttt_cache_lab.models.toy import ToyBackend
from ttt_cache_lab.updates.targets import parse_update_target
backend = ToyBackend(num_layers=4, hidden_size=8, vocab_size=16, seed=7)
baseline = backend.prefill('The secret passkey is 1234.\\nAnswer:')
updated = backend.simulate_update(baseline, parse_update_target('lora.k'), update_norm=0.01)
print(float(updated.cache_tensor[0, 0, 0]))
"""
    first = subprocess.check_output([sys.executable, "-c", code], text=True)
    second = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert first == second
