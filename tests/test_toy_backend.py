import subprocess
import sys


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
