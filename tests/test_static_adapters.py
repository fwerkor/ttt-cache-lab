from pathlib import Path

from ttt_cache_lab.configs import VersionedExperimentConfig
from ttt_cache_lab.experiments.static_adapters import StaticAdapterExperimentRunner


def test_static_adapter_runner_records_real_cache_hits(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-static",
            "experiment_id": "unit_static",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "update_norm": 0.01},
            "cache": {"strategies": ["no_adaptation", "adapter_specific_cache", "static_base_delta"]},
            "adapter": {
                "update_mode": "static_lora",
                "static_adapter_sequence": [0, 1, 2, 0, 2, 1],
            },
        }
    )
    records = StaticAdapterExperimentRunner(config).run().records
    specific = [record for record in records if record.cache_strategy == "adapter_specific_cache"]
    assert [record.action for record in specific] == [
        "reuse_exact",
        "full_recompute",
        "full_recompute",
        "reuse_exact",
        "reuse_exact",
        "reuse_exact",
    ]
    no_adaptation = [record for record in records if record.cache_strategy == "no_adaptation"]
    assert {record.action for record in no_adaptation} == {"reuse_exact"}
    delta = [record for record in records if record.cache_strategy == "static_base_delta"]
    assert delta[0].action == "reuse_exact"
    assert {record.action for record in delta[1:]} == {"delta_correct"}



def test_static_adapter_capacity_causes_rebuilds(tmp_path: Path) -> None:
    config = VersionedExperimentConfig.model_validate(
        {
            "name": "unit-static-capacity",
            "experiment_id": "unit_static_capacity",
            "seed": 7,
            "output_dir": tmp_path,
            "model": {"backend": "toy", "num_layers": 2, "hidden_size": 8, "vocab_size": 32},
            "data": {"task": "passkey", "num_samples": 1, "context_length": 64, "answer_length": 2},
            "updates": {"targets": ["lora.k"], "update_norm": 0.01},
            "cache": {"strategies": ["adapter_specific_cache"], "max_cache_entries": 2},
            "adapter": {
                "update_mode": "static_lora",
                "static_adapter_sequence": [0, 1, 2, 0],
            },
        }
    )
    records = StaticAdapterExperimentRunner(config).run().records
    assert [record.action for record in records] == [
        "reuse_exact",
        "full_recompute",
        "full_recompute",
        "full_recompute",
    ]
    assert max(record.cache_entry_count for record in records) <= 2
    assert records[-1].evicted_cache_entries > 0
