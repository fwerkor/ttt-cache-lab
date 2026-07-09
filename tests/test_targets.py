from ttt_cache_lab.updates.targets import ModuleKind, parse_update_target


def test_parse_late_target() -> None:
    target = parse_update_target("lora.mlp_late", num_layers=8)
    assert target.kind is ModuleKind.LORA_MLP
    assert target.layer == 7


def test_parse_layer_suffix() -> None:
    target = parse_update_target("attention.k:layer3")
    assert target.kind is ModuleKind.ATTENTION_K
    assert target.layer == 3
