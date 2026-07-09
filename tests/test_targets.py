from ttt_cache_lab.updates.targets import ModuleKind, parse_update_target


def test_parse_late_target() -> None:
    target = parse_update_target("lora.mlp_late", num_layers=8)
    assert target.kind is ModuleKind.LORA_MLP
    assert target.layer == 7


def test_parse_plain_mlp_late_target() -> None:
    target = parse_update_target("mlp.late", num_layers=8)
    assert target.kind is ModuleKind.MLP
    assert target.layer == 7


def test_parse_middle_target() -> None:
    target = parse_update_target("lora.k_middle", num_layers=9)
    assert target.kind is ModuleKind.LORA_K
    assert target.layer == 4


def test_parse_layer_suffix() -> None:
    target = parse_update_target("attention.k:layer3")
    assert target.kind is ModuleKind.ATTENTION_K
    assert target.layer == 3


def test_parse_composite_targets() -> None:
    assert parse_update_target("lora.qv").kind is ModuleKind.LORA_QV
    assert parse_update_target("lora.attn").kind is ModuleKind.LORA_ATTN
    assert parse_update_target("attention.qv").kind is ModuleKind.ATTENTION_QV
    assert parse_update_target("lora.all_late", num_layers=6).layer == 5
