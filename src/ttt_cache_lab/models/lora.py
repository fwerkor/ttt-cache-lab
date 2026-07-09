from __future__ import annotations

from typing import Any


class LoraLinearMixin:
    """Marker mixin used only for isinstance-free duck typing in tests/docs."""


def make_lora_linear(torch: Any, nn: Any, base: Any, *, rank: int, alpha: float) -> Any:
    class LoraLinear(nn.Module, LoraLinearMixin):  # type: ignore[misc]
        def __init__(self, base_module: Any) -> None:
            super().__init__()
            self.base = base_module
            for param in self.base.parameters():
                param.requires_grad_(False)
            in_features = int(base_module.in_features)
            out_features = int(base_module.out_features)
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / max(1, rank)
            self.lora_a = nn.Parameter(torch.empty(rank, in_features, dtype=base_module.weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=base_module.weight.dtype))
            self.capture_lora_input = False
            self.cached_lora_input = None
            self.lora_name = ""
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        def forward(self, x: Any) -> Any:
            if self.capture_lora_input:
                self.cached_lora_input = x.detach()
            base_out = self.base(x)
            lora_hidden = torch.nn.functional.linear(x, self.lora_a)
            lora_out = torch.nn.functional.linear(lora_hidden, self.lora_b)
            return base_out + lora_out * self.scaling

        def reset_lora(self) -> None:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
                self.lora_b.zero_()
            self.cached_lora_input = None
            self.capture_lora_input = False

        def lora_parameters(self) -> list[Any]:
            return [self.lora_a, self.lora_b]

        def lora_state(self) -> dict[str, Any]:
            return {
                "a": self.lora_a.detach().clone(),
                "b": self.lora_b.detach().clone(),
                "scaling": float(self.scaling),
                "rank": int(self.rank),
                "alpha": float(self.alpha),
            }

        def lora_delta_output(self, cached_input: Any, old_state: dict[str, Any]) -> Any:
            old_a = old_state["a"].to(device=cached_input.device, dtype=cached_input.dtype)
            old_b = old_state["b"].to(device=cached_input.device, dtype=cached_input.dtype)
            old_scaling = float(old_state.get("scaling", self.scaling))
            new_hidden = torch.nn.functional.linear(cached_input, self.lora_a)
            new_out = torch.nn.functional.linear(new_hidden, self.lora_b) * self.scaling
            old_hidden = torch.nn.functional.linear(cached_input, old_a)
            old_out = torch.nn.functional.linear(old_hidden, old_b) * old_scaling
            return new_out - old_out

    return LoraLinear(base)


def is_lora_linear(module: Any) -> bool:
    return hasattr(module, "lora_parameters") and hasattr(module, "reset_lora")
