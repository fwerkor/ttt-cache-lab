from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _norm_slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def generate(spec_path: Path, *, profiles: set[str] | None = None) -> list[Path]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    output_root = Path(spec["output_root"])
    config_root = Path(spec["config_root"])
    strategies = list(spec["strategies"])
    version_steps = list(spec["version_steps"])
    norm_control = spec.get("norm_control")
    generated: list[Path] = []

    for profile_name, profile in spec["profiles"].items():
        if profiles is not None and profile_name not in profiles:
            continue
        template_path = Path(profile["template"])
        template: dict[str, Any] = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        for norm_value in profile["norms"]:
            norm = float(norm_value)
            slug = _norm_slug(norm)
            config = yaml.safe_load(yaml.safe_dump(template))
            config["name"] = f"delta-quantization-{profile_name}-{slug}"
            config["output_dir"] = str(output_root / profile_name / slug)
            config["data"]["num_samples"] = int(profile["num_samples"])
            config["updates"]["targets"] = list(profile["targets"])
            config["updates"]["update_norm"] = norm
            if norm_control is not None:
                config["adapter"]["norm_control"] = str(norm_control)
            config["cache"]["strategies"] = strategies
            config["version_steps"] = version_steps
            destination = config_root / profile_name / f"{slug}.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            generated.append(destination)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concrete configs for the delta quantization sweep.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/sweeps/ascend_delta_quantization.yaml"),
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Generate only the named profile. Repeat to select multiple profiles.",
    )
    args = parser.parse_args()
    selected = set(args.profiles) if args.profiles else None
    for path in generate(args.spec, profiles=selected):
        print(path)


if __name__ == "__main__":
    main()
