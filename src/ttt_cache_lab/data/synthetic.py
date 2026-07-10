from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSample:
    prompt: str
    answer: str
    metadata: dict[str, Any]


class SyntheticTaskFactory:
    """Generate deterministic long-context diagnostic tasks."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def passkey(self, *, context_length: int, answer_length: int) -> TaskSample:
        answer = "".join(str(self.rng.randrange(10)) for _ in range(answer_length))
        filler_tokens = [f"tok{self.rng.randrange(10_000)}" for _ in range(max(1, context_length // 8))]
        insert_at = self.rng.randrange(len(filler_tokens))
        filler_tokens.insert(insert_at, f"The secret passkey is {answer}.")
        prompt = " ".join(filler_tokens) + "\nQuestion: What is the secret passkey?\nAnswer:"
        return TaskSample(prompt=prompt, answer=answer, metadata={"insert_at": insert_at, "task": "passkey"})

    def key_value(self, *, context_length: int, answer_length: int) -> TaskSample:
        key = f"key_{self.rng.randrange(10_000)}"
        value = "".join(chr(ord("a") + self.rng.randrange(26)) for _ in range(answer_length))
        pairs = [f"key_{i}: value_{self.rng.randrange(10_000)}" for i in range(max(1, context_length // 10))]
        insert_at = self.rng.randrange(len(pairs))
        pairs.insert(insert_at, f"{key}: {value}")
        prompt = "\n".join(pairs) + f"\nQuestion: What is the value for {key}?\nAnswer:"
        return TaskSample(
            prompt=prompt,
            answer=value,
            metadata={"insert_at": insert_at, "task": "key_value", "key": key},
        )

    def multi_needle(self, *, context_length: int, answer_length: int) -> TaskSample:
        needle_count = max(2, min(8, context_length // 512))
        needles = []
        target_index = self.rng.randrange(needle_count)
        for idx in range(needle_count):
            value = "".join(str(self.rng.randrange(10)) for _ in range(answer_length))
            needles.append((f"needle_{idx}", value))
        filler = [f"doc{self.rng.randrange(100_000)}" for _ in range(max(needle_count + 1, context_length // 9))]
        used_positions: set[int] = set()
        for key, value in needles:
            pos = self.rng.randrange(len(filler))
            while pos in used_positions:
                pos = self.rng.randrange(len(filler))
            used_positions.add(pos)
            filler.insert(pos, f"Record {key} has code {value}.")
        target_key, answer = needles[target_index]
        prompt = " ".join(filler) + f"\nQuestion: What code is stored in {target_key}?\nAnswer:"
        return TaskSample(
            prompt=prompt,
            answer=answer,
            metadata={"task": "multi_needle", "needle_count": needle_count, "target": target_key},
        )

    def variable_tracking(self, *, context_length: int, answer_length: int) -> TaskSample:
        variable = f"var_{self.rng.randrange(1000)}"
        updates = max(3, min(16, context_length // 256))
        values = ["".join(chr(ord("a") + self.rng.randrange(26)) for _ in range(answer_length)) for _ in range(updates)]
        lines = [f"Initial state: {variable} = {values[0]}."]
        for idx, value in enumerate(values[1:], start=1):
            distractor = f"var_{self.rng.randrange(1000)}"
            lines.append(f"Step {idx}: {distractor} = tmp_{self.rng.randrange(10_000)}.")
            lines.append(f"Step {idx}: {variable} = {value}.")
        filler = [f"trace_{self.rng.randrange(100_000)}" for _ in range(max(1, context_length // 12))]
        insert_at = self.rng.randrange(len(filler))
        filler.insert(insert_at, "\n".join(lines))
        prompt = "\n".join(filler) + f"\nQuestion: What is the final value of {variable}?\nAnswer:"
        return TaskSample(
            prompt=prompt,
            answer=values[-1],
            metadata={"task": "variable_tracking", "updates": updates, "variable": variable},
        )

    def build(
        self,
        task: str,
        *,
        num_samples: int,
        context_length: int,
        answer_length: int,
    ) -> list[TaskSample]:
        if task == "passkey":
            return [
                self.passkey(context_length=context_length, answer_length=answer_length) for _ in range(num_samples)
            ]
        if task == "key_value":
            return [
                self.key_value(context_length=context_length, answer_length=answer_length) for _ in range(num_samples)
            ]
        if task == "multi_needle":
            return [
                self.multi_needle(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        if task == "variable_tracking":
            return [
                self.variable_tracking(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        raise ValueError(f"Unsupported synthetic task: {task}")
