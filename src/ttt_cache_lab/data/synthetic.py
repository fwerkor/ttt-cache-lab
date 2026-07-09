from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSample:
    prompt: str
    answer: str
    metadata: dict[str, int | str]


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
        raise ValueError(f"Unsupported synthetic task: {task}")
