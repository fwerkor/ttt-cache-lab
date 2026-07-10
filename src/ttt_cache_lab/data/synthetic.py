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
        pairs = [
            f"key_{i}: value_{self.rng.randrange(10_000)}"
            for i in range(max(1, context_length // 16))
            if f"key_{i}" != key
        ]
        if not pairs:
            pairs.append(f"key_fallback: value_{self.rng.randrange(10_000)}")
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
        distractors: set[str] = set()
        for idx, value in enumerate(values[1:], start=1):
            distractor = f"var_{self.rng.randrange(1000)}"
            while distractor == variable or distractor in distractors:
                distractor = f"var_{self.rng.randrange(1000)}"
            distractors.add(distractor)
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

    def needle_absent(self, *, context_length: int, answer_length: int) -> TaskSample:
        del answer_length
        requested = f"needle_{self.rng.randrange(100_000, 200_000)}"
        records = [
            f"Record needle_{self.rng.randrange(100_000)} has code {self.rng.randrange(10_000):04d}."
            for _ in range(max(4, context_length // 20))
        ]
        prompt = " ".join(records) + (
            f"\nQuestion: What code is stored in {requested}? "
            "Reply NOT_FOUND when the record is absent.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer="NOT_FOUND",
            metadata={"task": "needle_absent", "target": requested},
        )

    def multi_hop_tracing(self, *, context_length: int, answer_length: int) -> TaskSample:
        hop_count = max(3, min(12, context_length // 512))
        used_entities: set[str] = set()

        def next_entity() -> str:
            candidate = f"entity_{self.rng.randrange(1_000_000)}"
            while candidate in used_entities:
                candidate = f"entity_{self.rng.randrange(1_000_000)}"
            used_entities.add(candidate)
            return candidate

        entities = [next_entity() for _ in range(hop_count + 1)]
        answer = "".join(chr(ord("a") + self.rng.randrange(26)) for _ in range(answer_length))
        facts = [f"{entities[index]} points to {entities[index + 1]}." for index in range(hop_count)]
        facts.append(f"{entities[-1]} stores value {answer}.")
        distractors = []
        for _ in range(max(hop_count, context_length // 24)):
            source = next_entity()
            destination = next_entity()
            distractors.append(f"{source} points to {destination}.")
        combined = distractors + facts
        self.rng.shuffle(combined)
        prompt = "\n".join(combined) + (
            f"\nQuestion: Follow the pointer chain beginning at {entities[0]}. "
            "What value is stored at the final entity?\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=answer,
            metadata={"task": "multi_hop_tracing", "hop_count": hop_count},
        )

    def aggregation(self, *, context_length: int, answer_length: int) -> TaskSample:
        del answer_length
        target = f"group_{self.rng.randrange(1000)}"
        target_count = self.rng.randrange(3, 12)
        lines = [f"Event belongs to {target}." for _ in range(target_count)]
        for _ in range(max(target_count, context_length // 10)):
            distractor = f"group_{self.rng.randrange(1000)}"
            while distractor == target:
                distractor = f"group_{self.rng.randrange(1000)}"
            lines.append(f"Event belongs to {distractor}.")
        self.rng.shuffle(lines)
        prompt = "\n".join(lines) + (
            f"\nQuestion: How many events belong to {target}? Reply with one integer.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=str(target_count),
            metadata={"task": "aggregation", "target": target, "count": target_count},
        )

    def common_words(self, *, context_length: int, answer_length: int) -> TaskSample:
        common_count = max(2, min(6, answer_length))
        common = [f"shared_{self.rng.randrange(100_000)}" for _ in range(common_count)]
        list_count = max(3, min(8, context_length // 512))
        lists: list[list[str]] = []
        for _ in range(list_count):
            unique = [f"item_{self.rng.randrange(1_000_000)}" for _ in range(max(3, context_length // 128))]
            items = unique + common
            self.rng.shuffle(items)
            lists.append(items)
        prompt = "\n".join(
            f"List {index + 1}: {', '.join(items)}" for index, items in enumerate(lists)
        ) + "\nQuestion: Which words occur in every list? Return a comma-separated set.\nAnswer:"
        return TaskSample(
            prompt=prompt,
            answer=", ".join(sorted(common)),
            metadata={"task": "common_words", "list_count": list_count, "common_count": common_count},
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
        if task == "needle_absent":
            return [
                self.needle_absent(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        if task == "multi_hop_tracing":
            return [
                self.multi_hop_tracing(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        if task == "aggregation":
            return [
                self.aggregation(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        if task == "common_words":
            return [
                self.common_words(context_length=context_length, answer_length=answer_length)
                for _ in range(num_samples)
            ]
        raise ValueError(f"Unsupported synthetic task: {task}")
