from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

SyntheticDifficulty = Literal["easy", "medium", "hard"]

_NEUTRAL_ADJECTIVES = (
    "quiet",
    "ordinary",
    "distant",
    "familiar",
    "gentle",
    "modest",
    "plain",
    "calm",
)
_NEUTRAL_SUBJECTS = (
    "forest",
    "garden",
    "library",
    "harbor",
    "meadow",
    "village",
    "workshop",
    "gallery",
)
_NEUTRAL_TOPICS = (
    "weather",
    "books",
    "rivers",
    "windows",
    "music",
    "lanterns",
    "clouds",
    "pathways",
)


def neutral_background_sentences(count: int, *, offset: int = 0) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    sentences: list[str] = []
    for index in range(offset, offset + count):
        adjective = _NEUTRAL_ADJECTIVES[index % len(_NEUTRAL_ADJECTIVES)]
        subject = _NEUTRAL_SUBJECTS[(index // len(_NEUTRAL_ADJECTIVES)) % len(_NEUTRAL_SUBJECTS)]
        topic = _NEUTRAL_TOPICS[
            (index // (len(_NEUTRAL_ADJECTIVES) * len(_NEUTRAL_SUBJECTS)))
            % len(_NEUTRAL_TOPICS)
        ]
        sentences.append(f"{adjective} {subject} {topic}.")
    return sentences


def _difficulty_value(
    difficulty: SyntheticDifficulty,
    *,
    easy: int,
    medium: int,
    hard: int,
) -> int:
    return {"easy": easy, "medium": medium, "hard": hard}[difficulty]


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

    def multi_needle(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        maximum = max(2, min(8, context_length // 512))
        needle_count = min(
            maximum,
            _difficulty_value(difficulty, easy=2, medium=4, hard=8),
        )
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
        prompt = " ".join(filler) + (
            f"\nQuestion: What code is stored in {target_key}? "
            "Reply with only the exact code.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=answer,
            metadata={
                "task": "multi_needle",
                "needle_count": needle_count,
                "target": target_key,
                "synthetic_difficulty": difficulty,
            },
        )

    def variable_tracking(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        variable = f"var_{self.rng.randrange(1000)}"
        maximum = max(3, min(16, context_length // 256))
        updates = min(
            maximum,
            _difficulty_value(difficulty, easy=3, medium=6, hard=16),
        )
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
        filler = neutral_background_sentences(
            max(1, context_length // 12),
            offset=self.rng.randrange(512),
        )
        insert_at = self.rng.randrange(len(filler))
        filler.insert(insert_at, "\n".join(lines))
        prompt = "\n".join(filler) + (
            f"\nQuestion: What is the value in the final assignment to {variable}? "
            "Reply with only the exact value.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=values[-1],
            metadata={
                "task": "variable_tracking",
                "updates": updates,
                "variable": variable,
                "synthetic_difficulty": difficulty,
            },
        )

    def needle_absent(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        del answer_length
        requested = f"needle_{self.rng.randrange(100_000, 200_000)}"
        records = [
            f"Record needle_{self.rng.randrange(100_000)} has code {self.rng.randrange(10_000):04d}."
            for _ in range(max(4, context_length // 20))
        ]
        prompt = " ".join(records) + (
            f"\nQuestion: What code is stored in {requested}? "
            "Use only an exact matching record above. If no record has exactly that key, "
            "reply with NOT_FOUND as the first answer token. Do not infer a code.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer="NOT_FOUND",
            metadata={
                "task": "needle_absent",
                "target": requested,
                "synthetic_difficulty": difficulty,
            },
        )

    def multi_hop_tracing(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        maximum = max(3, min(12, context_length // 512))
        hop_count = min(
            maximum,
            _difficulty_value(difficulty, easy=2, medium=4, hard=12),
        )
        used_entities: set[str] = set()

        def next_entity() -> str:
            candidate = "entity_" + "".join(
                chr(ord("a") + self.rng.randrange(26)) for _ in range(6)
            )
            while candidate in used_entities:
                candidate = "entity_" + "".join(
                    chr(ord("a") + self.rng.randrange(26)) for _ in range(6)
                )
            used_entities.add(candidate)
            return candidate

        entities = [next_entity() for _ in range(hop_count + 1)]
        answer = "".join(chr(ord("a") + self.rng.randrange(26)) for _ in range(answer_length))
        facts = [f"{entities[index]} points to {entities[index + 1]}." for index in range(hop_count)]
        facts.append(f"{entities[-1]} stores value {answer}.")
        hard_distractor_count = max(hop_count, context_length // 24)
        distractor_count = min(
            hard_distractor_count,
            _difficulty_value(
                difficulty,
                easy=4,
                medium=32,
                hard=hard_distractor_count,
            ),
        )
        distractors = []
        for _ in range(distractor_count):
            source = next_entity()
            destination = next_entity()
            distractors.append(f"{source} points to {destination}.")
        neutral = neutral_background_sentences(
            hard_distractor_count - distractor_count,
            offset=self.rng.randrange(512),
        )
        combined = distractors + neutral + facts
        self.rng.shuffle(combined)
        prompt = "\n".join(combined) + (
            f"\nQuestion: Follow the pointer chain beginning at {entities[0]} until an entity stores a value. "
            "Reply with only that stored value, not an entity name or explanation.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=answer,
            metadata={
                "task": "multi_hop_tracing",
                "hop_count": hop_count,
                "synthetic_difficulty": difficulty,
            },
        )

    def aggregation(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        del answer_length
        lower, upper = {
            "easy": (2, 5),
            "medium": (3, 7),
            "hard": (3, 12),
        }[difficulty]
        target_count = self.rng.randrange(lower, upper)
        hard_distractor_count = max(target_count, context_length // 10)
        distractor_count = min(
            hard_distractor_count,
            _difficulty_value(
                difficulty,
                easy=4,
                medium=16,
                hard=hard_distractor_count,
            ),
        )
        ledger = ["TARGET_EVENT" for _ in range(target_count)]
        ledger.extend("OTHER_EVENT" for _ in range(distractor_count))
        self.rng.shuffle(ledger)
        neutral = neutral_background_sentences(
            hard_distractor_count - distractor_count,
            offset=self.rng.randrange(512),
        )
        prompt = "\n".join(
            [
                *neutral,
                "BEGIN LEDGER",
                *ledger,
                "END LEDGER",
                "Question: Count only the lines equal to TARGET_EVENT between BEGIN LEDGER and END LEDGER. "
                "Do not count this question. Reply with one integer.",
                "Answer:",
            ]
        )
        return TaskSample(
            prompt=prompt,
            answer=str(target_count),
            metadata={
                "task": "aggregation",
                "target": "TARGET_EVENT",
                "count": target_count,
                "distractor_count": distractor_count,
                "synthetic_difficulty": difficulty,
            },
        )

    def common_words(
        self,
        *,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
    ) -> TaskSample:
        maximum_common = max(2, min(6, answer_length))
        common_count = min(
            maximum_common,
            _difficulty_value(difficulty, easy=2, medium=3, hard=maximum_common),
        )
        used_tokens: set[str] = set()

        def next_token() -> str:
            token = f"token_{self.rng.randrange(1_000_000)}"
            while token in used_tokens:
                token = f"token_{self.rng.randrange(1_000_000)}"
            used_tokens.add(token)
            return token

        common = [next_token() for _ in range(common_count)]
        maximum = max(3, min(8, context_length // 512))
        list_count = min(
            maximum,
            _difficulty_value(difficulty, easy=3, medium=5, hard=8),
        )
        unique_per_list = _difficulty_value(
            difficulty,
            easy=4,
            medium=8,
            hard=max(3, context_length // 128),
        )
        lists: list[list[str]] = []
        for _ in range(list_count):
            unique = [next_token() for _ in range(unique_per_list)]
            items = unique + common
            self.rng.shuffle(items)
            lists.append(items)
        prompt = "\n".join(
            f"List {index + 1}: {', '.join(items)}" for index, items in enumerate(lists)
        ) + (
            "\nQuestion: Which exact token_* entries occur in every list? Return all and only those "
            "entries as a comma-separated set with no explanation.\nAnswer:"
        )
        return TaskSample(
            prompt=prompt,
            answer=", ".join(sorted(common)),
            metadata={
                "task": "common_words",
                "list_count": list_count,
                "common_count": common_count,
                "unique_per_list": unique_per_list,
                "synthetic_difficulty": difficulty,
            },
        )

    def build(
        self,
        task: str,
        *,
        num_samples: int,
        context_length: int,
        answer_length: int,
        difficulty: SyntheticDifficulty = "hard",
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
                self.multi_needle(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        if task == "variable_tracking":
            return [
                self.variable_tracking(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        if task == "needle_absent":
            return [
                self.needle_absent(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        if task == "multi_hop_tracing":
            return [
                self.multi_hop_tracing(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        if task == "aggregation":
            return [
                self.aggregation(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        if task == "common_words":
            return [
                self.common_words(
                    context_length=context_length,
                    answer_length=answer_length,
                    difficulty=difficulty,
                )
                for _ in range(num_samples)
            ]
        raise ValueError(f"Unsupported synthetic task: {task}")
