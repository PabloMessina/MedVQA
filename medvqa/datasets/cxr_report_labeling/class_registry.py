"""Canonical CXR class registry with ontology metadata and prompt hashes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

from medvqa.settings import LLM_PROMPTS_DIR
from medvqa.utils.constants import (
    CANONICAL_CLASS_TO_NLI_HYPOTHESIS,
    CANONICAL_CLASS_TO_REGEX_CLASSES,
)

ONTOLOGY_PATH = os.path.join(
    os.path.dirname(__file__),
    "ontology",
    "canonical_cxr_ontology.yaml",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_system_prompt_text() -> str:
    path = os.path.join(LLM_PROMPTS_DIR, "report_nli.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


@dataclass
class ClassEntry:
    class_id: str
    display_name: str
    stem: str
    prompt_text: str
    prompt_hash: str
    regex_names: list[str] | None
    replaces: str | None = None
    replaced_by: str | None = None
    split_from: str | None = None
    merged_into: str | None = None

    @property
    def has_regex(self) -> bool:
        return bool(self.regex_names)


@dataclass
class ClassRegistry:
    ontology_version: str
    created_at: str
    system_prompt_hash: str
    system_prompt_path: str
    classes: list[ClassEntry] = field(default_factory=list)

    def by_id(self) -> dict[str, ClassEntry]:
        return {c.class_id: c for c in self.classes}

    def class_ids(self) -> list[str]:
        return [c.class_id for c in self.classes]

    def regex_backed_class_ids(self) -> list[str]:
        return [c.class_id for c in self.classes if c.has_regex]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ontology_version": self.ontology_version,
            "created_at": self.created_at,
            "system_prompt_hash": self.system_prompt_hash,
            "system_prompt_path": self.system_prompt_path,
            "classes": [asdict(c) for c in self.classes],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassRegistry:
        classes = [ClassEntry(**c) for c in data["classes"]]
        return cls(
            ontology_version=data["ontology_version"],
            created_at=data["created_at"],
            system_prompt_hash=data["system_prompt_hash"],
            system_prompt_path=data["system_prompt_path"],
            classes=classes,
        )

    @classmethod
    def load(cls, path: str) -> ClassRegistry:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_canonical_ontology() -> dict[str, Any]:
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        ontology = yaml.safe_load(f)
    classes = ontology.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"No classes found in canonical ontology: {ONTOLOGY_PATH}")
    return ontology


def resolve_regex_pattern_name(class_id: str) -> str | None:
    """Return the primary regex dict key for a canonical class, or None."""
    regs = CANONICAL_CLASS_TO_REGEX_CLASSES.get(class_id)
    if regs is None:
        return None
    assert isinstance(regs, list) and len(regs) >= 1
    return regs[0]


def build_class_registry(
    ontology_version: str | None = None,
    class_names: Sequence[str] | None = None,
) -> ClassRegistry:
    ontology = load_canonical_ontology()
    ontology_metadata = ontology["ontology"]
    ontology_classes = ontology["classes"]
    classes_by_id = {item["id"]: item for item in ontology_classes}
    if len(classes_by_id) != len(ontology_classes):
        raise ValueError(f"Duplicate class IDs in canonical ontology: {ONTOLOGY_PATH}")

    hypothesis_ids = set(CANONICAL_CLASS_TO_NLI_HYPOTHESIS)
    ontology_ids = set(classes_by_id)
    if hypothesis_ids != ontology_ids:
        raise ValueError(
            "Canonical ontology and NLI hypotheses disagree: "
            f"missing hypotheses={sorted(ontology_ids - hypothesis_ids)}, "
            f"unknown hypotheses={sorted(hypothesis_ids - ontology_ids)}"
        )

    if class_names is None:
        selected_classes = ontology_classes
    else:
        classes_by_name = {item["name"]: item for item in ontology_classes}
        selected_classes = []
        for value in class_names:
            item = classes_by_id.get(value) or classes_by_name.get(value)
            if item is None:
                raise KeyError(f"Unknown canonical CXR class: {value}")
            selected_classes.append(item)

    regex_ids = set(CANONICAL_CLASS_TO_REGEX_CLASSES)
    if regex_ids != ontology_ids:
        raise ValueError(
            "Canonical ontology and regex map disagree: "
            f"missing regex={sorted(ontology_ids - regex_ids)}, "
            f"unknown regex={sorted(regex_ids - ontology_ids)}"
        )

    system_prompt = load_system_prompt_text()
    entries: list[ClassEntry] = []
    for item in selected_classes:
        class_id = item["id"]
        prompt_text = CANONICAL_CLASS_TO_NLI_HYPOTHESIS[class_id]
        regs = CANONICAL_CLASS_TO_REGEX_CLASSES[class_id]
        regex_names = None if regs is None else list(regs)
        entries.append(
            ClassEntry(
                class_id=class_id,
                display_name=item["name"],
                stem=class_id,
                prompt_text=prompt_text,
                prompt_hash=_sha256_text(prompt_text),
                regex_names=regex_names,
            )
        )
    return ClassRegistry(
        ontology_version=ontology_version or str(ontology_metadata["version"]),
        created_at=datetime.now(timezone.utc).isoformat(),
        system_prompt_hash=_sha256_text(system_prompt),
        system_prompt_path=os.path.join(LLM_PROMPTS_DIR, "report_nli.txt"),
        classes=entries,
    )
