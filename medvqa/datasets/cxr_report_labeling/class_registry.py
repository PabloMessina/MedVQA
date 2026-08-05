"""Canonical 55-class registry with prompt hashes and regex name aliases."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from medvqa.settings import LLM_PROMPTS_DIR
from medvqa.utils.constants import (
    UNIFIED_CXRLT2024_VINDRCXR_CLASSES,
    UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_REGEX_CLASSES,
    UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_VERBOSE_PHRASE_FOR_REPORT_NLI,
    _cxr_class_name_to_phrase_stem,
)

# Prompt-facing name -> name in _CLASS_NAME_TO_REGEX_PATTERNS
REGEX_NAME_ALIASES: Dict[str, str] = {
    "Enlarged Pulmonary Artery": "Enlarged PA",
}

ONTOLOGY_VERSION = "v1"


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
    regex_names: Optional[List[str]]  # None for Normal
    replaces: Optional[str] = None
    replaced_by: Optional[str] = None
    split_from: Optional[str] = None
    merged_into: Optional[str] = None

    @property
    def has_regex(self) -> bool:
        return bool(self.regex_names)


@dataclass
class ClassRegistry:
    ontology_version: str
    created_at: str
    system_prompt_hash: str
    system_prompt_path: str
    classes: List[ClassEntry] = field(default_factory=list)

    def by_id(self) -> Dict[str, ClassEntry]:
        return {c.class_id: c for c in self.classes}

    def class_ids(self) -> List[str]:
        return [c.class_id for c in self.classes]

    def regex_backed_class_ids(self) -> List[str]:
        return [c.class_id for c in self.classes if c.has_regex]

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "ClassRegistry":
        classes = [ClassEntry(**c) for c in data["classes"]]
        return cls(
            ontology_version=data["ontology_version"],
            created_at=data["created_at"],
            system_prompt_hash=data["system_prompt_hash"],
            system_prompt_path=data["system_prompt_path"],
            classes=classes,
        )

    @classmethod
    def load(cls, path: str) -> "ClassRegistry":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def resolve_regex_pattern_name(class_name: str) -> Optional[str]:
    """Map a prompt class to the primary regex pattern dict key (or None)."""
    regs = UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_REGEX_CLASSES.get(class_name)
    if regs is None:
        return None
    assert isinstance(regs, list) and len(regs) >= 1
    name = regs[0]
    return REGEX_NAME_ALIASES.get(name, name)


def build_class_registry(
    ontology_version: str = ONTOLOGY_VERSION,
    class_names: Optional[Sequence[str]] = None,
) -> ClassRegistry:
    system_prompt = load_system_prompt_text()
    names = list(class_names) if class_names is not None else list(UNIFIED_CXRLT2024_VINDRCXR_CLASSES)
    entries: List[ClassEntry] = []
    for name in names:
        stem = _cxr_class_name_to_phrase_stem(name)
        prompt_text = UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_VERBOSE_PHRASE_FOR_REPORT_NLI[name]
        regs = UNIFIED_CXRLT2024_VINDRCXR_CLASS_TO_REGEX_CLASSES[name]
        if regs is None:
            regex_names = None
        else:
            regex_names = [REGEX_NAME_ALIASES.get(r, r) for r in regs]
        entries.append(
            ClassEntry(
                class_id=name,
                display_name=name,
                stem=stem,
                prompt_text=prompt_text,
                prompt_hash=_sha256_text(prompt_text),
                regex_names=regex_names,
            )
        )
    return ClassRegistry(
        ontology_version=ontology_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        system_prompt_hash=_sha256_text(system_prompt),
        system_prompt_path=os.path.join(LLM_PROMPTS_DIR, "report_nli.txt"),
        classes=entries,
    )
