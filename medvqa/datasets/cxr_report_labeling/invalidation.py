"""Diff class registries to decide what must be recomputed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry


@dataclass
class InvalidationReport:
    re_annotate_class_ids: List[str] = field(default_factory=list)
    rematch_regex_class_ids: List[str] = field(default_factory=list)
    top_up_sample_class_ids: List[str] = field(default_factory=list)
    safe_to_keep_class_ids: List[str] = field(default_factory=list)
    dropped_class_ids: List[str] = field(default_factory=list)
    renamed: Dict[str, str] = field(default_factory=dict)  # old -> new
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"re_annotate ({len(self.re_annotate_class_ids)}): {self.re_annotate_class_ids}",
            f"rematch_regex ({len(self.rematch_regex_class_ids)}): {self.rematch_regex_class_ids}",
            f"top_up_sample ({len(self.top_up_sample_class_ids)}): {self.top_up_sample_class_ids}",
            f"safe_to_keep ({len(self.safe_to_keep_class_ids)}): {self.safe_to_keep_class_ids}",
            f"dropped ({len(self.dropped_class_ids)}): {self.dropped_class_ids}",
            f"renamed: {self.renamed}",
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def diff_registries(old: ClassRegistry, new: ClassRegistry) -> InvalidationReport:
    old_by = old.by_id()
    new_by = new.by_id()
    old_ids = set(old_by)
    new_ids = set(new_by)

    report = InvalidationReport()
    report.dropped_class_ids = sorted(old_ids - new_ids)
    added = sorted(new_ids - old_ids)

    # Explicit rename links via replaced_by / replaces
    for oid, entry in old_by.items():
        if entry.replaced_by and entry.replaced_by in new_by:
            report.renamed[oid] = entry.replaced_by

    for nid in added:
        entry = new_by[nid]
        if entry.replaces and entry.replaces in old_by:
            report.renamed[entry.replaces] = nid
            # Prompt may still differ
            old_e = old_by[entry.replaces]
            if old_e.prompt_hash != entry.prompt_hash or old.system_prompt_hash != new.system_prompt_hash:
                report.re_annotate_class_ids.append(nid)
            report.rematch_regex_class_ids.append(nid)
            report.top_up_sample_class_ids.append(nid)
        elif entry.split_from:
            report.re_annotate_class_ids.append(nid)
            report.rematch_regex_class_ids.append(nid)
            report.top_up_sample_class_ids.append(nid)
            report.notes.append(f"{nid} split from {entry.split_from}")
        else:
            report.re_annotate_class_ids.append(nid)
            report.rematch_regex_class_ids.append(nid)
            report.top_up_sample_class_ids.append(nid)

    shared = old_ids & new_ids
    system_changed = old.system_prompt_hash != new.system_prompt_hash
    if system_changed:
        report.notes.append("system prompt hash changed → all shared classes need re-annotation")

    for cid in sorted(shared):
        o, n = old_by[cid], new_by[cid]
        prompt_changed = o.prompt_hash != n.prompt_hash or system_changed
        regex_changed = o.regex_names != n.regex_names
        if prompt_changed:
            report.re_annotate_class_ids.append(cid)
        if regex_changed:
            report.rematch_regex_class_ids.append(cid)
        if not prompt_changed and not regex_changed:
            report.safe_to_keep_class_ids.append(cid)

    # Deduplicate lists
    def _uniq(xs: List[str]) -> List[str]:
        seen: Set[str] = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    report.re_annotate_class_ids = _uniq(report.re_annotate_class_ids)
    report.rematch_regex_class_ids = _uniq(report.rematch_regex_class_ids)
    report.top_up_sample_class_ids = _uniq(report.top_up_sample_class_ids)
    return report
