"""Parent/child DAG helpers over canonical_cxr_ontology.yaml."""

from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import yaml

ONTOLOGY_PATH = os.path.join(
    os.path.dirname(__file__),
    "ontology",
    "canonical_cxr_ontology.yaml",
)


@dataclass(frozen=True)
class OntologyDag:
    """Directed edges child → parent (label inheritance / is-a)."""

    parents: Mapping[str, List[str]]
    children: Mapping[str, List[str]]
    class_ids: frozenset

    def ancestors(self, class_id: str) -> Set[str]:
        if class_id not in self.class_ids:
            raise KeyError(class_id)
        out: Set[str] = set()
        q = deque(self.parents.get(class_id, []))
        while q:
            p = q.popleft()
            if p in out:
                continue
            out.add(p)
            q.extend(self.parents.get(p, []))
        return out

    def descendants(self, class_id: str) -> Set[str]:
        if class_id not in self.class_ids:
            raise KeyError(class_id)
        out: Set[str] = set()
        q = deque(self.children.get(class_id, []))
        while q:
            c = q.popleft()
            if c in out:
                continue
            out.add(c)
            q.extend(self.children.get(c, []))
        return out

    def induce_subgraph(self, class_ids: Sequence[str]) -> "OntologyDag":
        """Induce subgraph on ``class_ids``; require the set equals the node set.

        Edges kept only when both endpoints are in ``class_ids``. Raises if any
        listed id is unknown or if the requested set is empty.
        """
        requested = list(dict.fromkeys(class_ids))
        if not requested:
            raise ValueError("class_ids must be non-empty")
        unknown = [c for c in requested if c not in self.class_ids]
        if unknown:
            raise KeyError(f"Unknown class_ids: {unknown}")
        node_set = frozenset(requested)
        parents: Dict[str, List[str]] = {}
        children: Dict[str, List[str]] = defaultdict(list)
        for cid in requested:
            ps = [p for p in self.parents.get(cid, []) if p in node_set]
            parents[cid] = ps
            for p in ps:
                children[p].append(cid)
        for cid in requested:
            children.setdefault(cid, [])
            # stable unique
            children[cid] = list(dict.fromkeys(children[cid]))
        return OntologyDag(
            parents=parents,
            children=dict(children),
            class_ids=node_set,
        )

    def as_parent_child_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """Per-node parents and children (within this DAG / subgraph)."""
        return {
            cid: {
                "parents": list(self.parents.get(cid, [])),
                "children": list(self.children.get(cid, [])),
            }
            for cid in sorted(self.class_ids)
        }


def load_ontology_dag(path: Optional[str] = None) -> OntologyDag:
    path = path or ONTOLOGY_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes") or []
    if not classes:
        raise ValueError(f"No classes in ontology: {path}")
    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    ids: List[str] = []
    for c in classes:
        cid = c["id"]
        ids.append(cid)
        ps = list(c.get("parents") or [])
        parents[cid] = ps
        for p in ps:
            children[p].append(cid)
    id_set = frozenset(ids)
    if len(id_set) != len(ids):
        raise ValueError(f"Duplicate class IDs in ontology: {path}")
    for p in list(children):
        if p not in id_set:
            raise ValueError(f"Parent {p!r} not in ontology class set")
        children[p] = list(dict.fromkeys(children[p]))
    for cid in ids:
        children.setdefault(cid, [])
    return OntologyDag(parents=parents, children=dict(children), class_ids=id_set)


def propagate_positive_labels(
    *,
    dag: OntologyDag,
    class_ids: Iterable[str],
    positive_by_class: Mapping[str, bool],
) -> Dict[str, bool]:
    """If any descendant is positive, mark ancestors positive (within class_ids)."""
    ids = list(class_ids)
    id_set = set(ids)
    out = {cid: bool(positive_by_class.get(cid, False)) for cid in ids}
    for cid in ids:
        if not out[cid]:
            continue
        for anc in dag.ancestors(cid):
            if anc in id_set:
                out[anc] = True
    return out
