from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


LEGAL_FEATURE_GROUPS = ("fact", "injury", "compensation")


def legal_feature_keys(records: Sequence[dict]) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {group: [] for group in LEGAL_FEATURE_GROUPS}
    seen: dict[str, set[str]] = {group: set() for group in LEGAL_FEATURE_GROUPS}
    for rec in records:
        matrix = rec.get("boolean_matrix", {})
        for group in LEGAL_FEATURE_GROUPS:
            for name in matrix.get(group, {}):
                if name not in seen[group]:
                    seen[group].add(name)
                    keys[group].append(name)
    return keys


def legal_feature_arrays(
    records: Sequence[dict],
    keys_by_group: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    keys = {group: list(values) for group, values in (keys_by_group or legal_feature_keys(records)).items()}
    arrays: dict[str, np.ndarray] = {}
    for group in LEGAL_FEATURE_GROUPS:
        rows = []
        group_keys = keys.get(group, [])
        for rec in records:
            values = rec.get("boolean_matrix", {}).get(group, {})
            rows.append([float(values.get(name, 0.0)) for name in group_keys])
        arrays[group] = np.asarray(rows, dtype=np.float32)
    return arrays, keys


def legal_feature_vector(
    boolean_matrix: Mapping[str, Mapping[str, int | float]],
    keys_by_group: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for group in LEGAL_FEATURE_GROUPS:
        values = boolean_matrix.get(group, {})
        vectors[group] = np.asarray(
            [[float(values.get(name, 0.0)) for name in keys_by_group.get(group, [])]],
            dtype=np.float32,
        )
    return vectors


def normalized_binary_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - intersection
    return np.divide(
        union - intersection,
        union,
        out=np.zeros_like(union, dtype=np.float32),
        where=union > 0,
    ).astype(np.float32)


def weighted_legal_feature_distance(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    fact_w: np.float32 | float,
    injury_w: np.float32 | float,
    comp_w: np.float32 | float,
) -> np.ndarray:
    return (
        np.float32(fact_w) * normalized_binary_distance(left["fact"], right["fact"])
        + np.float32(injury_w) * normalized_binary_distance(left["injury"], right["injury"])
        + np.float32(comp_w) * normalized_binary_distance(left["compensation"], right["compensation"])
    ).astype(np.float32)


def shared_activated_feature_mask(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> np.ndarray:
    shared = None
    for group in LEGAL_FEATURE_GROUPS:
        group_shared = (left[group] @ right[group].T) > 0
        shared = group_shared if shared is None else (shared | group_shared)
    if shared is None:
        return np.zeros((0, 0), dtype=bool)
    return shared
