"""
SDKG Phase 2: build database-level LH/HL severity retrieval trees.

For each experiment setting (p, u), the script:
- compares all database cases pairwise by d_ij^p
- keeps only pairs satisfying d_ij^p <= tau^u and Delta_ij^p != 0
- builds one database-level LH tree/forest by assigning each heavier node
  its nearest lighter parent
- builds one database-level HL tree/forest by assigning each lighter node
  its nearest heavier parent

Nodes that have no qualified parent are left as roots/unattached nodes.  This
matches the current thesis direction: do not force all cases into a complete
or balanced tree when no close comparable case exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path
from typing import Iterable

import numpy as np

from sdkg_distance import (
    LEGAL_FEATURE_GROUPS,
    legal_feature_arrays,
    shared_activated_feature_mask,
    weighted_legal_feature_distance,
)


BASE_DIR = Path(__file__).resolve().parent


def existing_input_file(*parts: str) -> Path:
    local_path = BASE_DIR.joinpath(*parts)
    if not local_path.exists():
        raise FileNotFoundError(f"Required SDKG input file not found: {local_path}")
    return local_path


INPUT_FILE = existing_input_file("phase1_boolean_severity_v1.jsonl")
OUTPUT_DIR = BASE_DIR / "experiment_outputs" / "severity_trees"

WEIGHT_PERMUTATIONS = sorted(set(permutations((0.5, 0.3, 0.2), 3)), reverse=True)
DISTANCE_THRESHOLDS = [0.100, 0.250, 0.500]
FEATURE_GROUPS = ("litigants", "fact", "injury", "compensation")
WEIGHT_CODES = {
    (0.5, 0.3, 0.2): "FI",
    (0.5, 0.2, 0.3): "FC",
    (0.3, 0.5, 0.2): "IF",
    (0.3, 0.2, 0.5): "CF",
    (0.2, 0.5, 0.3): "IC",
    (0.2, 0.3, 0.5): "CI",
}
TAU_CODES = {
    0.100: "L",
    0.250: "M",
    0.500: "H",
}


def iter_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def feature_keys(records: list[dict]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        matrix = rec.get("boolean_matrix", {})
        for group in FEATURE_GROUPS:
            for name in matrix.get(group, {}):
                key = (group, name)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def build_arrays(max_cases: int | None = None) -> dict:
    records: list[dict] = []
    for idx, rec in enumerate(iter_records(INPUT_FILE)):
        if max_cases is not None and idx >= max_cases:
            break
        records.append(rec)

    keys = feature_keys(records)
    case_ids: list[str] = []
    features: list[list[float]] = []
    fact_values: list[float] = []
    injury_values: list[float] = []
    comp_values: list[float] = []

    for rec in records:
        matrix = rec.get("boolean_matrix", {})
        case_ids.append(str(rec["case_id"]))
        features.append([
            float(matrix.get(group, {}).get(name, 0.0))
            for group, name in keys
        ])
        scores = rec.get("severity_scores", {})
        fact_values.append(float(scores.get("Fact", 0.0)))
        injury_values.append(float(scores.get("Injury", 0.0)))
        comp_values.append(float(scores.get("Compensation", 0.0)))

    legal_features, legal_keys = legal_feature_arrays(records)

    return {
        "case_ids": case_ids,
        "feature_keys": keys,
        "features": np.asarray(features, dtype=np.float32),
        "legal_feature_keys": legal_keys,
        "legal_features": legal_features,
        "feature_group_indices": {
            group: np.asarray(
                [idx for idx, (key_group, _) in enumerate(keys) if key_group == group],
                dtype=np.int64,
            )
            for group in FEATURE_GROUPS
        },
        "fact_values": np.asarray(fact_values, dtype=np.float32),
        "injury_values": np.asarray(injury_values, dtype=np.float32),
        "comp_values": np.asarray(comp_values, dtype=np.float32),
        "case_sort": np.asarray([
            int(cid) if cid.isdigit() else 10**12 + idx
            for idx, cid in enumerate(case_ids)
        ], dtype=np.int64),
    }


def build_experiments() -> list[dict]:
    experiments: list[dict] = []
    exp_num = 1
    for tau in DISTANCE_THRESHOLDS:
        for fact_w, injury_w, comp_w in WEIGHT_PERMUTATIONS:
            weight_key = (float(fact_w), float(injury_w), float(comp_w))
            experiments.append({
                "exp_id": f"E{exp_num:02d}",
                "config_code": f"{WEIGHT_CODES[weight_key]}-{TAU_CODES[float(tau)]}",
                "parameter_setting": (
                    f"alpha={fact_w:.1f}, beta={injury_w:.1f}, "
                    f"1-alpha-beta={comp_w:.1f}, tau={tau:.3f}"
                ),
                "fact_w": float(fact_w),
                "injury_w": float(injury_w),
                "comp_w": float(comp_w),
                "tau": float(tau),
            })
            exp_num += 1
    return experiments


def update_best_parent(
    best_parent: np.ndarray,
    best_distance: np.ndarray,
    best_delta: np.ndarray,
    candidate_parents: np.ndarray,
    candidate_children: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_deltas: np.ndarray,
    case_sort: np.ndarray,
) -> None:
    for parent, child, distance, delta in zip(
        candidate_parents,
        candidate_children,
        candidate_distances,
        candidate_deltas,
    ):
        prev_parent = best_parent[child]
        prev_distance = best_distance[child]
        if (
            distance < prev_distance
            or (
                distance == prev_distance
                and (
                    prev_parent < 0
                    or case_sort[parent] < case_sort[prev_parent]
                )
            )
        ):
            best_parent[child] = parent
            best_distance[child] = distance
            best_delta[child] = delta


def parent_depths(best_parent: np.ndarray) -> tuple[np.ndarray, int]:
    n = len(best_parent)
    depths = np.full(n, -1, dtype=np.int32)
    cycle_count = 0

    for idx in range(n):
        if depths[idx] >= 0:
            continue
        stack: list[int] = []
        visiting: set[int] = set()
        cur = idx
        while cur >= 0 and depths[cur] < 0 and cur not in visiting:
            visiting.add(cur)
            stack.append(cur)
            cur = int(best_parent[cur])
        if cur in visiting:
            cycle_count += 1
            cycle_start = stack.index(cur)
            for node in stack[cycle_start:]:
                depths[node] = 0
            base_depth = 0
            remaining = stack[:cycle_start]
        else:
            base_depth = int(depths[cur]) if cur >= 0 else 0
            remaining = stack
        start_depth = 1 if cur >= 0 else 0
        for depth_offset, node in enumerate(reversed(remaining), start=start_depth):
            depths[node] = base_depth + depth_offset

    return depths, cycle_count


def component_stats(best_parent: np.ndarray) -> tuple[int, int]:
    n = len(best_parent)
    adj: list[list[int]] = [[] for _ in range(n)]
    for child, parent in enumerate(best_parent):
        if parent >= 0:
            p = int(parent)
            adj[p].append(child)
            adj[child].append(p)

    visited = np.zeros(n, dtype=bool)
    component_count = 0
    largest_component = 0
    for start in range(n):
        if visited[start]:
            continue
        component_count += 1
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        largest_component = max(largest_component, size)
    return component_count, largest_component


def summarize_tree(
    direction: str,
    exp: dict,
    arrays: dict,
    best_parent: np.ndarray,
    best_distance: np.ndarray,
    best_delta: np.ndarray,
    lambda_legal: float,
) -> dict:
    n = len(best_parent)
    edge_count = int(np.count_nonzero(best_parent >= 0))
    root_count = n - edge_count
    depths, cycle_count = parent_depths(best_parent)
    component_count, largest_component = component_stats(best_parent)
    parent_distances = best_distance[best_parent >= 0]
    child_counts = np.bincount(
        best_parent[best_parent >= 0].astype(np.int64),
        minlength=n,
    )

    root_idx = int(np.argmin(arrays["severity_score"])) if direction == "LH" else int(np.argmax(arrays["severity_score"]))

    return {
        "exp_id": exp["exp_id"],
        "config_code": exp["config_code"],
        "parameter_setting": exp["parameter_setting"],
        "direction": direction,
        "feature_rule": "shared_FUP_activated_feature",
        "alpha": exp["fact_w"],
        "beta": exp["injury_w"],
        "one_minus_alpha_beta": exp["comp_w"],
        "lambda_legal": float(lambda_legal),
        "lambda_severity": float(1.0 - lambda_legal),
        "tau": exp["tau"],
        "case_count": n,
        "root_case_id_by_score": arrays["case_ids"][root_idx],
        "root_score": float(arrays["severity_score"][root_idx]),
        "tree_edge_count": edge_count,
        "root_or_unattached_count": root_count,
        "coverage_rate": edge_count / max(n - 1, 1),
        "component_count": component_count,
        "largest_component": largest_component,
        "largest_component_rate": largest_component / n if n else 0.0,
        "max_depth": int(depths.max()) if n else 0,
        "avg_depth": float(np.mean(depths)) if n else 0.0,
        "median_depth": float(np.quantile(depths, 0.5)) if n else 0.0,
        "avg_parent_distance": float(np.mean(parent_distances)) if parent_distances.size else 0.0,
        "max_parent_distance": float(np.max(parent_distances)) if parent_distances.size else 0.0,
        "max_children_per_node": int(child_counts.max()) if child_counts.size else 0,
        "cycle_count": cycle_count,
    }


def build_tree_for_experiment(
    arrays: dict,
    exp: dict,
    lambda_legal: float,
    chunk_size: int,
) -> tuple[dict, dict, list[dict]]:
    legal_features = arrays["legal_features"]
    fact_values = arrays["fact_values"]
    injury_values = arrays["injury_values"]
    comp_values = arrays["comp_values"]
    case_sort = arrays["case_sort"]
    n = len(arrays["case_ids"])

    fact_w = np.float32(exp["fact_w"])
    injury_w = np.float32(exp["injury_w"])
    comp_w = np.float32(exp["comp_w"])
    tau = np.float32(exp["tau"])
    lam = np.float32(lambda_legal)

    severity_score = fact_w * fact_values + injury_w * injury_values + comp_w * comp_values
    arrays["severity_score"] = severity_score

    lh_parent = np.full(n, -1, dtype=np.int32)
    lh_distance = np.full(n, np.inf, dtype=np.float32)
    lh_delta = np.zeros(n, dtype=np.float32)
    hl_parent = np.full(n, -1, dtype=np.int32)
    hl_distance = np.full(n, np.inf, dtype=np.float32)
    hl_delta = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        parent_legal_features = {
            group: legal_features[group][start:stop]
            for group in LEGAL_FEATURE_GROUPS
        }
        parent_score = severity_score[start:stop]
        parent_fact = fact_values[start:stop]
        parent_injury = injury_values[start:stop]
        parent_comp = comp_values[start:stop]

        df = weighted_legal_feature_distance(
            parent_legal_features,
            legal_features,
            fact_w,
            injury_w,
            comp_w,
        )
        ds = (
            fact_w * np.abs(parent_fact[:, None] - fact_values[None, :])
            + injury_w * np.abs(parent_injury[:, None] - injury_values[None, :])
            + comp_w * np.abs(parent_comp[:, None] - comp_values[None, :])
        )
        dist = lam * df + (np.float32(1.0) - lam) * ds
        parent_idx = np.arange(start, stop)[:, None]
        child_idx = np.arange(n)[None, :]
        not_self = parent_idx != child_idx
        delta = severity_score[None, :] - parent_score[:, None]
        shared_feature = shared_activated_feature_mask(parent_legal_features, legal_features)

        lh_mask = shared_feature & (dist <= tau) & (delta > 0) & not_self
        rows, cols = np.nonzero(lh_mask)
        if rows.size:
            update_best_parent(
                lh_parent,
                lh_distance,
                lh_delta,
                rows + start,
                cols,
                dist[rows, cols],
                delta[rows, cols],
                case_sort,
            )

        hl_mask = shared_feature & (dist <= tau) & (delta < 0) & not_self
        rows, cols = np.nonzero(hl_mask)
        if rows.size:
            update_best_parent(
                hl_parent,
                hl_distance,
                hl_delta,
                rows + start,
                cols,
                dist[rows, cols],
                delta[rows, cols],
                case_sort,
            )

    lh_summary = summarize_tree("LH", exp, arrays, lh_parent, lh_distance, lh_delta, lambda_legal)
    hl_summary = summarize_tree("HL", exp, arrays, hl_parent, hl_distance, hl_delta, lambda_legal)

    links: list[dict] = []
    for direction, parents, distances, deltas in (
        ("LH", lh_parent, lh_distance, lh_delta),
        ("HL", hl_parent, hl_distance, hl_delta),
    ):
        depths, _ = parent_depths(parents)
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            links.append({
                "exp_id": exp["exp_id"],
                "config_code": exp["config_code"],
                "parameter_setting": exp["parameter_setting"],
                "feature_rule": "shared_FUP_activated_feature",
                "direction": direction,
                "parent_case_id": arrays["case_ids"][int(parent)],
                "child_case_id": arrays["case_ids"][child],
                "parent_score": float(severity_score[int(parent)]),
                "child_score": float(severity_score[child]),
                "delta_parent_child": float(deltas[child]),
                "distance": float(distances[child]),
                "child_depth": int(depths[child]),
            })

    return lh_summary, hl_summary, links


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--lambda-legal", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-links", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.lambda_legal <= 1.0:
        raise ValueError("--lambda-legal must be in [0, 1]")

    arrays = build_arrays(args.max_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    link_rows: list[dict] = []
    for exp in build_experiments():
        print(
            f"Building {exp['exp_id']} {exp['parameter_setting']} "
            f"lambda={args.lambda_legal:.3f}"
        )
        lh_summary, hl_summary, links = build_tree_for_experiment(
            arrays,
            exp,
            args.lambda_legal,
            args.chunk_size,
        )
        summary_rows.extend([lh_summary, hl_summary])
        if not args.skip_links:
            link_rows.extend(links)

    suffix = f"lambda_{args.lambda_legal:.2f}".replace(".", "p")
    if args.max_cases is not None:
        suffix += f"_n{args.max_cases}"

    summary_csv = args.output_dir / f"severity_tree_summary_{suffix}.csv"
    summary_json = args.output_dir / f"severity_tree_summary_{suffix}.json"
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")

    if not args.skip_links:
        links_csv = args.output_dir / f"severity_tree_parent_links_{suffix}.csv"
        write_csv(links_csv, link_rows)
        print(f"Wrote {links_csv}")


if __name__ == "__main__":
    main()
