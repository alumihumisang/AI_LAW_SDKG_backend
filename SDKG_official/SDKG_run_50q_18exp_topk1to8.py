from __future__ import annotations

import argparse
import json
from pathlib import Path

from SDKG_query_generate import (
    DEFAULT_MODEL,
    OUTPUT_DIR,
    QUERY_FILE,
    append_record_jsonl,
    load_case_corpus,
    load_queries,
    rewrite_progress_csv,
    run_generation_for_query,
    select_target_experiments,
)


DEFAULT_TOP_K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


def parse_exp_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def output_paths(output_prefix: str, exp_name: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        OUTPUT_DIR / f"{output_prefix}_{exp_name}.jsonl",
        OUTPUT_DIR / f"{output_prefix}_{exp_name}.csv",
    )


def tree_links_path(lambda_legal: float) -> Path:
    suffix = f"lambda_{lambda_legal:.2f}".replace(".", "p")
    return OUTPUT_DIR.parent / "experiment_outputs" / "severity_trees" / f"severity_tree_parent_links_{suffix}.csv"


def load_existing_records(path: Path) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (int(rec["query_id"]), int(rec["top_k"]))
            records[key] = rec
    return records


def sorted_records(records_by_key: dict[tuple[int, int], dict]) -> list[dict]:
    return [
        records_by_key[key]
        for key in sorted(records_by_key, key=lambda item: (item[1], item[0]))
    ]


def rewrite_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("", encoding="utf-8")
    for record in records:
        append_record_jsonl(path, record)


def make_error_record(query_row: dict, exp: dict, top_k: int, model: str, exc: Exception) -> dict:
    return {
        "query_id": query_row["query_id"],
        "category": query_row["category"],
        "exp_id": exp["exp_id"],
        "exp_name": exp["short_name"],
        "retrieval_mode": "dual_tree",
        "tree_exp_id": exp["exp_id"],
        "anchor_case_id": "",
        "anchor_distance": "",
        "anchor_score": "",
        "weight_code": exp["weight_code"],
        "tau_code": exp["tau_code"],
        "fact_w": exp["fact_w"],
        "injury_w": exp["injury_w"],
        "comp_w": exp["comp_w"],
        "tau": exp["distance_threshold"],
        "top_k": top_k,
        "model": model,
        "query_text": query_row["query_text"],
        "silver_reference_text": query_row["ground_truth_text"],
        "legacy_silver_reference_text": query_row["ground_truth_text"],
        "human_reference_text": query_row["ground_truth_text"],
        "ground_truth_text": query_row["ground_truth_text"],
        "gpt_baseline_text": query_row["gpt_baseline_text"],
        "facts_section": "",
        "laws_section": "",
        "damages_section": "",
        "conclusion_section": "",
        "generated_text": "",
        "run_status": "error",
        "error_message": str(exc),
    }


def select_queries(query_ids_arg: str) -> list[dict]:
    queries = load_queries(QUERY_FILE)
    query_ids = parse_int_list(query_ids_arg)
    if not query_ids:
        return queries

    query_by_id = {row["query_id"]: row for row in queries}
    missing = [qid for qid in query_ids if qid not in query_by_id]
    if missing:
        raise ValueError(f"query_ids {missing} not found in {QUERY_FILE}")
    return [query_by_id[qid] for qid in query_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SDKG official 50-query x 18-experiment x top-k generation with resumable per-experiment outputs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--query-ids", default="", help="Optional comma-separated query_ids. Empty means all 50 queries.")
    parser.add_argument("--exp-names", default="", help="Optional comma-separated names such as FI-L,FC-L. Empty means all 18 experiments.")
    parser.add_argument("--top-k-values", default=",".join(str(k) for k in DEFAULT_TOP_K_VALUES))
    parser.add_argument("--lambda-legal", type=float, default=0.5, help="Tree-link lambda setting to load, e.g. 0.2, 0.5, 0.7.")
    args = parser.parse_args()

    top_k_values = parse_int_list(args.top_k_values)
    if not top_k_values:
        raise ValueError("--top-k-values cannot be empty")

    qg_links_path = tree_links_path(args.lambda_legal)
    if not qg_links_path.exists():
        raise FileNotFoundError(f"Tree links not found: {qg_links_path}")
    import SDKG_query_generate as qg
    qg.SEVERITY_TREE_LINKS_FILE = qg_links_path
    qg.ACTIVE_LAMBDA_LEGAL = float(args.lambda_legal)
    qg.EXPERIMENT_TREE_CACHE.clear()
    print(f"Using tree links: {qg_links_path}", flush=True)

    target_queries = select_queries(args.query_ids)
    target_experiments = select_target_experiments(None)
    requested_exp_names = set(parse_exp_names(args.exp_names))
    if requested_exp_names:
        target_experiments = [exp for exp in target_experiments if exp["short_name"] in requested_exp_names]
        missing = sorted(requested_exp_names - {exp["short_name"] for exp in target_experiments})
        if missing:
            raise ValueError(f"Unknown experiment names: {missing}")

    corpus_rows, litigant_values, legal_features, legal_keys, fact_values, injury_values, comp_values, case_sort = load_case_corpus()
    corpus_by_id = {str(rec["case_id"]): rec for rec in corpus_rows}
    case_idx_by_id = {str(rec["case_id"]): idx for idx, rec in enumerate(corpus_rows)}

    total_jobs = len(target_experiments) * len(top_k_values) * len(target_queries)
    job_idx = 0
    print(
        f"SDKG run started: experiments={len(target_experiments)} "
        f"queries={len(target_queries)} top_k_values={top_k_values} total_jobs={total_jobs}",
        flush=True,
    )

    for exp in target_experiments:
        jsonl_path, csv_path = output_paths(args.output_prefix, exp["short_name"])
        records_by_key = load_existing_records(jsonl_path)
        rewrite_jsonl(jsonl_path, sorted_records(records_by_key))
        rewrite_progress_csv(csv_path, sorted_records(records_by_key))

        exp_expected = len(top_k_values) * len(target_queries)
        exp_ok = sum(1 for rec in records_by_key.values() if rec.get("run_status") == "ok")
        print(
            f"[exp {exp['short_name']}] output={jsonl_path.name} "
            f"existing_ok={exp_ok}/{exp_expected}",
            flush=True,
        )

        for top_k in top_k_values:
            for query_row in target_queries:
                job_idx += 1
                key = (int(query_row["query_id"]), int(top_k))
                existing = records_by_key.get(key)
                if existing and existing.get("run_status") == "ok":
                    print(
                        f"[{job_idx}/{total_jobs}] skip query_id={query_row['query_id']} "
                        f"exp={exp['short_name']} top_k={top_k}",
                        flush=True,
                    )
                    continue

                print(
                    f"[{job_idx}/{total_jobs}] running query_id={query_row['query_id']} "
                    f"exp={exp['short_name']} top_k={top_k} model={args.model}",
                    flush=True,
                )
                try:
                    record = run_generation_for_query(
                        query_row,
                        exp,
                        corpus_rows,
                        litigant_values,
                        legal_features,
                        legal_keys,
                        fact_values,
                        injury_values,
                        comp_values,
                        case_sort,
                        corpus_by_id,
                        case_idx_by_id,
                        top_k,
                        args.model,
                    )
                    record["run_status"] = "ok"
                    record["error_message"] = ""
                    print(
                        f"[{job_idx}/{total_jobs}] done query_id={query_row['query_id']} "
                        f"exp={exp['short_name']} top_k={top_k}",
                        flush=True,
                    )
                except Exception as exc:
                    record = make_error_record(query_row, exp, top_k, args.model, exc)
                    print(
                        f"[{job_idx}/{total_jobs}] failed query_id={query_row['query_id']} "
                        f"exp={exp['short_name']} top_k={top_k} error={exc}",
                        flush=True,
                    )

                records_by_key[key] = record
                current_records = sorted_records(records_by_key)
                rewrite_jsonl(jsonl_path, current_records)
                rewrite_progress_csv(csv_path, current_records)

        exp_records = sorted_records(records_by_key)
        exp_ok = sum(1 for rec in exp_records if rec.get("run_status") == "ok")
        exp_err = sum(1 for rec in exp_records if rec.get("run_status") == "error")
        print(
            f"[exp {exp['short_name']}] saved_jsonl={jsonl_path} saved_csv={csv_path} "
            f"rows={len(exp_records)}/{exp_expected} ok={exp_ok} error={exp_err}",
            flush=True,
        )

    print("SDKG run finished.", flush=True)


if __name__ == "__main__":
    main()
