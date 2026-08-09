from __future__ import annotations

"""
Small API-facing wrapper for SDKG indictment generation.

The research script `SDKG_query_generate.py` is intentionally kept as the
batch-experiment entry point.  This module exposes the same core pipeline as a
stable function that a web backend can call with one user-provided case at a
time.
"""

from dataclasses import dataclass
from typing import Any

try:
    from . import SDKG_query_generate as pipeline
except ImportError:
    import SDKG_query_generate as pipeline


DEFAULT_EXPERIMENT = "FC-H"
DEFAULT_TOP_K = 8


@dataclass(frozen=True)
class GenerationConfig:
    experiment: str = DEFAULT_EXPERIMENT
    top_k: int = DEFAULT_TOP_K
    model: str = pipeline.DEFAULT_MODEL
    case_limit: int | None = None
    llm_url: str = pipeline.LLM_URL


class IndictmentGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self._loaded = False
        self._exp: dict[str, Any] | None = None
        self._corpus_rows: list[dict[str, Any]] = []
        self._litigant_values = None
        self._fact_values = None
        self._injury_values = None
        self._comp_values = None
        self._case_sort = None
        self._corpus_by_id: dict[str, dict[str, Any]] = {}
        self._case_idx_by_id: dict[str, int] = {}

    def load(self) -> None:
        if self._loaded:
            return

        pipeline.LLM_URL = self.config.llm_url
        self._exp = select_experiment(self.config.experiment)
        (
            self._corpus_rows,
            self._litigant_values,
            self._legal_features,
            self._legal_keys,
            self._fact_values,
            self._injury_values,
            self._comp_values,
            self._case_sort,
        ) = pipeline.load_case_corpus(self.config.case_limit)
        self._corpus_by_id = {
            str(record["case_id"]): record
            for record in self._corpus_rows
        }
        self._case_idx_by_id = {
            str(record["case_id"]): idx
            for idx, record in enumerate(self._corpus_rows)
        }
        self._loaded = True

    def generate(
        self,
        query_text: str,
        *,
        category: str = "",
        query_id: int | str = "web",
        top_k: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")

        self.load()
        assert self._exp is not None

        record = pipeline.run_generation_for_query(
            {
                "query_id": query_id,
                "category": category,
                "query_text": query_text.strip(),
                "reference_text": "",
                "ground_truth_text": "",
                "gpt_baseline_text": "",
            },
            self._exp,
            self._corpus_rows,
            self._litigant_values,
            self._legal_features,
            self._legal_keys,
            self._fact_values,
            self._injury_values,
            self._comp_values,
            self._case_sort,
            self._corpus_by_id,
            self._case_idx_by_id,
            top_k or self.config.top_k,
            model or self.config.model,
        )
        return public_response(record)


def select_experiment(experiment: str) -> dict[str, Any]:
    normalized = experiment.strip().upper()
    for exp in pipeline.build_experiments():
        if exp["short_name"].upper() == normalized or exp["exp_id"].upper() == normalized:
            return exp
    valid = ", ".join(exp["short_name"] for exp in pipeline.build_experiments())
    raise ValueError(f"Unknown experiment '{experiment}'. Valid short names: {valid}")


def public_response(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_text": record["generated_text"],
        "sections": {
            "facts": record["facts_section"],
            "laws": record["laws_section"],
            "damages": record["damages_section"],
            "conclusion": record["conclusion_section"],
        },
        "parties": record["parties"],
        "retrieval": {
            "mode": record["retrieval_mode"],
            "experiment": record["exp_name"],
            "top_k": record["top_k"],
            "anchor_case_id": record["anchor_case_id"],
            "anchor_distance": record["anchor_distance"],
            "similar_cases": [
                {
                    "rank": case["rank"],
                    "case_id": case["case_id"],
                    "distance": case["distance"],
                    "case_score": case["case_score"],
                    "retrieval_side": case.get("retrieval_side", ""),
                }
                for case in record["similar_cases"]
            ],
        },
        "model": record["model"],
    }


def generate_indictment(
    query_text: str,
    *,
    category: str = "",
    experiment: str = DEFAULT_EXPERIMENT,
    top_k: int = DEFAULT_TOP_K,
    model: str = pipeline.DEFAULT_MODEL,
    case_limit: int | None = None,
    llm_url: str = pipeline.LLM_URL,
) -> dict[str, Any]:
    generator = IndictmentGenerator(
        GenerationConfig(
            experiment=experiment,
            top_k=top_k,
            model=model,
            case_limit=case_limit,
            llm_url=llm_url,
        )
    )
    return generator.generate(query_text, category=category)
