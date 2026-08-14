from __future__ import annotations

"""
Small API-facing wrapper for SDKG indictment generation.

The research script `SDKG_query_generate.py` is intentionally kept as the
batch-experiment entry point.  This module exposes the same core pipeline as a
stable function that a web backend can call with one user-provided case at a
time.
"""

from dataclasses import dataclass
import re
from typing import Any

try:
    from . import SDKG_query_generate as pipeline
    from . import sdkg_generation_sections as section_tools
    from .sdkg_generation_legal import extract_parties_fallback
except ImportError:
    import SDKG_query_generate as pipeline
    import sdkg_generation_sections as section_tools
    from sdkg_generation_legal import extract_parties_fallback


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
        normalized_query_text = normalize_web_query_text(query_text.strip())

        record = pipeline.run_generation_for_query(
            {
                "query_id": query_id,
                "category": category,
                "query_text": normalized_query_text,
                "original_query_text": query_text.strip(),
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

    def analyze(
        self,
        query_text: str,
        *,
        category: str = "",
    ) -> dict[str, Any]:
        return analyze_query_text(query_text, category=category)


def select_experiment(experiment: str) -> dict[str, Any]:
    normalized = experiment.strip().upper()
    for exp in pipeline.build_experiments():
        if exp["short_name"].upper() == normalized or exp["exp_id"].upper() == normalized:
            return exp
    valid = ", ".join(exp["short_name"] for exp in pipeline.build_experiments())
    raise ValueError(f"Unknown experiment '{experiment}'. Valid short names: {valid}")


def public_response(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_text": record.get("original_query_text", record["query_text"]),
        "normalized_query_text": record["query_text"],
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


def analyze_query_text(query_text: str, *, category: str = "") -> dict[str, Any]:
    if not query_text or not query_text.strip():
        raise ValueError("query_text is required")

    original_text = query_text.strip()
    normalized_query_text = normalize_web_query_text(original_text)
    raw_sections = pipeline.extract_sections(original_text)
    sections = pipeline.extract_sections(normalized_query_text)
    parties = extract_parties_fallback(normalized_query_text)
    damage_constraints = section_tools.extract_damage_constraints(
        sections["compensation_facts"],
        sections["injuries"],
    )
    damage_items = format_damage_items(
        damage_constraints["required_items"],
        context_text=f"{sections['injuries']}\n{sections['compensation_facts']}",
    )
    total_amounts = extract_overall_total_amounts(sections["compensation_facts"])
    item_total = sum(
        int(item["amount_value"])
        for item in damage_items
        if str(item.get("amount_value", "")).isdigit()
    )
    extracted = {
        "plaintiffs": improve_party_detection("原告", parties.get("原告", "原告"), normalized_query_text),
        "defendants": improve_party_detection("被告", parties.get("被告", "被告"), normalized_query_text),
        "accident_facts": sections["accident_facts"],
        "injury_facts": sections["injuries"],
        "compensation_facts": sections["compensation_facts"],
        "injuries": extract_injury_terms(sections["injuries"]),
        "treatment": sorted(damage_constraints["allowed_hospitals"]),
        "damage_items": damage_items,
        "detected_total_amounts": total_amounts,
        "calculated_item_total": f"{item_total:,}元" if item_total else "",
        "feature_profile": pipeline.build_query_boolean_matrix(normalized_query_text, category),
    }
    warnings = build_query_warnings(
        original_text=original_text,
        raw_sections=raw_sections,
        sections=sections,
        parties=parties,
        extracted=extracted,
        item_total=item_total,
        total_amounts=total_amounts,
    )
    return {
        "query_text": original_text,
        "normalized_query_text": normalized_query_text,
        "extracted": extracted,
        "warnings": warnings,
        "ready_to_generate": len(warnings) == 0 or only_soft_warnings(warnings),
    }


def format_damage_items(items: list[dict], *, context_text: str = "") -> list[dict[str, str]]:
    formatted = []
    for item in items:
        label = item.get("label", "")
        amount_value = item.get("amount_value", "")
        amount = item.get("amount_raw", "")
        source = item.get("source_line", "")
        hint_source = "\n".join(part for part in [source, context_text] if part)
        formatted.append({
            "label": label,
            "amount": f"{amount}元" if amount and not str(amount).endswith("元") else str(amount),
            "amount_value": str(amount_value),
            "source_text": source,
            "evidence_hint": extract_evidence_hint(label, hint_source),
            "basis_hint": extract_basis_hint(label, hint_source),
        })
    return formatted


def extract_evidence_hint(label: str, text: str) -> str:
    if label in {"精神慰撫金", "工作損失", "勞動能力減損"}:
        return ""
    hints = []
    for token in ["診斷證明書", "醫療費用收據", "門診收據", "收據", "單據", "估價單", "發票", "薪資證明", "扣繳憑單"]:
        if token in text and token not in hints:
            hints.append(token)
    return "、".join(hints)


def extract_basis_hint(label: str, text: str) -> str:
    if label in {"工作損失", "勞動能力減損"}:
        terms = []
        for pattern in [r"休養[0-9一二三四五六七八九十]+個月", r"不能工作[0-9一二三四五六七八九十]+個月", r"每月薪資[0-9,]+元", r"月薪[0-9,]+元", r"日薪[0-9,]+元"]:
            for match in re.finditer(pattern, text):
                if match.group(0) not in terms:
                    terms.append(match.group(0))
        return "、".join(terms)
    if label == "醫療費用":
        hospitals = sorted(section_tools.extract_hospital_like_terms(text))
        return "、".join(hospitals)
    return ""


def split_party_display(text: str) -> list[str]:
    if not text:
        return []
    values = [part.strip() for part in re.split(r"[、,，]+", text) if part.strip()]
    return values or [text]


def improve_party_detection(role: str, fallback_text: str, query_text: str) -> list[str]:
    fallback_values = split_party_display(fallback_text)
    if fallback_values and fallback_values != [role]:
        return fallback_values
    pattern = rf"{role}([\u4e00-\u9fff]{{1,2}}○○|○○|[甲乙丙丁戊己庚辛壬癸])"
    values = []
    for match in re.finditer(pattern, query_text):
        value = match.group(1)
        if value and value not in values:
            values.append(value)
    return values or fallback_values


def extract_injury_terms(text: str) -> list[str]:
    terms = []
    injury_keywords = [
        "骨折", "挫傷", "擦傷", "裂傷", "扭傷", "創傷", "損傷", "脫臼",
        "腦震盪", "疼痛", "撕裂傷", "粉碎性骨折",
    ]
    for sentence in split_sentences(text):
        if any(keyword in sentence for keyword in injury_keywords):
            cleaned = sentence.strip(" ，。；")
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
    return terms[:8]


def extract_overall_total_amounts(text: str) -> list[str]:
    totals = []
    patterns = [
        rf"(?:上開|上揭|前述|全部|各項)?(?:損害|請求|賠償)?(?:合計|共計|總計|總共|總額)[^0-9。；\n]{{0,12}}{section_tools.AMOUNT_PATTERN}",
        rf"(?:請求被告|向被告請求|應賠償|應連帶賠償)[^0-9。；\n]{{0,40}}{section_tools.AMOUNT_PATTERN}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or ""):
            amount = section_tools.normalize_amount_display(match.groups()[-1])
            if amount not in totals:
                totals.append(amount)
    return [f"{amount}元" for amount in totals]


def build_query_warnings(
    *,
    original_text: str,
    raw_sections: dict[str, str],
    sections: dict[str, str],
    parties: dict,
    extracted: dict[str, Any],
    item_total: int,
    total_amounts: list[str],
) -> list[str]:
    warnings = []
    if not all(raw_sections.values()):
        warnings.append("輸入未完整使用三段式標題，系統已自動整理；建議依「一、事故發生緣由」「二、原告受傷情形」「三、請求賠償的事實根據」補齊。")
    if extracted["plaintiffs"] == ["原告"]:
        warnings.append("未明確偵測到原告姓名；建議寫明原告姓名或以原告1、原告2表示。")
    if extracted["defendants"] == ["被告"]:
        warnings.append("未明確偵測到被告姓名；建議寫明被告姓名或以被告1、被告2表示。")
    if len(sections["accident_facts"]) < 30:
        warnings.append("事故事實偏短；建議補充事故時間、地點、車輛行向、碰撞方式及被告過失。")
    if not extracted["injuries"]:
        warnings.append("未偵測到具體傷勢；建議補充骨折、挫傷、擦傷、住院、手術或休養期間等內容。")
    if not extracted["damage_items"]:
        warnings.append("未偵測到明確賠償項目與金額；建議逐項寫明醫療費用、交通費用、工作損失、看護費用、慰撫金等金額。")
    add_damage_item_warnings(extracted["damage_items"], warnings)
    if total_amounts and item_total:
        for total in total_amounts:
            total_value = section_tools.normalize_amount_value(total.replace("元", ""))
            if total_value.isdigit() and int(total_value) != item_total:
                warnings.append(f"偵測到總額{total}，但逐項金額加總為{item_total:,}元；建議確認金額是否一致。")
    if "元" in original_text and not total_amounts:
        warnings.append("偵測到多個金額但未偵測到總請求金額；建議補上損害合計金額。")
    return warnings


def add_damage_item_warnings(items: list[dict[str, str]], warnings: list[str]) -> None:
    for item in items:
        label = item["label"]
        source = item.get("source_text", "")
        if label == "醫療費用":
            if not item.get("basis_hint"):
                warnings.append("醫療費用有金額，但未偵測到醫院或診所名稱；建議補充就醫院所。")
            if not item.get("evidence_hint"):
                warnings.append("醫療費用有金額，但未偵測到收據或診斷證明等證據描述。")
        elif label == "工作損失":
            if not item.get("basis_hint") and not re.search(r"(休養|不能工作|無法工作|月薪|日薪|薪資|收入)", source):
                warnings.append("工作損失有金額，但未偵測到不能工作期間、薪資或收入計算依據。")
        elif label == "看護費用":
            if not re.search(r"(看護|照護|照顧|住院|休養|每日|每月)", source):
                warnings.append("看護費用有金額，但未偵測到看護期間或計算依據。")
        elif label in {"車輛修復費用", "財物損失"}:
            if not item.get("evidence_hint"):
                warnings.append(f"{label}有金額，但未偵測到估價單、維修收據或其他證據描述。")


def only_soft_warnings(warnings: list[str]) -> bool:
    soft_prefixes = [
        "輸入未完整使用三段式標題",
        "偵測到多個金額但未偵測到總請求金額",
    ]
    return warnings and all(any(warning.startswith(prefix) for prefix in soft_prefixes) for warning in warnings)


def normalize_web_query_text(query_text: str) -> str:
    """Convert free-form web input into the 3-section format used internally."""
    text = normalize_spaces(query_text)
    if not text:
        return text

    sections = pipeline.extract_sections(text)
    if any(sections.values()):
        accident = sections["accident_facts"] or infer_accident_facts(text)
        injuries = sections["injuries"] or infer_injury_facts(text)
        compensation = sections["compensation_facts"] or infer_compensation_facts(text)
    else:
        accident = infer_accident_facts(text)
        injuries = infer_injury_facts(text)
        compensation = infer_compensation_facts(text)

    return (
        f"一、事故發生緣由:\n{accident}\n\n"
        f"二、原告受傷情形:\n{injuries}\n\n"
        f"三、請求賠償的事實根據:\n{compensation}"
    )


def normalize_spaces(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。；;])|\n+", text)
    return [part.strip(" ，,。；;\n") for part in parts if part.strip(" ，,。；;\n")]


def infer_accident_facts(text: str) -> str:
    candidates = []
    for sentence in split_sentences(text):
        if any(token in sentence for token in ["被告", "駕駛", "騎乘", "行經", "撞", "碰撞", "車禍", "事故", "過失"]):
            if not any(token in sentence for token in ["醫療費", "慰撫金", "工作損失", "看護費", "交通費"]):
                candidates.append(sentence)
    if candidates:
        return ensure_sentence_end("。".join(candidates[:3]))
    return ensure_sentence_end(text[:900].rstrip("，,；;、 "))


def infer_injury_facts(text: str) -> str:
    candidates = []
    for sentence in split_sentences(text):
        if any(token in sentence for token in ["受傷", "受有", "骨折", "挫傷", "擦傷", "裂傷", "診斷", "住院", "手術", "休養"]):
            candidates.append(sentence)
    if candidates:
        return ensure_sentence_end("。".join(candidates[:4]))
    return "原告因本件事故受有傷害。"


def infer_compensation_facts(text: str) -> str:
    candidates = []
    for sentence in split_sentences(text):
        if any(token in sentence for token in ["元", "費用", "慰撫金", "工作損失", "收入損失", "看護", "交通", "醫療", "修理", "賠償", "請求"]):
            candidates.append(sentence)
    if candidates:
        return ensure_sentence_end("。".join(candidates[:8]))
    return "原告請求被告賠償因本件事故所生之損害。"


def ensure_sentence_end(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in "。！？；;":
        return text
    return text + "。"


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
