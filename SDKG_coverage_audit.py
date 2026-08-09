from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import SDKG_query_generate as qg
from sdkg_generation_sections import (
    build_plaintiff_damage_outline,
    extract_damage_constraints,
    extract_party_names,
    extract_scoped_damage_items,
    normalize_amount_value,
)


def normalize_text(text: str) -> str:
    text = text.replace("臺", "台")
    text = re.sub(r"\s+", "", text)
    return text


def amount_variants(raw: str, value: str) -> set[str]:
    value = normalize_amount_value(value)
    variants = {raw, raw.replace(",", ""), value, f"{int(value):,}" if value.isdigit() else value}
    if value.isdigit() and int(value) % 10000 == 0 and int(value) >= 10000:
        variants.add(f"{int(value) // 10000}萬")
    return {normalize_text(v + "元") if not v.endswith("元") else normalize_text(v) for v in variants if v}


def term_present(term: str, text: str) -> bool:
    term = normalize_text(term)
    text = normalize_text(text)
    return bool(term) and term in text


def compact_injury_terms(injury: str) -> list[str]:
    cleaned = injury.replace("之等傷害", "").replace("等傷害", "").replace("之傷害", "").replace("傷害", "")
    parts = [p.strip(" ，、,。；;") for p in re.split(r"[、，,；;及]", cleaned)]
    return [p for p in parts if len(p) >= 4]


def source_fact_terms(source: str) -> list[str]:
    patterns = [
        r"加護病房",
        r"住院治療",
        r"普通病房",
        r"出院",
        r"生活無法自理",
        r"居家照護",
        r"專人照顧",
        r"日常生活",
        r"需休養[^\n，。；]{0,12}",
        r"需專人照顧[^\n，。；]{0,12}",
        r"以每日[^\n，。；]{0,12}",
        r"學畢業",
        r"[國高中大專]{1,2}學畢業",
        r"從事[^\n，。；]{0,18}",
        r"沒有工作",
        r"沒有月收入",
        r"月收入[^\n，。；]{0,18}",
        r"所得給付總額[^\n，。；]{0,18}",
        r"財產給付總額[^\n，。；]{0,18}",
        r"年事已高",
        r"子女[^\n，。；]{0,18}",
        r"肇事逃逸",
        r"不聞不問",
        r"拒不負責",
        r"心靈[^\n，。；]{0,18}",
        r"精神痛苦",
    ]
    terms = []
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            term = re.sub(r"[\.。…」]+$", "", match.group(0).strip())
            if len(term) >= 3 and term not in terms:
                terms.append(term)
    return terms


def source_fact_missing_entries(query_text: str, generated: str, parties: dict, constraints: dict) -> list[str]:
    sections = qg.extract_sections(query_text)
    plaintiff_names = extract_party_names(parties.get("原告", ""))
    for name in constraints.get("plaintiff_injuries", {}):
        if name not in plaintiff_names:
            plaintiff_names.append(name)
    items = extract_scoped_damage_items(sections["compensation_facts"], plaintiff_names)
    generated_norm = normalize_text(generated)
    missing = []
    for item in items:
        if item.get("label") not in {"看護費用", "工作損失", "精神慰撫金"}:
            continue
        terms = source_fact_terms(item.get("source_span") or item.get("source_line", ""))
        missing_terms = [term for term in terms if normalize_text(term) not in generated_norm]
        if missing_terms:
            label = item.get("label", "")
            amount = item.get("amount_raw", "")
            plaintiff = item.get("plaintiff", "")
            missing.append(f"{plaintiff}:{label}:{amount}元:{'、'.join(missing_terms[:5])}")
    return missing


def audit_record(record: dict) -> dict:
    query_text = record.get("query_text", "")
    generated = record.get("generated_text") or record.get("damages_section", "")
    sections = qg.extract_sections(query_text)
    constraints = extract_damage_constraints(sections["compensation_facts"], sections["injuries"])
    plaintiff_names = extract_party_names(record.get("parties", {}).get("原告", ""))
    for name in constraints.get("plaintiff_injuries", {}):
        if name not in plaintiff_names:
            plaintiff_names.append(name)

    missing_plaintiffs = [name for name in plaintiff_names if not term_present(name, generated)]

    missing_injuries = []
    for name, injury in constraints.get("plaintiff_injuries", {}).items():
        terms = compact_injury_terms(injury)
        missing_terms = [term for term in terms if not term_present(term, generated)]
        if missing_terms:
            missing_injuries.append(f"{name}:{'、'.join(missing_terms)}")

    missing_items = []
    missing_amounts = []
    for item in constraints["required_items"]:
        label = item["label"]
        raw = item["amount_raw"]
        value = item["amount_value"]
        if not term_present(label, generated):
            missing_items.append(f"{label}:{raw}元")
        if not any(variant in normalize_text(generated) for variant in amount_variants(raw, value)):
            missing_amounts.append(f"{label}:{raw}元")

    outline = build_plaintiff_damage_outline(sections["compensation_facts"], record.get("parties", {}), constraints)
    missing_source_facts = source_fact_missing_entries(
        query_text,
        generated,
        record.get("parties", {}),
        constraints,
    )
    required_count = len(constraints["required_items"])
    missing_count = len(missing_plaintiffs) + len(missing_injuries) + len(missing_items) + len(missing_amounts)
    return {
        "query_id": record.get("query_id"),
        "exp_name": record.get("exp_name"),
        "top_k": record.get("top_k"),
        "required_items": required_count,
        "missing_total": missing_count,
        "missing_plaintiffs": "；".join(missing_plaintiffs),
        "missing_injuries": "；".join(missing_injuries),
        "missing_items": "；".join(missing_items),
        "missing_amounts": "；".join(missing_amounts),
        "missing_source_fact_count": len(missing_source_facts),
        "missing_source_facts": "；".join(missing_source_facts),
        "damage_outline": outline,
    }


def iter_records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit factual coverage for SDKG generation outputs.")
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    rows = [audit_record(record) for record in iter_records(args.jsonl)]
    fields = [
        "query_id",
        "exp_name",
        "top_k",
        "required_items",
        "missing_total",
        "missing_plaintiffs",
        "missing_injuries",
        "missing_items",
        "missing_amounts",
        "missing_source_fact_count",
        "missing_source_facts",
    ]

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields + ["damage_outline"])
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{field: row[field] for field in fields} for row in rows])


if __name__ == "__main__":
    main()
