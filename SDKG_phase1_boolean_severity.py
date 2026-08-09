from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable


BASE_DIR = Path(__file__).resolve().parent


def existing_input_file(name: str) -> Path:
    local_path = BASE_DIR / name
    if not local_path.exists():
        raise FileNotFoundError(f"Required SDKG input file not found: {local_path}")
    return local_path


SRC_FILE = existing_input_file("phase1_boolean_matrix_v1.jsonl")
OUT_FILE = BASE_DIR / "phase1_boolean_severity_v1.jsonl"

FACT_SCORE = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "N": 0.0}
INJURY_SCORE = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "N": 0.0}
COMP_SCORE = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "N": 0.0}
INSURANCE_DEDUCTION_PENALTY = 0.2

A_FACT_PATTERNS = [
    r"酒[後醉]駕",
    r"吐氣酒精濃度",
    r"無照",
    r"駕照.*吊銷",
    r"駕照.*註銷",
]
B_FACT_PATTERNS = [
    r"闖紅燈",
    r"逆向",
    r"超速",
    r"肇事逃逸",
    r"兩段式左轉",
    r"未依.*燈號",
]
C_FACT_PATTERNS = [
    r"未保持安全距離",
    r"追撞",
    r"未讓直行車",
    r"轉彎未讓",
    r"未禮讓",
    r"連帶",
    r"共同侵權",
    r"法定代理人",
    r"僱用人",
]
E_FACT_PATTERNS = [
    r"閃避不及",
    r"猝不及防",
]

A_INJURY_PATTERNS = [
    r"植物人",
    r"截肢",
    r"全身癱瘓",
    r"無法脫離呼吸器",
]
B_INJURY_PATTERNS = [
    r"顱內出血",
    r"硬腦膜",
    r"脊椎骨折",
    r"脊髓",
    r"失明",
    r"癱瘓",
    r"需專人.*看護",
    r"長期看護",
    r"顱骨缺損",
]
C_INJURY_PATTERNS = [
    r"骨折",
    r"鋼釘",
    r"住院",
    r"手術",
    r"開刀",
    r"粉碎性",
    r"韌帶撕裂",
]
D_INJURY_PATTERNS = [
    r"挫傷",
    r"腦震盪",
    r"復健",
    r"拉傷",
    r"扭傷",
    r"骨裂",
]
E_INJURY_PATTERNS = [
    r"擦傷",
    r"破皮",
    r"瘀青",
    r"瘀傷",
    r"擦挫傷",
]

INSURANCE_DED_PATTERNS = [
    r"強制險",
    r"已獲賠償",
    r"已領.*保險",
    r"保險公司",
    r"扣除",
]
TOTAL_AMOUNT_PATTERNS = [
    r"(?:合計|總計|共計|請求(?:被告)?(?:給付|賠償)?|賠償(?:金額)?(?:總計)?)\s*([0-9,]+)\s*元",
]


def compile_any(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


A_FACT_RE = compile_any(A_FACT_PATTERNS)
B_FACT_RE = compile_any(B_FACT_PATTERNS)
C_FACT_RE = compile_any(C_FACT_PATTERNS)
E_FACT_RE = compile_any(E_FACT_PATTERNS)

A_INJURY_RE = compile_any(A_INJURY_PATTERNS)
B_INJURY_RE = compile_any(B_INJURY_PATTERNS)
C_INJURY_RE = compile_any(C_INJURY_PATTERNS)
D_INJURY_RE = compile_any(D_INJURY_PATTERNS)
E_INJURY_RE = compile_any(E_INJURY_PATTERNS)

INSURANCE_DED_RE = compile_any(INSURANCE_DED_PATTERNS)


def iter_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_nonempty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value
    return ""


def classify_fact(rec: dict) -> str:
    fact_flags = rec["boolean_matrix"]["fact"]
    text = first_nonempty(rec.get("LI_fact", ""), rec.get("fact_text", ""), rec.get("lawyer_input", ""))
    if fact_flags.get("gross_negligence") and A_FACT_RE.search(text):
        return "A"
    if fact_flags.get("gross_negligence"):
        return "B"
    if fact_flags.get("joint_liability") or C_FACT_RE.search(text):
        return "C"
    if fact_flags.get("negligence"):
        return "D"
    if E_FACT_RE.search(text):
        return "E"
    return "N"


def classify_injury(rec: dict) -> str:
    injury_flags = rec["boolean_matrix"]["injury"]
    text = first_nonempty(rec.get("LI_injury", ""), rec.get("fact_text", ""), rec.get("lawyer_input", ""))
    if A_INJURY_RE.search(text):
        return "A"
    if B_INJURY_RE.search(text):
        return "B"
    if C_INJURY_RE.search(text):
        return "C"
    if D_INJURY_RE.search(text):
        return "D"
    if E_INJURY_RE.search(text):
        return "E"
    if injury_flags.get("head_neck") or injury_flags.get("trunk") or injury_flags.get("extremities"):
        return "D"
    if injury_flags.get("psych_other"):
        return "E"
    return "N"


def extract_total_amount(text: str) -> int:
    amounts: list[int] = []
    for pattern in TOTAL_AMOUNT_PATTERNS:
        for match in re.finditer(pattern, text):
            raw = match.group(1).replace(",", "")
            try:
                amounts.append(int(raw))
            except ValueError:
                continue
    if amounts:
        return max(amounts)

    fallback = []
    for match in re.finditer(r"([0-9][0-9,]{3,})\s*元", text):
        raw = match.group(1).replace(",", "")
        try:
            fallback.append(int(raw))
        except ValueError:
            continue
    return max(fallback) if fallback else 0


def amount_to_comp_level(amount: int) -> str:
    if amount > 2_500_000:
        return "A"
    if amount > 1_000_000:
        return "B"
    if amount > 300_000:
        return "C"
    if amount > 0:
        return "D"
    return "N"


def classify_comp(rec: dict) -> tuple[str, int, bool]:
    text = "\n".join(
        value for value in [
            rec.get("LI_compensation", ""),
            rec.get("compensation_text", ""),
            rec.get("conclusion_text", ""),
            rec.get("lawyer_input", ""),
        ]
        if value and value.strip()
    )
    amount = extract_total_amount(text)
    has_deduction = bool(INSURANCE_DED_RE.search(text))
    level = amount_to_comp_level(amount)
    if level == "N":
        comp_flags = rec["boolean_matrix"]["compensation"]
        if any(comp_flags.get(k, 0) for k in ["medical_rehab", "lost_income", "non_pecuniary", "care_other"]):
            level = "D"
    return level, amount, has_deduction


def build_record(rec: dict) -> dict:
    fact_level = classify_fact(rec)
    injury_level = classify_injury(rec)
    comp_level, total_amount, has_deduction = classify_comp(rec)
    base_comp_score = COMP_SCORE[comp_level]
    final_comp_score = base_comp_score - INSURANCE_DEDUCTION_PENALTY if has_deduction else base_comp_score

    return {
        "case_id": str(rec["case_id"]),
        "boolean_matrix": rec["boolean_matrix"],
        "severity_levels": {
            "Fact": fact_level,
            "Injury": injury_level,
            "Compensation": comp_level,
        },
        "severity_scores": {
            "Fact": FACT_SCORE[fact_level],
            "Injury": INJURY_SCORE[injury_level],
            "Compensation": final_comp_score,
        },
        "base_comp_score": base_comp_score,
        "comp_total_amount": total_amount,
        "has_insurance_deduction": has_deduction,
        "LI_fact": rec.get("LI_fact", ""),
        "LI_injury": rec.get("LI_injury", ""),
        "LI_compensation": rec.get("LI_compensation", ""),
        "fact_text": rec.get("fact_text", ""),
        "compensation_text": rec.get("compensation_text", ""),
        "conclusion_text": rec.get("conclusion_text", ""),
    }


def main() -> None:
    rows = [build_record(rec) for rec in iter_records(SRC_FILE)]
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
