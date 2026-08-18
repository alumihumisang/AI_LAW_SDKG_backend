from __future__ import annotations

import re
import unicodedata
from typing import Callable

import requests


LLM_TIMEOUT_SECONDS = 60


def extract_parties_structured(query_text: str, llm_url: str, model: str) -> dict:
    """Prefer LLM extraction, then repair names with regex references."""
    parties = extract_parties_with_llm(query_text, llm_url, model)
    return verify_and_fix_party_names(parties, query_text)


def extract_parties_with_llm(query_text: str, llm_url: str, model: str) -> dict:
    prompt = f"""請你幫我從以下車禍案件的法律文件中提取並列出所有原告和被告的真實姓名。

以下是案件內容：
{query_text}

重要提取規則：
1. 只能提取明確標示為「原告XXX」的人。
2. 絕對不能提取標示為「訴外人XXX」的人。
3. 絕對不能提取標示為「乘客」、「搭載」、「車上乘客」等非原告身份的人。
4. 完整保留姓名，不可截斷。
5. 如果文中沒有明確姓名，就直接寫「原告」或「被告」。
6. 多個姓名用逗號分隔。

輸出格式只允許這兩行：
原告:姓名1,姓名2
被告:姓名1,姓名2
"""
    default_result = {"原告": "原告", "被告": "被告", "原告數量": 1, "被告數量": 1}
    try:
        response = requests.post(
            llm_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        llm_result = response.json()["response"].strip()
        return parse_llm_parties_result(llm_result)
    except Exception:
        return extract_parties_fallback(query_text)


def parse_llm_parties_result(llm_result: str) -> dict:
    result = {"原告": "原告", "被告": "被告", "原告數量": 1, "被告數量": 1}
    invalid_responses = ["請提供", "無法提取", "沒有提供", "由於您沒有"]
    if any(invalid in llm_result for invalid in invalid_responses):
        return result

    for raw_line in llm_result.splitlines():
        line = raw_line.strip()
        if line.startswith("原告:") or line.startswith("原告："):
            plaintiff_text = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            plaintiffs = [name.strip() for name in plaintiff_text.split(",") if name.strip()]
            if plaintiffs:
                result["原告"] = "、".join(plaintiffs)
                result["原告數量"] = len(plaintiffs)
        elif line.startswith("被告:") or line.startswith("被告："):
            defendant_text = line.split(":", 1)[1].strip() if ":" in line else line.split("：", 1)[1].strip()
            defendants = [name.strip() for name in defendant_text.split(",") if name.strip()]
            invalid_defendant_values = {"原告", "被告", "原告本人", "不明", "未提及", ""}
            defendants = [name for name in defendants if name not in invalid_defendant_values]
            if defendants:
                result["被告"] = "、".join(defendants)
                result["被告數量"] = len(defendants)
    return result


def verify_and_fix_party_names(parties: dict, original_text: str) -> dict:
    reference_plaintiffs = collect_reference_names(original_text, "原告")
    reference_defendants = collect_reference_names(original_text, "被告")
    corrected = dict(parties)
    corrected["原告"] = "、".join(reference_plaintiffs) if reference_plaintiffs else "原告"
    corrected["被告"] = "、".join(reference_defendants) if reference_defendants else repair_party_names(parties.get("被告", "被告"), reference_defendants, "被告")
    corrected["原告數量"] = count_joined_names(corrected["原告"], "原告")
    corrected["被告數量"] = count_joined_names(corrected["被告"], "被告")
    return corrected


def collect_reference_names(text: str, role: str) -> list[str]:
    names = []
    for name in find_role_names(text, role):
        clean = clean_party_name(name)
        if clean and not is_invalid_party_name(clean) and clean not in names:
            names.append(clean)
    return names


def repair_party_names(joined_names: str, reference_names: list[str], default_role: str) -> str:
    if not joined_names or joined_names in {"原告", "被告"}:
        return joined_names
    repaired = []
    for name in [item.strip() for item in joined_names.replace("、", ",").split(",") if item.strip()]:
        repaired_name = repair_single_name(clean_party_name(name), reference_names)
        if is_invalid_party_name(repaired_name):
            continue
        if repaired_name and repaired_name not in repaired:
            repaired.append(repaired_name)
    for ref_name in reference_names:
        clean_ref = clean_party_name(ref_name)
        if is_invalid_party_name(clean_ref):
            continue
        normalized_ref = normalize_name_for_match(clean_ref)
        if not any(normalize_name_for_match(name) == normalized_ref for name in repaired):
            repaired.append(clean_ref)
    return "、".join(repaired) if repaired else default_role


def repair_single_name(name: str, reference_names: list[str]) -> str:
    normalized_name = normalize_name_for_match(name)
    if not (2 <= len(normalized_name) <= 5):
        return name
    surname = normalized_name[0]
    last_char = normalized_name[-1]
    for ref_name in reference_names:
        normalized_ref = normalize_name_for_match(ref_name)
        if len(normalized_ref) >= 3 and normalized_ref[0] == surname and normalized_ref[-1] == last_char:
            return ref_name
    return name


def normalize_name_for_match(name: str) -> str:
    return unicodedata.normalize("NFKC", name or "")


def clean_party_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name or "")
    name = re.sub(r"^(?:原告|被告|訴外人)", "", name)
    name = re.split(r"(?:確|因|則|受|主張|所|需|支|於|騎|駕|搭|並|同樣|另|將|明知|無|對|相|之|均|為|發|遭|出|回|當|目前|酒|驟)", name, maxsplit=1)[0]
    return name.strip(" ：:，,、。；;()（） \n")


def is_invalid_party_name(name: str) -> bool:
    if not name:
        return True
    normalized = unicodedata.normalize("NFKC", name)
    if "○" not in normalized and not (2 <= len(normalized) <= 4):
        return True
    if normalized in {"損害", "本件", "請求", "具有過失", "起訴", "機車", "見狀", "和解", "過失行"}:
        return True
    return any(token in normalized for token in [
        "痛苦", "極其", "緣由", "情形", "賠償", "事實", "內容", "車輛", "工作", "學歷",
        "兩人", "二人", "三人", "多人", "被迫", "縱未", "畢業", "文強", "行動",
        "精神上", "未提及", "具有", "本件", "人車倒地", "至少", "身心",
        "發生", "碰撞", "遭被告", "傷勢照片", "驟然", "高中肄業", "國中肄業",
        "汽車", "機車", "視線", "行進", "目前依", "當時", "回診", "臉部", "牙齒",
        "動物飼主", "為此", "出院後", "酒後", "起訴", "刑事偵審", "見狀", "和解", "過失",
    ])


def count_joined_names(joined_names: str, default_role: str) -> int:
    if not joined_names or joined_names == default_role:
        return 1
    return len([item for item in joined_names.split("、") if item.strip()])


def extract_parties_fallback(query_text: str) -> dict:
    result = {"原告": "原告", "被告": "被告", "原告數量": 1, "被告數量": 1}
    plaintiffs = find_role_names(query_text, "原告")
    defendants = find_role_names(query_text, "被告")
    if plaintiffs:
        result["原告"] = "、".join(plaintiffs)
        result["原告數量"] = len(plaintiffs)
    if defendants:
        result["被告"] = "、".join(defendants)
        result["被告數量"] = len(defendants)
    return result


def find_role_names(text: str, role: str) -> list[str]:
    boundary = r"(?=[，。；、:\s()（）]|應|因|則|受|主張|所|需|支|於|騎|駕|搭|並|同樣|另|$)"
    matches = re.findall(rf"{role}([甲乙丙丁戊己庚辛壬癸○]{{1,12}}|[\u3400-\u9fff\uf900-\ufaff]{{2,8}}){boundary}", text)
    invalid = {
        "受有", "所有", "因本", "部分", "應負", "甲車", "乙車", "受傷情形", "事故發生",
        "發生緣由", "請求賠償", "事實根據", "人車倒地", "至少", "身心", "亦", "汽車",
        "當時正", "回診時", "為此", "臉部", "牙齒", "隨身", "起訴", "刑事偵審", "機車",
        "見狀", "和解", "過失行",
    }
    seen = set()
    results = []
    for value in matches:
        clean = value.strip()
        clean = re.split(r"(?:應|因|則|受|主張|所|需|支|於|騎|駕|搭|並|同樣|另|將|明知|無|對|相|之|均)", clean, maxsplit=1)[0]
        if not clean or clean in invalid or clean in seen:
            continue
        if any(token in clean for token in ["緣由", "情形", "賠償", "事實", "內容", "車輛", "工作", "學歷", "兩人", "二人", "三人", "痛苦", "極其"]):
            continue
        seen.add(clean)
        results.append(clean)
    return results


def detect_special_relationships(text: str, parties: dict) -> dict:
    defendant_count = parties.get("被告數量", 1) or 1
    relationships = {
        "未成年": False,
        "雇傭關係": False,
        "動物損害": False,
        "多被告": defendant_count > 1,
        "多原告": parties.get("原告數量", 1) > 1,
    }

    if any(keyword in text for keyword in ["法定代理人", "監護人", "未滿十八歲", "未滿18歲"]):
        relationships["未成年"] = True
    if "未成年" in text:
        exclude_patterns = [
            r"未成年子女",
            r"[一二三四五六七八九十數]名未成年",
            r"原告.*?未成年",
            r"扶養.*?未成年",
            r"照顧.*?未成年",
        ]
        if not any(re.search(pattern, text) for pattern in exclude_patterns):
            if re.search(r"被告.*?未成年", text):
                relationships["未成年"] = True
    for age_str in re.findall(r"(\d+)\s*歲", text):
        if int(age_str) < 18:
            relationships["未成年"] = True
            break
    if any(keyword in text for keyword in ["國中生", "國小生", "高中生"]):
        relationships["未成年"] = True

    employment_patterns = [
        r"僱用人責任",
        r"雇用人責任",
        r"受僱於.*?(?:被告|即)",
        r"受雇於.*?(?:被告|即)",
        r"被告.*?僱用",
        r"被告.*?雇主",
        r"被告.*?受僱",
        r"執行.*?職務",
        r"執行.*?業務",
        r"職務上.*?行為",
        r"係在執行",
        r"公司車",
        r"被告.*?員工",
        r"被告公司.*?員工",
    ]
    relationships["雇傭關係"] = any(re.search(pattern, text) for pattern in employment_patterns)
    relationships["動物損害"] = any(keyword in text for keyword in ["狗", "貓", "犬", "動物", "寵物", "咬傷", "抓傷"])
    return relationships


def determine_applicable_laws_structured(accident_facts: str, injuries: str, comp_facts: str, parties: dict) -> list[str]:
    text = f"{accident_facts}\n{injuries}\n{comp_facts}"
    relationships = detect_special_relationships(text, parties)
    applicable_laws = ["民法第184條第1項前段"]

    if not relationships["動物損害"]:
        traffic_keywords = ["汽車", "機車", "車輛", "駕駛", "交通", "撞", "碰撞"]
        if any(keyword in accident_facts for keyword in traffic_keywords):
            applicable_laws.append("民法第191條之2")

    if injuries or any(keyword in comp_facts for keyword in ["醫療", "看護", "工作損失", "薪資", "收入", "勞動能力"]):
        applicable_laws.append("民法第193條第1項")

    if any(keyword in comp_facts for keyword in ["精神", "慰撫", "痛苦", "名譽", "人格"]):
        applicable_laws.append("民法第195條第1項前段")

    if relationships["未成年"]:
        applicable_laws.append("民法第187條第1項前段")
    elif relationships["雇傭關係"]:
        applicable_laws.append("民法第188條第1項本文")
    elif relationships["多被告"]:
        applicable_laws.append("民法第185條第1項前段")

    if relationships["動物損害"]:
        applicable_laws.append("民法第190條第1項前段")

    normalized = [normalize_article_number(law) for law in applicable_laws]
    return list(dict.fromkeys(normalized))


def normalize_article_number(article: str) -> str:
    return re.sub(r"第(\d+)-(\d+)條", r"第\1條之\2", article)


def build_strict_facts_prompt(accident_facts: str) -> str:
    return f"""你是台灣律師助理，請將律師提供的事故經過內容轉換為起訴狀格式。

核心原則：你的工作是格式轉換，不是內容改寫。

律師提供的事故經過（原文）：
───────────────────────────
{accident_facts}
───────────────────────────

唯一任務：
1. 移除原本標題。
2. 在內容開頭加上「一、緣」。
3. 其餘內容完整保留，不得改寫。

必須保留：
- 所有姓名，以及原告、被告、訴外人之身分與角色
- 所有時間、地點、道路、路口、行向與車牌
- 所有車輛代稱，例如A車、B車、C車、系爭車輛
- 被告應注意而未注意的具體過失內容
- 事故發生方式、碰撞順序與損害結果
- 原告受傷情形與財產損害
- 若原文提到刑事判決、緩起訴、未成年、僱用人、動物占有人或連帶責任，必須保留

嚴格禁止：
- 不可簡化或省略內容
- 不可重組句子
- 不可自行補新事實
- 不可拆成多段

請直接輸出轉換後的事實段落，不要附說明。
"""
