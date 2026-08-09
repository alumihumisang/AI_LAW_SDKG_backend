from __future__ import annotations

import re
import unicodedata


FORBIDDEN_DAMAGE_LINES = [
    "綜上所述",
    "綜上所陳",
    "按年息5%",
    "繕本送達翌日起",
]

AMOUNT_PATTERN = r"([0-9][0-9,]*(?:萬[0-9,]*)?)\s*元"
NAME_CHARS = r"\u3400-\u9fff\uf900-\ufaff○"
BAD_NAME_TOKENS = [
    "原告", "被告", "本件", "系爭", "事故", "車禍", "碰撞", "因此", "因而",
    "故使", "導致", "致使", "此次", "本次", "受有", "均受", "傷害",
    "兩人", "二人", "三人", "多人", "痛苦", "極其",
    "損害", "被迫", "縱未", "畢業", "文強", "行動", "精神上", "本件",
    "隨身", "牙齒", "臉部", "損傷", "挫傷", "骨折", "擦傷", "扭傷",
    "哺乳", "住院", "右手", "左手", "年事", "實際", "和解", "閃避",
    "仍須", "表示", "生活起居", "達成和解",
]
PLACEHOLDER_NAMES = {"甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"}


def extract_damage_constraints(comp_facts: str, injuries: str) -> dict:
    source_text = f"{injuries}\n{comp_facts}"
    required_items = extract_required_damage_items(comp_facts)
    allowed_amounts = extract_currency_amounts(source_text)
    allowed_amounts.update(aggregate_required_item_amounts(required_items))
    return {
        "allowed_amounts": allowed_amounts,
        "allowed_hospitals": extract_hospital_like_terms(source_text),
        "allowed_income_terms": extract_income_terms(source_text),
        "allowed_personal_context": extract_personal_context_terms(source_text),
        "allowed_damage_labels": infer_damage_labels(source_text),
        "required_items": required_items,
        "plaintiff_injuries": extract_plaintiff_injury_map(injuries),
    }


def aggregate_required_item_amounts(required_items: list[dict]) -> set[str]:
    totals: dict[str, int] = {}
    for item in required_items:
        label = item.get("label", "")
        amount = safe_parse_amount(item.get("amount_value", item.get("amount_raw", ""))) or 0
        if amount <= 0:
            continue
        totals[label] = totals.get(label, 0) + amount
    allowed = set()
    for total in totals.values():
        if total <= 0:
            continue
        allowed.add(str(total))
        allowed.add(f"{total:,}")
    return allowed


def extract_currency_amounts(text: str) -> set[str]:
    amounts = set()
    for match in re.finditer(AMOUNT_PATTERN, text):
        raw = match.group(1)
        amounts.add(normalize_amount_value(raw))
        amounts.add(raw.replace(",", ""))
    return amounts


def extract_hospital_like_terms(text: str) -> set[str]:
    terms = set()
    pattern = r"[\u4e00-\u9fffA-Za-z0-9○]{2,30}(?:醫院|診所|中醫|紀念醫院)"
    for match in re.finditer(pattern, text):
        terms.add(match.group(0).strip())
    return terms


def extract_income_terms(text: str) -> set[str]:
    terms = set()
    patterns = [
        r"日薪[^\n，。；]{0,20}",
        r"月薪[^\n，。；]{0,20}",
        r"年收入[^\n，。；]{0,20}",
        r"所得[^\n，。；]{0,20}",
        r"從事[^\n，。；]{0,20}",
        r"任職[^\n，。；]{0,20}",
        r"工作[^\n，。；]{0,20}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(0).strip())
    return terms


def extract_personal_context_terms(text: str) -> set[str]:
    terms = set()
    patterns = [
        r"年事已高",
        r"小學畢業",
        r"高職畢業",
        r"大學畢業",
        r"專科畢業",
        r"高中畢業",
        r"國中畢業",
        r"學歷[^\n，。；]{0,30}",
        r"從事[^\n，。；]{0,30}",
        r"任職[^\n，。；]{0,30}",
        r"職業為[^\n，。；]{0,30}",
        r"日薪[^\n，。；]{0,30}",
        r"月薪[^\n，。；]{0,30}",
        r"所得[^\n，。；]{0,30}",
        r"名下[^\n，。；]{0,30}",
        r"子女[^\n，。；]{0,30}",
        r"未成年子女[^\n，。；]{0,30}",
        r"配偶[^\n，。；]{0,30}",
        r"家庭[^\n，。；]{0,30}",
        r"加護病房",
        r"刑事偵審",
        r"民事訴訟",
        r"肇事逃逸",
        r"不聞不問",
        r"拒不負責",
        r"心靈[^\n，。；]{0,30}",
        r"精神[^\n，。；]{0,30}",
        r"痛苦[^\n，。；]{0,30}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            terms.add(match.group(0).strip())
    return terms


def extract_plaintiff_injury_map(injuries: str) -> dict[str, str]:
    injuries = unicodedata.normalize("NFC", injuries or "")
    injury_map = {}
    generic_match = re.search(
        r"原告[^，。；\n]{0,24}(?:受有|受了很嚴重的傷[，,]?包括[:：]?|受了)(.+?)(?:等傷害|等之傷害|之傷害|傷害|。)",
        injuries,
    )
    if generic_match:
        injury_text = generic_match.group(1).strip(" ：:，,。；;之")
        if injury_text:
            injury_map["原告"] = f"{injury_text}等傷害"

    group_pattern = (
        rf"((?:原告)?[{NAME_CHARS}]{{2,5}}(?:[、,，]|及|與|和)(?:原告)?[{NAME_CHARS}]{{2,5}}"
        rf"(?:[、,，及與和](?:原告)?[{NAME_CHARS}]{{2,5}})*)"
        r"[^，。；\n]{0,40}[，,]?\s*均?受有(.+?)(?:等傷害|等之傷害|之傷害|傷害|。|；|\n)"
    )
    for match in re.finditer(group_pattern, injuries):
        injury_text = match.group(2).strip(" ，,。；;之")
        if not injury_text:
            continue
        for name in split_plaintiff_names(match.group(1)):
            if name:
                injury_map[name] = f"{injury_text}等傷害"

    pattern = rf"(?:導致|致)?(?:原告)?([{NAME_CHARS}]{{1,5}})(?:因此|因而|則)?(?:因[^，。；\n]{{0,30}})?[，,]?\s*受有(.+?)(?:等傷害|等之傷害|之傷害|傷害|。|；|\n)"
    for match in re.finditer(pattern, injuries):
        name = normalize_plaintiff_name(match.group(1))
        if not name:
            continue
        injury_text = match.group(2).strip(" ，,。；;之")
        if name and injury_text:
            injury_map[name] = f"{injury_text}等傷害"
    return injury_map


def normalize_plaintiff_name(name: str) -> str:
    name = unicodedata.normalize("NFC", name or "")
    name = re.sub(r"^(?:原告|被告|訴外人)", "", name or "")
    name = re.split(r"(?:因|受|均|則)", name, maxsplit=1)[0]
    name = re.sub(r"(?:均|因此|因而|則|之|部分)$", "", name)
    name = name.strip(" ：:，,、。；; \n")
    if name in PLACEHOLDER_NAMES:
        return name
    if "○" not in name and len(name) > 4:
        return ""
    if not (2 <= len(name) <= 5):
        return ""
    if any(token in name for token in BAD_NAME_TOKENS):
        return ""
    return name


def is_specific_plaintiff_name(name: str) -> bool:
    return normalize_plaintiff_name(name) == name


def split_plaintiff_names(raw: str) -> list[str]:
    raw = unicodedata.normalize("NFC", raw or "")
    normalized = re.sub(r"(?:及|與)原告", "、原告", raw or "")
    names = []
    for part in re.split(r"[、,，\s]+|及|與", normalized):
        name = normalize_plaintiff_name(part)
        if name and name not in names:
            names.append(name)
    return names


def extract_required_damage_items(comp_facts: str) -> list[dict]:
    comp_facts = unicodedata.normalize("NFC", comp_facts or "")
    items = []
    seen: set[tuple] = set()
    for match in re.finditer(r"^\s*\d+\.\s*([^\n]+)", comp_facts, re.M):
        line = match.group(1).strip()
        if is_personal_context_only_line(line):
            continue
        amount_match = first_claim_amount_match(line)
        if not amount_match:
            continue
        amount_raw = amount_match.group(1)
        amount_value = normalize_amount_value(amount_raw)
        amount_display = normalize_amount_display(amount_raw)
        label = infer_damage_label_for_amount(line, amount_match)
        if not label:
            label = line.replace(amount_match.group(0), "").strip(" ：:，,。")
        if label == "一般損害項目":
            continue
        key = (label, amount_value)
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "label": label,
            "amount_raw": amount_display,
            "amount_value": amount_value,
            "source_line": line,
        })
    for sentence in split_damage_source_sentences(comp_facts):
        if is_personal_context_only_line(sentence):
            continue
        for amount_match in re.finditer(AMOUNT_PATTERN, sentence):
            if is_rate_or_reference_amount(sentence, amount_match):
                continue
            amount_raw = amount_match.group(1)
            amount_value = normalize_amount_value(amount_raw)
            amount_display = normalize_amount_display(amount_raw)
            label = infer_damage_label_for_amount(sentence, amount_match)
            if label == "一般損害項目":
                continue
            amount_context = extract_amount_context(sentence, amount_match)
            key = (label, amount_value, normalize_required_source_key(amount_context))
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "label": label,
                "amount_raw": amount_display,
                "amount_value": amount_value,
                "source_line": amount_context or sentence,
            })
    return items


def normalize_required_source_key(text: str) -> str:
    text = re.sub(r"^\s*\d+\.\s*", "", text.strip())
    text = re.sub(r"\s+", "", text)
    return text.rstrip("。；;")


def first_claim_amount_match(sentence: str) -> re.Match[str] | None:
    for amount_match in re.finditer(AMOUNT_PATTERN, sentence):
        if not is_rate_or_reference_amount(sentence, amount_match):
            return amount_match
    return None


def is_personal_context_only_line(line: str) -> bool:
    personal_tokens = [
        "月薪", "日薪", "月收入", "年收入", "所得", "財產", "名下", "不動產",
        "畢業", "學歷", "任職", "上班", "從事",
    ]
    claim_tokens = [
        "醫藥", "醫療", "診療", "復健", "交通費", "車資", "看護", "照護", "幫傭",
        "工作損失", "不能工作之損失", "無法工作", "收入損失", "薪資損失",
        "勞動能力損失", "減少勞動能力", "共損失", "損失約", "修理費", "修復費", "維修費", "慰撫金", "精神賠償",
        "請求", "爰請求", "支出", "花費", "費用",
    ]
    return any(token in line for token in personal_tokens) and not any(token in line for token in claim_tokens)


def normalize_amount_value(raw: str) -> str:
    raw = raw.replace(",", "")
    if "萬" not in raw:
        return raw
    high, _, low = raw.partition("萬")
    high_value = int(high or "0") * 10000
    low_value = int(low or "0")
    return str(high_value + low_value)


def normalize_amount_display(raw: str) -> str:
    value = normalize_amount_value(raw)
    if not value.isdigit():
        return raw
    return f"{int(value):,}"


def normalize_amount_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{normalize_amount_display(match.group(1))}元"

    text = re.sub(AMOUNT_PATTERN, replace, text)
    text = re.sub(
        r"([0-9][0-9,]*)萬多元",
        lambda match: f"{int(match.group(1).replace(',', '')) * 10000:,}多元",
        text,
    )
    return text


def split_damage_source_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    parts: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        paragraph = re.sub(r"(?<!^)(?=\d+\.\s*)", "\n", paragraph.strip())
        lines = [line.strip(" ，,。；;") for line in paragraph.splitlines() if line.strip(" ，,。；;")]
        if any(re.match(r"^\d+\.\s*", line) for line in lines):
            current: list[str] = []
            for line in lines:
                if re.match(r"^\d+\.\s*", line):
                    if current:
                        parts.append(" ".join(current).strip(" ，,。；;"))
                    current = [line]
                elif current:
                    current.append(line)
                else:
                    parts.extend(split_plain_damage_sentences(line))
            if current:
                parts.append(" ".join(current).strip(" ，,。；;"))
        else:
            parts.extend(split_plain_damage_sentences(paragraph))
    return [part for part in parts if part]


def split_plain_damage_sentences(text: str) -> list[str]:
    return [part.strip(" ，,。；;") for part in re.split(r"(?<=。)\s*", text) if part.strip(" ，,。；;")]


def is_rate_or_reference_amount(sentence: str, amount_match: re.Match[str]) -> bool:
    if amount_match.start() > 0 and sentence[amount_match.start() - 1] in {".", "．"}:
        return True
    last_open = sentence.rfind("【", 0, amount_match.start())
    last_close = sentence.rfind("】", 0, amount_match.start())
    if last_open > last_close and "計算式" in sentence[last_open:amount_match.start()]:
        return True
    paren_open = max(sentence.rfind("（", 0, amount_match.start()), sentence.rfind("(", 0, amount_match.start()))
    paren_close = max(sentence.rfind("）", 0, amount_match.start()), sentence.rfind(")", 0, amount_match.start()))
    if paren_open > paren_close and "計算式" in sentence[paren_open:amount_match.start()]:
        return True
    window_start = max(0, amount_match.start() - 20)
    window_end = min(len(sentence), amount_match.end() + 20)
    window = sentence[window_start:window_end]
    immediate_prefix = sentence[max(0, amount_match.start() - 12):amount_match.start()]
    claim_prefix_window = sentence[max(0, amount_match.start() - 48):amount_match.start()]
    heading_prefix = sentence[:amount_match.start()]
    if re.match(r"\s*[:：]", sentence[amount_match.end():amount_match.end() + 3]):
        return False
    if any(token in window for token in ["每月收入", "月收入", "每月", "月薪", "日薪", "基本工資"]) or re.search(r"每\s*[0-9一二三四五六七八九十]+個月", window):
        if not any(token in immediate_prefix for token in ["工作損失", "無法工作損失", "收入損失", "薪資損失", "請求", "爰請求", "合計", "共計", "損失"]):
            return True
    personal_finance_prefix = sentence[max(0, amount_match.start() - 18):amount_match.start()]
    if any(token in personal_finance_prefix for token in ["所得為", "所得給付", "所得總額", "財產給付", "財產總額"]):
        if not any(token in immediate_prefix for token in ["工作損失", "收入損失", "薪資損失", "請求", "爰請求", "損失"]):
            return True
    rate_prefix = sentence[max(0, amount_match.start() - 16):amount_match.start()]
    if any(token in rate_prefix for token in ["每日", "每月", "日薪", "月薪", "基本工資", "月收入"]) or re.search(r"(?:每|一|1)\s*日|每\s*[0-9一二三四五六七八九十]+個月", rate_prefix):
        tail = sentence[amount_match.end():amount_match.end() + 60]
        if re.search(r"(?:請求|損害|費用|損失|合計|共計)[^。；\n]{0,45}" + AMOUNT_PATTERN, tail):
            return True
    strong_claim_tokens = [
        "醫療費用", "醫藥費用", "牙齒醫療費用", "交通費用", "工作損失費用", "工作損失",
        "不能工作損失", "收入損失", "看護費用", "修復費用", "修理費用", "維修費用",
        "財物損失", "精神慰撫金", "慰撫金", "共計", "合計", "請求", "爰請求", "支出",
    ]
    if any(token in claim_prefix_window for token in strong_claim_tokens):
        return False
    if re.search(r"(共|合計|總計|支出|請求|爰請求|為)\s*$", immediate_prefix):
        return False
    colon_positions = [pos for pos in [sentence.find("："), sentence.find(":")] if pos >= 0]
    colon_pos = min(colon_positions) if colon_positions else -1
    if colon_pos >= 0 and amount_match.start() > colon_pos and re.search(AMOUNT_PATTERN, sentence[:colon_pos]):
        return True
    if any(token in sentence for token in ["達成和解", "和解並未", "賠償原告"]) and any(token in sentence for token in ["訴外人", "和解", "已與"]):
        return True
    if any(token in window for token in ["事故前市價", "修復後價值", "價值僅剩"]):
        if not any(token in immediate_prefix for token in ["請求", "減損", "損害"]):
            return True
    if "\n" not in heading_prefix and re.search(r"^[（(]?[一二三四五六七八九十]+[）)]?[^。；\n]{0,30}部分(?:合計|共計)?", heading_prefix):
        return True
    if re.search(r"(合計|總計|共計|總共|共損失|損失約)[^。；\n]{0,40}元", sentence[amount_match.end():]):
        if not any(token in immediate_prefix for token in ["總計", "合計", "總共", "共計"]):
            return True
    if re.search(r"僅以[^。；\n]{0,20}元為度", sentence[amount_match.end():]):
        if not any(token in immediate_prefix for token in ["僅以", "為度", "請求", "賠償其中"]):
            return True
    if any(token in immediate_prefix for token in ["行情", "僅", "未超出"]):
        return True
    if "各" not in immediate_prefix and re.search(rf"各\s*{AMOUNT_PATTERN}", sentence[amount_match.end():]):
        return True
    if any(token in window for token in ["薪資所得", "所得給付", "所得總額", "所得為", "給付總額", "財產給付", "財產總額", "月收入", "日薪", "月薪", "基本工資"]):
        if not any(token in immediate_prefix for token in ["工作損失", "無法工作損失", "收入損失", "薪資損失", "請求", "爰請求", "合計", "受有", "損失"]):
            return True
    if "包括" in sentence and any(token in window for token in ["工資費用", "零件費用"]):
        return True
    if any(token in immediate_prefix for token in ["損害", "總計", "合計", "共計", "請求", "爰請求"]):
        return False
    rate_tokens = [
        "行情", "約為", "每小時", "1日", "一日", "每日", "每次", "每月", "月薪", "日薪",
        "基本工資", "半日", "全日看護以", "就近照護以", "以每日", "計算基準", "手術費用", "療程",
    ]
    if any(token in window for token in rate_tokens):
        claim_prefix_tokens = [
            "工作損失", "無法工作損失", "收入損失", "薪資損失", "看護費用",
            "醫療費用", "醫藥費", "診療費", "治療費用", "復健費用",
            "請求", "爰請求", "合計", "共計",
        ]
        if not any(token in immediate_prefix for token in claim_prefix_tokens):
            return True
    if any(token in window for token in ["總計", "合計", "共計", "總共", "支出", "請求", "爰請求", "計新台幣", "計新臺幣", "慰撫金"]):
        return False
    return False


def split_compensation_facts_into_items(text: str) -> str:
    text = re.sub(r"(\d+\.\s*)", r"\n\1", text)
    text = re.sub(r"(（[一二三四五六七八九十]+）)", r"\n\1", text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return "\n\n".join(paragraphs)


def infer_damage_labels(text: str) -> list[str]:
    labels = []
    rules = [
        ("醫療費用", ["醫療", "醫藥", "診療", "住院", "門診", "手術", "復健"]),
        ("交通費用", ["交通費", "計程車", "車資", "往返"]),
        ("勞動能力減損", ["勞動能力損失", "減少勞動能力", "勞動能力減損"]),
        ("工作損失", ["工作損失", "薪資", "收入", "不能工作", "無法工作"]),
        ("看護費用", ["看護", "照護", "照顧", "幫傭"]),
        ("車輛修復費用", ["修車", "修復", "修理", "維修", "零件", "工資", "鈑金", "機車", "汽車", "車輛"]),
        ("精神慰撫金", ["慰撫金", "精神", "痛苦", "憂鬱", "失眠"]),
        ("財物損失", ["手機", "眼鏡", "安全帽", "衣服", "褲子", "鞋子", "手錶", "物品"]),
        ("其他必要費用", ["增加生活必要費用", "生活必要費用", "營養品", "護具", "輔具", "尿布", "便器", "用品", "牙齒"]),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels or ["一般損害項目"]


def infer_damage_label_for_amount(text: str, amount_match: re.Match[str]) -> str:
    start, end = amount_match.span()
    close_window = text[max(0, start - 18):end]
    window = text[max(0, start - 36):end]
    wide_prefix = text[max(0, start - 90):end]
    local_prefix = text[max(0, start - 180):start]
    nearest_label = infer_nearest_damage_label(local_prefix)
    if nearest_label:
        return nearest_label
    close_rules = [
        ("交通費用", ["交通接送費", "交通費", "計程車費", "計程車費用", "車資", "往返"]),
        ("勞動能力減損", ["勞動能力損失", "減少勞動能力", "勞動能力減損"]),
        ("工作損失", ["工作損失", "薪資損失", "收入損失", "不能工作", "無法工作"]),
        ("看護費用", ["看護費", "看護費用", "照護費", "照顧費", "幫傭"]),
        ("其他必要費用", ["增加生活必要費用", "生活必要費用", "醫療用品費", "用品費", "器材費"]),
        ("醫療費用", ["醫藥及輔具費", "醫藥費", "醫療費用", "醫療費", "診療費", "診用", "看診費", "住院費", "門診費"]),
        ("財物損失", ["眼鏡", "安全帽", "衣服", "褲子", "鞋子", "手錶", "物品", "手機", "電腦", "筆記型電腦", "筆電"]),
        ("車輛修復費用", ["修理費", "修復費", "維修費", "修車費"]),
        ("精神慰撫金", ["慰撫金", "精神賠償"]),
    ]
    for label, keywords in close_rules:
        if any(keyword in close_window for keyword in keywords):
            return label
    if any(keyword in wide_prefix for keyword in ["精神科", "心理疾患", "心理治療", "治療心理"]):
        if "慰撫金" not in wide_prefix and "精神賠償" not in wide_prefix:
            return "醫療費用"
    if any(keyword in wide_prefix for keyword in ["診用", "看診費", "診療費", "醫療費", "醫藥費", "醫療費用", "診斷及證明書費"]):
        return "醫療費用"
    if any(keyword in wide_prefix for keyword in ["手機", "眼鏡", "安全帽", "鞋子", "衣服", "褲子", "手錶", "電腦", "筆記型電腦", "筆電"]):
        return "財物損失"
    if any(keyword in wide_prefix for keyword in ["修理費", "修復費", "維修費", "修車", "零件", "鈑金", "系爭汽車", "系爭機車", "機車", "汽車"]):
        return "車輛修復費用"
    rules = [
        ("精神慰撫金", ["慰撫金", "精神"]),
        ("看護費用", ["看護", "照護", "照顧", "幫傭"]),
        ("交通費用", ["交通接送費", "交通費", "車資", "往返"]),
        ("勞動能力減損", ["勞動能力損失", "減少勞動能力", "勞動能力減損"]),
        ("工作損失", ["工作損失", "薪資損失", "收入損失", "不能工作", "無法工作", "休養期間"]),
        ("醫療費用", ["醫藥費", "醫藥及輔具費", "醫療費用", "醫療復健", "復健費用", "診療費", "診用", "看診費", "住院費", "門診費"]),
        ("財物損失", ["眼鏡", "安全帽", "衣服", "褲子", "鞋子", "手錶", "物品", "手機", "電腦", "筆記型電腦"]),
        ("其他必要費用", ["增加生活必要費用", "生活必要費用", "輔具費", "營養品", "護具", "尿布", "便器", "用品", "牙齒"]),
    ]
    for label, keywords in rules:
        if any(keyword in window for keyword in keywords):
            return label
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "一般損害項目"


def infer_nearest_damage_label(text: str) -> str:
    recent = text[-90:]
    if any(token in recent for token in ["筆記型電腦", "筆電", "電腦", "手機", "眼鏡", "安全帽", "鞋子"]) and any(token in recent for token in ["修理費用", "維修費用", "修復費用"]):
        return "財物損失"
    rules = [
        ("醫療費用", ["牙齒醫療費用", "醫療費用", "醫藥費用", "醫藥費", "治療費用"]),
        ("交通費用", ["交通費用", "計程車", "車資", "來回", "回診"]),
        ("工作損失", ["工作損失費用", "工作損失", "無法工作", "不能工作", "收入損失", "薪資損失"]),
        ("看護費用", ["看護費用", "看護費", "照護費用", "照顧費用"]),
        ("車輛修復費用", ["車輛修復費用", "修復費用", "修車費", "機車因本件", "系爭機車"]),
        ("財物損失", ["筆記型電腦", "電腦", "手機", "眼鏡", "安全帽", "鞋子", "財物"]),
        ("精神慰撫金", ["精神慰撫金", "慰撫金"]),
    ]
    positions: list[tuple[int, str]] = []
    for label, keywords in rules:
        for keyword in keywords:
            pos = text.rfind(keyword)
            if pos >= 0:
                positions.append((pos, label))
    if not positions:
        return ""
    return max(positions, key=lambda item: item[0])[1]


def build_generation_support_context(similar_cases: list[dict], parent_map: dict[str, str], corpus_by_id: dict[str, dict]) -> str:
    case_blocks = []
    for case in similar_cases:
        parent_id = parent_map.get(case["case_id"])
        parent_note = ""
        if parent_id and parent_id in corpus_by_id:
            parent = corpus_by_id[parent_id]
            parent_note = (
                f"；父案例={parent_id}"
                f"（F={parent['severity_scores']['Fact']},"
                f" I={parent['severity_scores']['Injury']},"
                f" C={parent['severity_scores']['Compensation']}）"
            )
        damage_labels = "、".join(infer_damage_labels(case["comp_text"]))
        case_blocks.append(
            f"相似案例{case['rank']}：case_id={case['case_id']}，distance={case['distance']:.4f}，score={case['case_score']:.4f}{parent_note}\n"
            f"- 可參考的損害結構：{damage_labels}\n"
            f"- 事故摘要：{case['fact_text'][:120]}\n"
            f"- 傷勢摘要：{case['injury_text'][:120]}"
        )
    return "\n\n".join(case_blocks)


def build_damages_prompt(comp_facts: str, injuries: str, support_context: str, parties: dict, constraints: dict) -> str:
    preprocessed = split_compensation_facts_into_items(comp_facts)
    damage_outline = build_plaintiff_damage_outline(comp_facts, parties, constraints)
    allowed_amounts_text = "、".join(sorted(constraints["allowed_amounts"])) if constraints["allowed_amounts"] else "無"
    allowed_hospitals_text = "、".join(sorted(constraints["allowed_hospitals"])) if constraints["allowed_hospitals"] else "無"
    allowed_income_text = "、".join(sorted(constraints["allowed_income_terms"])) if constraints["allowed_income_terms"] else "無"
    allowed_personal_context_text = "、".join(sorted(constraints["allowed_personal_context"])) if constraints["allowed_personal_context"] else "無"
    plaintiff_injuries_text = "；".join(
        f"{name}：{injury}" for name, injury in constraints["plaintiff_injuries"].items()
    ) if constraints["plaintiff_injuries"] else "無"
    required_items_text = "；".join(
        f"{item['label']}={item['amount_raw']}元（原文：{item['source_line']}）" for item in constraints["required_items"]
    ) if constraints["required_items"] else "無"
    return f"""你是台灣律師，請把以下損害賠償事實整理成起訴狀中的損害項目段落。

當事人資訊：
原告：{parties.get('原告', '原告')}（共{parties.get('原告數量', 1)}名）
被告：{parties.get('被告', '被告')}（共{parties.get('被告數量', 1)}名）

原告受傷情形：
{injuries}

原始損害描述：
{preprocessed}

結構化損害事實：
{damage_outline}

SDKG相似案例輔助資訊：
{support_context}

注意：以下相似案例僅供損害項目分類、段落結構與法律書寫方式參考，不是本案事實來源。

Query 中明示可使用的敏感資訊：
- 金額：{allowed_amounts_text}
- 醫療院所：{allowed_hospitals_text}
- 工作/收入資訊：{allowed_income_text}
- 慰撫金相關個人化事實：{allowed_personal_context_text}
- 各原告傷勢對照：{plaintiff_injuries_text}
- 必須保留的賠償項目：{required_items_text}

要求：
1. 使用「（一）」「（二）」等編號。
2. 保留原文中的金額、醫院名稱、計算式與重要細節。
3. 項目名稱可整理成：醫療費用、交通費用、工作損失、看護費用、慰撫金、車輛修復費用等。
4. SDKG相似案例只能幫助你判斷常見損害項目結構與起訴狀寫法，不可借用其中的金額、醫院名稱、日期、職業、收入、姓名、事故地點或事故細節。
5. 直接輸出損害段落，不要解釋。
6. 不要輸出「綜上所述」「綜上所陳」「總計」「請求被告賠償」「按年息5%」「繕本送達翌日起」等結論句。
7. 若原始損害描述未出現某金額或事實，不能自行補寫。
8. 只有上面列出的敏感資訊可以出現在答案中；若 query 沒有提到某醫院、職業、薪資、所得或金額，就不能自行補寫。
9. 若 query 已明列賠償項目與金額，該項目不得漏掉。
10. 每一個賠償項目都要盡量引用原始損害描述中的事實理由，例如「需休養1個月」、「傷勢不良於行」、「車輛因事故受損需修理」、「耗費時間與精神」等；不要只剩項目名稱與金額。
11. 若有多名原告，應依原文分別保留各原告的傷勢、賠償項目與金額；不得混淆不同原告的損害，也不得自行拆分或合併金額。
12. 第一次寫到各原告「因本件事故受傷」時，不可只寫抽象句，必須接續引用「原告受傷情形」中的具體傷勢，例如頭部外傷、蜘蛛網膜下出血、硬腦膜下出血、骨折、挫傷、撕裂傷等；若有多名原告，應分別寫出各原告受有何種傷害。
13. 寫精神慰撫金時，必須優先使用 query 已提到的個人化事實，例如年齡、學歷、職業、所得、名下財產、家庭負擔、加護病房治療、長期就醫、刑事偵審或民事訴訟所造成的時間與精神耗費；若 query 沒有提到，不得自行補寫。
14. 多名原告案件中，不得用「亦因本件事故受傷」「同受傷害」取代第二名或後續原告的具體傷勢；每一名原告第一次出現於醫療費用、看護費用或慰撫金項目時，都要寫出其具體傷勢。
15. 不得只輸出空的賠償項目標題；若建立「看護費用」「醫療費用」「精神慰撫金」等項目，該標題下必須包含原文中的事實理由與金額。
16. 事實理由應優先沿用「結構化損害事實」中「事實依據」的原文用語，不要過度摘要；看護費用應保留住院、出院、生活無法自理、居家照護、每日看護費與總額等描述；精神慰撫金應保留學歷、職業、收入、所得、財產、無工作、無收入、長期就醫、肇事逃逸、不聞不問等原文已有的個人化事實。
17. 若有多名原告，輸出格式應採原告分組：先以「一、原告○○部分」「二、原告○○部分」作為中標題，再於各原告底下使用「（一）」「（二）」列出該原告的損害項目；未特定或共同財產損害可置於最後獨立段落。
18. 「結構化損害事實」中的 draft 欄位是主要寫作骨架，應優先採用；field 欄位有 evidence、calculation、period、care_need、personal_context 時，必須盡量寫入該項目段落。
"""


def build_plaintiff_damage_outline(comp_facts: str, parties: dict, constraints: dict) -> str:
    plaintiff_names = extract_party_names(parties.get("原告", ""))
    for name in constraints.get("plaintiff_injuries", {}):
        if is_specific_plaintiff_name(name) and name not in plaintiff_names:
            plaintiff_names.append(name)
    if not plaintiff_names:
        return "無"
    items = extract_scoped_damage_items(comp_facts, plaintiff_names)
    if not items:
        items = constraints.get("required_items", [])
    if not items:
        return "無"
    items = attach_injuries_to_damage_items(items, constraints.get("plaintiff_injuries", {}), "")

    if len(plaintiff_names) > 1:
        return format_grouped_damage_outline(items, plaintiff_names)

    lines = []
    for idx, item in enumerate(items, start=1):
        fact = build_damage_fact(item, plaintiff_names)
        lines.extend(format_damage_fact_lines(f"{idx}.", fact))
    return "\n".join(lines)


def build_structured_damage_section(
    comp_facts: str,
    injuries: str,
    parties: dict,
    constraints: dict | None = None,
    accident_facts: str = "",
    style_level: int = 0,
) -> str:
    if constraints is None:
        constraints = extract_damage_constraints(comp_facts, injuries)
    plaintiff_names = extract_party_names(parties.get("原告", ""))
    for name in constraints.get("plaintiff_injuries", {}):
        if is_specific_plaintiff_name(name) and name not in plaintiff_names:
            plaintiff_names.append(name)
    context_text = "\n".join(part for part in [accident_facts, comp_facts] if part)
    items = extract_scoped_damage_items(comp_facts, plaintiff_names, context_text)
    if not items:
        items = constraints.get("required_items", [])
    items = attach_damage_reason_context(items, comp_facts)
    items = attach_injuries_to_damage_items(items, constraints.get("plaintiff_injuries", {}), injuries)
    items = attach_plaintiff_damage_context(items, plaintiff_names)
    items = normalize_generic_plaintiff_items(items, plaintiff_names)
    items = merge_same_damage_label_items(items)
    constraints["allowed_amounts"].update(aggregate_merged_item_amounts(items))

    if len(plaintiff_names) > 1:
        section = render_grouped_damage_section(items, plaintiff_names, style_level)
    else:
        section = render_flat_damage_section(items, plaintiff_names, style_level)
    return clean_damage_section(normalize_amount_text(section), constraints)


def render_grouped_damage_section(items: list[dict], plaintiff_names: list[str], style_level: int = 0) -> str:
    plaintiff_names = [name for name in plaintiff_names if is_specific_plaintiff_name(name)]
    groups = {name: [] for name in plaintiff_names}
    shared = []
    for item in items:
        plaintiff = item.get("plaintiff", "")
        if plaintiff in groups:
            groups[plaintiff].append(item)
        else:
            shared.append(item)

    blocks = []
    section_idx = 1
    for plaintiff in plaintiff_names:
        group_items = groups[plaintiff]
        if not group_items:
            continue
        blocks.append(f"{to_chinese_section_marker(section_idx)}原告{plaintiff}部分")
        for item_idx, item in enumerate(group_items, start=1):
            blocks.append(render_damage_item_block(item_idx, item, plaintiff_names, style_level))
        section_idx += 1
    if shared:
        blocks.append(f"{to_chinese_section_marker(section_idx)}未特定或共同財產損害")
        for item_idx, item in enumerate(shared, start=1):
            blocks.append(render_damage_item_block(item_idx, item, plaintiff_names, style_level))
    return "\n\n".join(blocks)


def render_flat_damage_section(items: list[dict], plaintiff_names: list[str], style_level: int = 0) -> str:
    return "\n\n".join(render_damage_item_block(idx, item, plaintiff_names, style_level) for idx, item in enumerate(items, start=1))


def normalize_generic_plaintiff_items(items: list[dict], plaintiff_names: list[str]) -> list[dict]:
    specific = [name for name in plaintiff_names if is_specific_plaintiff_name(name)]
    if len(specific) != 1:
        return items
    normalized = []
    for item in items:
        copied = dict(item)
        if copied.get("plaintiff") in {"", "原告", "未特定原告"}:
            copied["plaintiff"] = specific[0]
        normalized.append(copied)
    return normalized


def merge_same_damage_label_items(items: list[dict]) -> list[dict]:
    mergeable_labels = {
        "醫療費用", "交通費用", "看護費用", "工作損失", "勞動能力減損",
        "車輛修復費用", "財物損失", "其他必要費用",
    }
    merged: list[dict] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for item in items:
        plaintiff = item.get("plaintiff", "")
        label = item.get("label", "")
        key = (plaintiff, label)
        subclaim = damage_item_subclaim(item)
        if label not in mergeable_labels:
            if key in index_by_key and label == "精神慰撫金":
                existing = merged[index_by_key[key]]
                if mental_item_score(item) > mental_item_score(existing):
                    copied = dict(item)
                    copied["subclaims"] = [subclaim]
                    merged[index_by_key[key]] = copied
                continue
            if key in index_by_key and should_skip_repeated_same_amount_subclaim(label, merged[index_by_key[key]].get("subclaims", []), subclaim):
                continue
            copied = dict(item)
            copied["subclaims"] = [subclaim]
            if key not in index_by_key:
                index_by_key[key] = len(merged)
            merged.append(copied)
            continue

        if key not in index_by_key:
            copied = dict(item)
            copied["subclaims"] = [subclaim]
            index_by_key[key] = len(merged)
            merged.append(copied)
            continue

        target = merged[index_by_key[key]]
        if should_skip_repeated_same_amount_subclaim(label, target.get("subclaims", []), subclaim):
            continue
        target.setdefault("subclaims", []).append(subclaim)
        old_amount = safe_parse_amount(target.get("amount_value", target.get("amount_raw", ""))) or 0
        new_amount = safe_parse_amount(item.get("amount_value", item.get("amount_raw", ""))) or 0
        total = old_amount + new_amount
        target["amount_value"] = str(total)
        target["amount_raw"] = normalize_amount_display(str(total))
        for field in ["source_line", "source_span", "amount_context", "plaintiff_context", "injury_source"]:
            target[field] = merge_text_fragments(target.get(field, ""), item.get(field, ""))
    return merged


def mental_item_score(item: dict) -> int:
    text = " ".join(str(item.get(field, "")) for field in ["amount_context", "source_line", "source_span"])
    score = 0
    if re.search(r"(?:請求|爰請求|賠償|命被告賠償)[^。；\n]{0,20}(?:精神慰撫金|慰撫金)", text):
        score += 20
    if re.search(r"(?:精神慰撫金|慰撫金)[^。；\n]{0,20}" + AMOUNT_PATTERN, text):
        score += 10
    if re.search(r"(所得|財產|月收入|日薪|月薪|收入)", text):
        score -= 10
    amount = safe_parse_amount(item.get("amount_value", item.get("amount_raw", ""))) or 0
    return score + min(amount // 100000, 5)


def should_skip_repeated_same_amount_subclaim(label: str, existing_subclaims: list[dict], new_subclaim: dict) -> bool:
    new_amount = new_subclaim.get("amount_value", "")
    if not new_amount:
        return False
    if any(is_total_summary_subclaim(old) for old in existing_subclaims):
        return True
    if existing_subclaims and is_total_summary_subclaim(new_subclaim):
        return True
    return any(old.get("amount_value") == new_amount for old in existing_subclaims)


def is_total_summary_subclaim(subclaim: dict) -> bool:
    text = subclaim.get("text", "")
    return bool(re.search(r"(?:合計|總計|共計|總共|總計支出|合計共|共為|總額)", text))


def aggregate_merged_item_amounts(items: list[dict]) -> set[str]:
    allowed = set()
    for item in items:
        amount = safe_parse_amount(item.get("amount_value", item.get("amount_raw", ""))) or 0
        if amount <= 0:
            continue
        allowed.add(str(amount))
        allowed.add(f"{amount:,}")
    return allowed


def damage_item_subclaim(item: dict) -> dict:
    amount_raw = item.get("amount_raw", "")
    amount_value = item.get("amount_value", normalize_amount_value(amount_raw))
    text = item.get("amount_context") or item.get("source_line", "")
    return {
        "label": item.get("label", ""),
        "amount_raw": amount_raw,
        "amount_value": amount_value,
        "text": clean_source_fragment(text),
    }


def merge_text_fragments(left: str, right: str) -> str:
    values = []
    for value in [left, right]:
        value = clean_source_fragment(value)
        if not value:
            continue
        if any(value == old or value in old for old in values):
            continue
        values = [old for old in values if old not in value]
        values.append(value)
    return "；".join(values)


def render_damage_item_block(index: int, item: dict, plaintiff_names: list[str], style_level: int = 0) -> str:
    fact = build_damage_fact(item, plaintiff_names, style_level)
    title = fact["item"]
    body = fact.get("draft") or fact.get("source_span") or ""
    return f"{to_chinese_item_marker(index)}{title}\n{body}".strip()


def to_chinese_section_marker(index: int) -> str:
    numerals = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    if 1 <= index <= 10:
        return f"{numerals[index]}、"
    return f"{index}、"


def format_grouped_damage_outline(items: list[dict], plaintiff_names: list[str]) -> str:
    plaintiff_names = [name for name in plaintiff_names if is_specific_plaintiff_name(name)]
    groups = {name: [] for name in plaintiff_names}
    shared = []
    for item in items:
        plaintiff = item.get("plaintiff", "")
        if plaintiff in groups:
            groups[plaintiff].append(item)
        else:
            shared.append(item)

    lines = []
    section_idx = 1
    for plaintiff in plaintiff_names:
        group_items = groups[plaintiff]
        if not group_items:
            continue
        lines.append(f"{section_idx}. 原告{plaintiff}部分")
        for item_idx, item in enumerate(group_items, start=1):
            fact = build_damage_fact(item, plaintiff_names)
            lines.extend(format_damage_fact_lines(f"  {item_idx})", fact))
        section_idx += 1
    if shared:
        lines.append(f"{section_idx}. 未特定或共同財產損害")
        for item_idx, item in enumerate(shared, start=1):
            fact = build_damage_fact(item, plaintiff_names)
            lines.extend(format_damage_fact_lines(f"  {item_idx})", fact))
    return "\n".join(lines)


def attach_injuries_to_damage_items(items: list[dict], injury_map: dict[str, str], injuries: str = "") -> list[dict]:
    enriched = []
    for item in items:
        copied = dict(item)
        if injuries:
            copied["injury_source"] = injuries
        plaintiff = copied.get("plaintiff", "")
        if plaintiff in injury_map:
            copied["injury"] = injury_map[plaintiff]
        elif plaintiff in {"", "原告", "未特定原告"} and "原告" in injury_map:
            copied["injury"] = injury_map["原告"]
            copied["plaintiff"] = "原告"
        elif "原告" in injury_map and not any(is_specific_plaintiff_name(name) for name in injury_map):
            copied["injury"] = injury_map["原告"]
        elif plaintiff == "原告2人" and injury_map:
            copied["injury"] = "；".join(f"{name}：{injury}" for name, injury in injury_map.items())
        enriched.append(copied)
    return enriched


def attach_plaintiff_damage_context(items: list[dict], plaintiff_names: list[str]) -> list[dict]:
    context_by_plaintiff: dict[str, list[str]] = {}
    all_context: list[str] = []
    for item in items:
        source = item.get("source_span") or item.get("source_line") or ""
        if not source:
            continue
        all_context.append(source)
        plaintiff = item.get("plaintiff", "")
        if plaintiff:
            claimant_source = source_for_claimant(source, plaintiff, plaintiff_names)
            context_by_plaintiff.setdefault(plaintiff, []).append(claimant_source or source)

    enriched = []
    for item in items:
        copied = dict(item)
        plaintiff = copied.get("plaintiff", "")
        if plaintiff in context_by_plaintiff:
            copied["plaintiff_context"] = "；".join(context_by_plaintiff[plaintiff])
        elif plaintiff == "原告2人":
            copied["plaintiff_context"] = "；".join(all_context)
        enriched.append(copied)
    return enriched


def attach_damage_reason_context(items: list[dict], comp_facts: str) -> list[dict]:
    enriched = []
    for item in items:
        copied = dict(item)
        label = copied.get("label", "")
        original_source = copied.get("source_span") or copied.get("source_line", "")
        reason = "" if has_substantive_damage_reason(label, original_source) else select_damage_reason_context(label, comp_facts)
        if label == "財物損失":
            property_context = select_damage_reason_context(label, comp_facts)
            if property_context and property_context not in original_source:
                reason = "；".join(part for part in [reason, property_context] if part)
        if reason and reason not in (copied.get("source_span") or ""):
            copied["source_span"] = "；".join(part for part in [copied.get("source_span") or copied.get("source_line", ""), reason] if part)
        enriched.append(copied)
    return enriched


def has_substantive_damage_reason(label: str, source: str) -> bool:
    source = clean_source_fragment(source)
    keywords_by_label = {
        "交通費用": ["不良於行", "計程車", "就醫", "搭乘", "往返"],
        "工作損失": ["無法工作", "不能工作", "需休養", "需要休養", "每月", "月薪", "日薪", "所得", "扣繳憑單", "薪資"],
        "勞動能力減損": ["勞動能力", "減少勞動能力", "勞動能力損失", "減損"],
        "車輛修復費用": ["受損", "修理", "修復", "維修", "零件", "工資"],
        "財物損失": ["毀損", "損壞", "受損", "單據", "收據", "發票", "維修"],
        "其他必要費用": ["增加生活必要費用", "生活必要費用", "醫療用品", "器材", "輔具", "毀損", "損壞", "受損", "單據", "收據", "發票"],
    }
    keywords = keywords_by_label.get(label, [])
    return any(keyword in source for keyword in keywords)


def select_damage_reason_context(label: str, comp_facts: str) -> str:
    keywords_by_label = {
        "交通費用": ["不良於行", "計程車", "交通費", "搭乘", "上下班"],
        "工作損失": ["無法工作", "不能工作", "休養", "月薪", "日薪", "薪資", "工作收入損失"],
        "勞動能力減損": ["勞動能力", "減少勞動能力", "勞動能力損失", "勞動能力減損"],
        "車輛修復費用": ["車輛", "機車", "汽車", "受損", "修理", "修復", "維修"],
        "財物損失": ["財物損壞", "毀損", "手機", "安全帽", "鞋子", "眼鏡", "衣服", "褲子", "手錶", "單據", "收據"],
        "其他必要費用": ["增加生活必要費用", "生活必要費用", "醫療用品", "器材", "營養品", "護具", "輔具", "尿布", "便器", "用品", "牙齒", "單據", "收據"],
    }
    keywords = keywords_by_label.get(label)
    if not keywords:
        return ""
    selected = []
    for sentence in split_damage_source_sentences(comp_facts):
        clean = clean_source_fragment(sentence)
        if not clean:
            continue
        if any(keyword in clean for keyword in keywords):
            selected.append(clean)
    return "；".join(selected[-2:])


def build_damage_fact(item: dict, plaintiff_names: list[str], style_level: int = 0) -> dict:
    plaintiff = item.get("plaintiff") or infer_plaintiff_from_text(item.get("source_line", ""), plaintiff_names) or "原告"
    label = item.get("label", "一般損害項目")
    amount = item.get("amount_raw", "")
    amount_value = item.get("amount_value", normalize_amount_value(amount))
    source = item.get("source_span") or item.get("source_line", "")
    source_line = item.get("amount_context") or item.get("source_line", source)
    injury_source = item.get("injury_source", "")
    injury = item.get("injury", "")
    local_claim = extract_amount_local_phrase(source_line, amount, amount_value)
    claimant_source = source_for_claimant(source, plaintiff, plaintiff_names)
    item_source = source if label == "工作損失" else source_line if label in {"財物損失", "車輛修復費用"} else claimant_source or source
    base_evidence_source = source if label == "醫療費用" else item_source if label in {"財物損失", "車輛修復費用"} else claimant_source or source
    if label == "醫療費用" and plaintiff in {"原告", "未特定原告", ""} and not extract_evidence_terms(base_evidence_source):
        evidence_source = "\n".join(part for part in [base_evidence_source, injury_source] if part)
    else:
        evidence_source = base_evidence_source
    context_source = item.get("plaintiff_context", "")
    if label == "精神慰撫金":
        detail_source = "\n".join(part for part in [source, context_source] if part)
    else:
        detail_source = claimant_source or source
    medical_source = "\n".join(part for part in [detail_source, injury_source] if part) if label == "醫療費用" else detail_source
    fields = {
        "claimant": plaintiff,
        "item": label,
        "amount": f"{amount}元" if amount else "未列金額",
        "amount_value": amount_value,
        "injury": injury,
        "basis": extract_basis_terms(item_source, label),
        "evidence": clean_joined_terms(extract_evidence_terms(evidence_source)),
        "calculation": clean_joined_terms(extract_calculation_terms(source_line if label == "工作損失" else source if label == "看護費用" else claimant_source or source)),
        "period": extract_period_terms(medical_source),
        "care_need": extract_care_need_terms(detail_source),
        "personal_context": clean_joined_terms(extract_ordered_personal_context(detail_source)),
        "medical_process": extract_medical_process_terms(medical_source),
        "damage_object": extract_damage_object_terms(item_source),
        "local_claim": local_claim,
        "subclaims": item.get("subclaims", []),
        "source_line": source_line,
        "source_span": source,
        "style_level": style_level,
    }
    fields["draft"] = draft_damage_sentence(fields)
    return fields


def format_damage_fact_lines(prefix: str, fact: dict) -> list[str]:
    ordered_keys = [
        "claimant", "item", "amount", "injury", "basis", "evidence", "calculation", "period",
        "care_need", "personal_context", "medical_process", "damage_object", "local_claim", "draft", "source_span",
    ]
    first = "｜".join(f"{key}={fact[key] or '無'}" for key in ["claimant", "item", "amount"])
    lines = [f"{prefix} {first}"]
    for key in ordered_keys[3:]:
        value = fact.get(key) or "無"
        lines.append(f"     {key}: {value}")
    return lines


def extract_basis_terms(source: str, label: str) -> str:
    if label == "醫療費用":
        return first_matching_clause(source, ["支出醫藥費", "支出醫療費用", "支出醫藥及輔具費", "為治療"])
    if label == "交通費用":
        return first_matching_clause(source, ["不良於行", "就醫", "往返醫院", "交通接送", "搭乘", "計程車"])
    if label == "看護費用":
        return first_matching_clause(source, ["看護", "照護", "照顧", "生活無法自理", "專人照顧"])
    if label == "工作損失":
        return extract_work_loss_basis_terms(source)
    if label == "勞動能力減損":
        return first_matching_clause(source, ["減少勞動能力", "勞動能力損失", "勞動能力減損", "勞動能力"])
    if label == "精神慰撫金":
        return first_matching_clause(source, ["精神痛苦", "身心", "痛苦", "慰撫金"])
    if label == "車輛修復費用":
        return first_matching_clause(source, ["修理費", "修復費", "受損", "維修", "需要修理"])
    if label == "財物損失":
        return first_matching_clause(source, ["維修費", "修理費", "損壞費用", "毀損", "受損", "支出"])
    return first_matching_clause(source, ["支出", "請求", "損害"])


def extract_work_loss_basis_terms(source: str) -> str:
    reason_tokens = ["需休養", "需要休養", "不能工作", "無法工作", "無法正常工作"]
    calculation_tokens = ["日薪", "月薪", "薪資", "所得", "扣繳憑單", "基本工資", "計算基準", "計算"]
    selected = []
    fallback = []

    clauses = [clean_source_fragment(part) for part in re.split(r"[。；\n]", source) if clean_source_fragment(part)]
    for clause in clauses:
        has_reason = any(token in clause for token in reason_tokens)
        has_calculation = any(token in clause for token in calculation_tokens)
        has_work_claim = re.search(r"(工作|收入|薪資|不能工作|無法工作).{0,20}(損失|請求)", clause)
        bare_amount_item = bool(re.search(AMOUNT_PATTERN, clause)) and len(clause) <= 28 and not has_reason and not has_calculation
        if bare_amount_item:
            continue
        if has_reason or has_calculation:
            selected.append(clause)
        elif has_work_claim:
            fallback.append(clause)

    values = selected or fallback
    return clean_joined_terms("；".join(values[-2:]))


def extract_amount_local_phrase(source: str, amount_raw: str, amount_value: str = "") -> str:
    if not amount_raw:
        return ""
    match = None
    target_value = amount_value or normalize_amount_value(amount_raw)
    for amount_match in re.finditer(AMOUNT_PATTERN, source):
        if normalize_amount_value(amount_match.group(1)) == target_value:
            start = max(0, amount_match.start() - 45)
            end = amount_match.end()
            match = re.search(r"[^，。；、]{0,45}" + re.escape(source[amount_match.start():amount_match.end()]), source[start:end])
            if match:
                phrase = match.group(0).strip(" ，。；、")
            else:
                phrase = source[start:end].strip(" ，。；、")
            if "）" in phrase and "（" not in phrase:
                vehicle_start = max(
                    source.rfind("系爭", 0, amount_match.start()),
                    source.rfind("車輛", 0, amount_match.start()),
                    source.rfind("汽車", 0, amount_match.start()),
                    source.rfind("機車", 0, amount_match.start()),
                )
                if vehicle_start >= 0:
                    phrase = source[vehicle_start:end].strip(" ，。；、")
            break
    else:
        escaped = re.escape(amount_raw)
        fallback = re.search(rf"[^，。；、]{{0,45}}{escaped}\s*元", source)
        if not fallback:
            return ""
        phrase = fallback.group(0).strip(" ，。；、")
    phrase = clean_source_fragment(normalize_amount_text(phrase))
    for marker in ["原告", "系爭", "支出", "請求", "爰請求", "合計", "共支出"]:
        pos = phrase.find(marker)
        if pos >= 0:
            return phrase[pos:]
    return phrase


def source_for_claimant(source: str, claimant: str, plaintiff_names: list[str]) -> str:
    if claimant in {"未特定原告", "原告2人"} or not claimant:
        return source
    start = source.find(claimant)
    if start < 0:
        return source
    end = len(source)
    for other in plaintiff_names:
        if other == claimant:
            continue
        pos = source.find(other, start + len(claimant))
        if pos >= 0:
            end = min(end, pos)
    return source[start:end].strip(" ，。；")


def extract_evidence_terms(source: str) -> str:
    return join_matches(source, [
        r"原告於[^。；]{0,80}醫院[^，。；]{0,24}(?:門診|就診|治療|檢查)",
        r"[^，。；]{0,24}醫院[^，。；]{0,24}(?:門診|就診|治療|檢查)",
        r"並提出[^，。；]{0,50}(?:醫療費用明細收據|住院醫療費用證明書|門診醫療費用明細表|醫療費用收據|醫療收據|診斷證明書|估價單|取貨單|扣繳憑單|所得資料|統一發票|免用統一發票收據|相關收據|維修單據|薪資證明書|員工服務證明書|在職證明書)[^，。；]{0,20}(?:為證|可證|可以證明|憑證|證明)?",
        r"此有[^，。；]{0,60}(?:為證|可證|可以證明|證明)",
        r"有[^，。；]{0,60}(?:醫療費用明細收據|住院醫療費用證明書|門診醫療費用明細表|醫療費用收據|醫療收據|診斷證明書|估價單|取貨單|扣繳憑單|所得資料|統一發票|免用統一發票收據|相關收據|維修單據|薪資證明書|員工服務證明書|在職證明書)[^，。；]{0,20}(?:為證|可證|可以證明|憑證|證明)?",
        r"[^，。；]{0,24}(?:醫療費用明細收據|住院醫療費用證明書|門診醫療費用明細表|醫療費用收據|醫療收據|診斷證明書|估價單|取貨單|扣繳憑單|所得資料|統一發票|免用統一發票收據|相關收據|維修單據|薪資證明書|員工服務證明書|在職證明書)[^，。；]{0,18}(?:為證|可證|可以證明|憑證|證明)?",
        r"並有[^，。；]{0,40}(?:為證|可證)",
        r"有[^，。；]{0,40}(?:為證|可證)",
    ])


def extract_calculation_terms(source: str) -> str:
    return join_matches(source, [
        r"參以[^，。；]{0,80}計算基準",
        r"以每日[^，。；]{0,40}計算",
        r"以日薪[^，。；]{0,40}計算",
        r"以月薪[^，。；]{0,40}計算",
        r"以[^，。；]{0,80}(?:平均數|計算)",
        r"每月工作所得[^，。；]{0,30}",
        r"每日[0-9萬,]+元",
        r"日薪約?[0-9萬,]+元",
        r"月薪約?[0-9萬,]+元",
        r"每月薪資[0-9萬,]+元",
        r"基本工資[^，。；]{0,40}",
    ])


def extract_period_terms(source: str) -> str:
    return join_matches(source, [
        r"自[^，。；]{0,30}起至[^，。；]{0,30}止",
        r"住院[^，。；]{0,24}",
        r"出院[^，。；]{0,24}",
        r"需休養[^，。；]{0,18}",
        r"需要休養[^，。；）)]{0,24}",
        r"休養期間[^，。；]{0,24}",
        r"一個月",
        r"[0-9]+(?:\.[0-9]+)?個月",
        r"[0-9]+(?:\.[0-9]+)?月",
        r"[一二三四五六七八九十0-9]+個月",
        r"[一二三四五六七八九十0-9]+週",
        r"[一二三四五六七八九十0-9]+天",
    ])


def extract_care_need_terms(source: str) -> str:
    return join_matches(source, [
        r"生活無法自理",
        r"日常生活[^，。；]{0,24}(?:照護|照顧|協助)",
        r"需專人照顧[^，。；]{0,18}",
        r"需他人[^，。；]{0,24}(?:照護|照顧|協助)",
        r"居家照護[^，。；]{0,18}",
        r"就近照護[^，。；]{0,18}",
    ])


def extract_ordered_personal_context(source: str) -> str:
    return join_matches(source, [
        r"年事已高",
        r"[小國高中大專]{1,2}學畢業",
        r"於事故發生時為[^，。；]{0,18}",
        r"職業為[^，。；]{0,24}",
        r"從事[^，。；]{0,24}",
        r"曾以[^，。；]{0,24}為業",
        r"現已退休",
        r"沒有工作",
        r"沒有月收入",
        r"月收入[^，。；]{0,24}",
        r"110年度所得給付總額[^，。；]{0,30}",
        r"所得給付總額[^，。；]{0,30}",
        r"財產給付總額[^，。；]{0,30}",
        r"名下[^，。；]{0,24}",
        r"子女[^，。；]{0,24}",
        r"未成年子女[^，。；]{0,30}",
        r"配偶[^，。；]{0,30}",
        r"肇事逃逸",
        r"拒絕承認肇事責任",
        r"毫無悔意",
        r"不聞不問",
        r"拒不負責",
        r"身心恐懼",
        r"經常失眠",
        r"自信受創",
        r"業績下降",
        r"心生恐懼",
        r"生活起居[^，。；]{0,24}",
        r"仰賴家人[^，。；]{0,18}",
        r"心靈[^，。；]{0,40}",
        r"精神痛苦",
        r"痛苦[^，。；]{0,40}",
    ])


def extract_medical_process_terms(source: str) -> str:
    return join_matches(source, [
        r"於[^，。；]{0,60}(?:醫院|該院)[^，。；]{0,30}(?:住院|門診|就診|接受物理治療|復健治療|治療|檢查)",
        r"原告於[^。；]{0,80}(?:門診|就診|治療|檢查)",
        r"於[^，。；]{0,40}(?:門診|就診|治療|檢查)",
        r"進行[^，。；]{0,24}檢查",
        r"急診[^，。；]{0,24}",
        r"入院治療",
        r"住院治療",
        r"加護病房",
        r"普通病房",
        r"接受[^，。；]{0,30}手術",
        r"植入[^，。；]{0,18}",
        r"出院",
        r"復健治療",
        r"往返醫療院所就醫",
    ])


def extract_damage_object_terms(source: str) -> str:
    return join_matches(source, [
        r"系爭汽車",
        r"系爭機車",
        r"行車紀錄器",
        r"隔熱紙",
        r"眼鏡",
        r"安全帽",
        r"鞋子",
        r"手機",
        r"筆記型電腦",
        r"電腦",
        r"手錶",
        r"衣服",
        r"機車",
        r"汽車",
        r"車輛",
        r"尿布",
        r"便器",
        r"輔具",
    ])


def format_damage_object_text(text: str) -> str:
    values = [clean_source_fragment(part) for part in text.split("；") if clean_source_fragment(part)]
    if not values:
        return ""
    values = list(dict.fromkeys(values))
    generic_vehicle_terms = {"車輛", "汽車", "機車"}
    if len(values) > 1 and any(value not in generic_vehicle_terms for value in values):
        values = [value for value in values if value not in generic_vehicle_terms]
    if len(values) == 1:
        return values[0]
    return "、".join(values[:-1]) + "及" + values[-1]


def first_matching_clause(source: str, keywords: list[str]) -> str:
    clauses = [part.strip(" ，。；") for part in re.split(r"[。；]", source) if part.strip(" ，。；")]
    for clause in clauses:
        if any(keyword in clause for keyword in keywords):
            return clean_source_fragment(clause)
    return ""


def join_matches(source: str, patterns: list[str]) -> str:
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            value = clean_source_fragment(match.group(0))
            if value and not any(value == old or value in old for old in values):
                values = [old for old in values if old not in value]
                values.append(value)
    return "；".join(values)


def clean_joined_terms(text: str) -> str:
    if not text:
        return ""
    values = []
    for value in [clean_source_fragment(part) for part in text.split("；")]:
        if value and not any(value == old or value in old for old in values):
            values = [old for old in values if old not in value]
            values.append(value)
    return "；".join(values)


def clean_source_fragment(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = normalize_amount_text(text)
    text = re.sub(r"[㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩]\s*", "", text)
    text = re.sub(r"(分別支出)\s*\1", r"\1", text)
    text = text.replace("計算式:", "計算式為").replace("計算式：", "計算式為")
    text = remove_duplicate_total_phrases(text)
    text = strip_inline_damage_field_label(text)
    text = re.sub(r"原告(?:且原告|並且原告)", "原告", text)
    text = text.replace("原告按", "原告依")
    text = text.replace("原告縱未因", "原告雖未因").replace("原告縱未", "原告雖未")
    text = text.replace("原告汽車", "系爭汽車")
    text = text.replace("原告機車", "系爭機車")
    text = text.replace("原告傷勢照片", "傷勢照片")
    text = text.replace("原告牙齒", "牙齒")
    text = text.replace("原告至少", "至少")
    if not re.match(r"^\s*[0-9]+\.[0-9]", text):
        text = re.sub(r"^\s*(?:\(?[0-9]+\)?[.、．]|（[一二三四五六七八九十]+）)\s*", "", text)
    text = strip_leading_damage_subheading(text)
    text = re.sub(r"(?<=[，。；\n])\s*(?:\(?[0-9]+\)?[、．]|\(?[0-9]+\)?\.(?![0-9])|（[一二三四五六七八九十]+）)\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:此外|又|另外|另查|末查)", "", text)
    text = re.sub(r"原告已(?=[0-9一二三四五六七八九十]+年基本工資)", "原告以", text)
    text = re.sub(r"^此有", "", text)
    text = re.sub(r"並有此有", "並有", text)
    text = re.sub(r"並有[、，；\s]+", "並有", text)
    text = re.sub(r"[，,、]\s*原告\s*$", "", text)
    return text.strip(" ：:，,。；; ")


def strip_inline_damage_field_label(text: str) -> str:
    labels = (
        "醫療費用|醫藥費用|醫藥費|醫療復健費用|交通費用|交通費|看護費用|看護費|"
        "工作損失|不能工作之損失|無法工作損失|薪資損失|收入損失|"
        "勞動能力減損|勞動能力損失|車輛修復費用|車輛修理費|車輛維修費用|車損修理費用|修復費用|財物損失|"
        "其他必要費用|精神慰撫金|慰撫金"
    )
    name = rf"(?:原告)?[{NAME_CHARS}]{{0,5}}"
    text = re.sub(rf"^\s*{name}(?:之)?(?:{labels})\s*(?:{AMOUNT_PATTERN})?\s*[:：]\s*", "", text)
    text = re.sub(rf"(?<=[。；\n])\s*{name}(?:之)?(?:{labels})\s*(?:{AMOUNT_PATTERN})?\s*[:：]\s*", "", text)
    text = re.sub(rf"(?<=[，,])\s*(?:{labels})\s*(?:{AMOUNT_PATTERN})?\s*[:：]\s*", "", text)
    text = re.sub(rf"^\s*(?:{labels})\s*(?:為)?\s*(?:{AMOUNT_PATTERN})?\s*[:：]\s*", "", text)
    return text


def strip_leading_damage_subheading(text: str) -> str:
    labels = (
        "醫療費用|醫藥費用|醫藥費|醫療復健費用|計程車費用|交通費用|看護費用|"
        "薪資損失|工作損失|不能工作之損失|筆電維修費用|機車維修費用|"
        "車輛修復費用|醫療用品費用|住宿費用|精神慰撫金|慰撫金"
    )
    return re.sub(
        rf"^(?:兩人之|原告[{NAME_CHARS}]{{1,5}})?(?:請求)?(?:{labels})部分\s*",
        "",
        text,
    )


def remove_duplicate_total_phrases(text: str) -> str:
    text = re.sub(r"(合計\s*([0-9,]+元))，\s*合計\s*\2", r"\1", text)
    text = re.sub(r"(共計\s*([0-9,]+元))，\s*合計\s*\2", r"\1", text)
    text = re.sub(r"(總計\s*([0-9,]+元))，\s*合計\s*\2", r"\1", text)
    return text


def filter_joined_terms_against(text: str, existing: str) -> str:
    kept = []
    for value in [clean_source_fragment(part) for part in text.split("；") if clean_source_fragment(part)]:
        if value in existing:
            continue
        if any(value == old or value in old for old in kept):
            continue
        kept = [old for old in kept if old not in value]
        kept.append(value)
    return "；".join(kept)


def format_evidence_phrase(evidence: str) -> str:
    evidence = filter_joined_terms_against(evidence, "")
    if not evidence:
        return ""
    evidence_values = [
        value for value in evidence.split("；")
        if not re.search(r"^[0-9,]+元|元[:：]", value)
    ]
    evidence = "；".join(evidence_values)
    if not evidence:
        return ""
    evidence = re.sub(r"^根據", "", evidence).strip(" ：:，,、。；")
    evidence = re.sub(r"^此有", "", evidence).strip(" ：:，,、。；")
    if evidence.startswith(("並有", "有", "並提出", "提出")):
        phrase = evidence
    else:
        phrase = f"並有{evidence}"
    if not re.search(r"(為證|可證|證明)$", phrase):
        phrase += "為證"
    return phrase


def ensure_claim_verb(claim: str) -> str:
    if not claim:
        return claim
    if re.search(r"^(原告|系爭|支出|請求|爰請求|合計|共支出|計|已支出)", claim):
        return claim
    return f"支出{claim}"


def evidence_tail(evidence: str) -> str:
    if not evidence:
        return "。"
    evidence = re.sub(r"^此有", "", evidence).strip(" ，。；")
    if evidence.startswith(("並", "有", "依")):
        return f"，{evidence}。"
    return f"，並有{evidence}。"


def labeled_evidence_tail(label: str, evidence: str) -> str:
    if not evidence:
        return "。"
    values = [clean_source_fragment(part) for part in evidence.split("；") if clean_source_fragment(part)]
    if label in {"車輛修復費用", "財物損失", "其他必要費用"}:
        relevant = [
            value for value in values
            if any(token in value for token in ["維修單據", "修理", "修復", "估價單", "收據", "發票", "取貨單"])
        ]
        if any("維修單據" in value for value in relevant):
            return "，並有維修單據為證。"
        if relevant:
            return evidence_tail("；".join(relevant[:1]))
        return "。"
    return evidence_tail("；".join(values))


def phrase_list(text: str) -> str:
    return text.replace("；", "，")


def draft_mental_damage_sentence(
    subject: str,
    injury_text: str,
    medical: str,
    period: str,
    care_need: str,
    personal: str,
    amount: str,
    style_level: int = 0,
) -> str:
    facts = [subject + injury_text]
    treatment = mental_treatment_phrase(medical)
    hospital_period = mental_hospital_period(period)
    rest_period = mental_rest_period(period)
    if treatment:
        facts.append(treatment)
    if hospital_period:
        facts.append(hospital_period)
    if rest_period:
        facts.append(rest_period)

    personal_text = clean_mental_personal_context(personal)
    if personal_text:
        facts.append(personal_text)

    mental_subject = "其等" if subject == "原告2人" else subject
    impact = f"上開傷勢及治療過程已造成{mental_subject}相當之身體痛苦，並使其日常生活受到影響，精神上亦受有痛苦"
    if care_need:
        impact = f"上開傷勢、治療及生活照顧需求已造成{mental_subject}相當之身體痛苦，並使其日常生活受到影響，精神上亦受有痛苦"
    if style_level >= 2 and not care_need:
        impact = f"上開傷勢及治療過程已造成{mental_subject}相當之身體痛苦，並使其日常生活與心理狀態受到影響，精神上亦受有痛苦"
    elif style_level >= 3 and care_need:
        impact = f"上開傷勢、治療及生活照顧需求已造成{mental_subject}相當之身體痛苦，並使其日常生活與心理狀態受到影響，精神上亦受有痛苦"
    if style_level == 4:
        impact = f"上開傷勢及治療經過已造成{mental_subject}身體與精神上之痛苦，爰有請求慰撫金之必要"
    elif style_level >= 5:
        impact = f"上開傷勢已造成{mental_subject}身體及精神上痛苦，爰請求慰撫金"
    facts.append(impact)
    if style_level == 4:
        facts.append(f"審酌{mental_subject}所受傷害、治療經過及生活影響等情，請求精神慰撫金{amount}")
    elif style_level >= 5:
        facts.append(f"審酌上開一切情狀，請求精神慰撫金{amount}")
    elif style_level >= 3:
        facts.append(f"審酌{mental_subject}所受傷害程度、治療過程、休養期間、生活影響及其資力情形，爰請求精神慰撫金{amount}")
    else:
        facts.append(f"審酌{mental_subject}所受傷害程度、治療過程、休養期間及其資力情形，爰請求精神慰撫金{amount}")
    return "。".join(fact.strip(" ，。；") for fact in facts if fact.strip(" ，。；")) + "。"


def mental_treatment_phrase(medical: str) -> str:
    if not medical:
        return ""
    values = [part.strip(" ，。；") for part in medical.split("；") if part.strip(" ，。；")]
    selected = []
    for value in values:
        if any(token in value for token in ["手術", "植入", "加護病房", "急診", "入院"]):
            if value == "入院治療" and any("急診" in old for old in selected):
                continue
            if value.startswith("急診"):
                value = "經急診入院治療"
            if value not in selected:
                selected.append(value)
    if not selected:
        return ""
    return "，".join(selected[:3])


def mental_hospital_period(period: str) -> str:
    if not period:
        return ""
    for value in [part.strip(" ，。；") for part in period.split("；")]:
        match = re.search(r"出院共計([0-9一二三四五六七八九十]+天)", value)
        if match:
            return f"住院{match.group(1)}"
        if "住院" in value or "出院共計" in value:
            return value
    for value in [part.strip(" ，。；") for part in period.split("；")]:
        if re.search(r"[0-9一二三四五六七八九十]+天", value):
            return f"住院{value}"
    return ""


def clean_mental_personal_context(personal: str) -> str:
    kept = []
    seen_kinds = set()
    for part in re.split(r"[；;]", personal or ""):
        part = part.strip(" ，,、；;")
        if not part:
            continue
        if any(token in part for token in ["工作損失", "計算式", "基本工資", "與從事家務勞動", "輪流代為照顧"]):
            continue
        part = re.sub(r"(月收入[^，。；]*?元)原告.*", r"\1", part)
        part = re.sub(r"(月薪[^，。；]*?元)原告.*", r"\1", part)
        part = re.sub(r"^痛苦\s*原告主張其因受有上開傷害", "原告主張其因受有上開傷害而精神痛苦", part)
        part = re.sub(r"痛苦外$", "痛苦", part)
        if part in {"精神痛苦", "痛苦"}:
            continue
        kind = ""
        if "月收入" in part:
            kind = "月收入"
        elif "所得給付" in part or "財產給付" in part:
            kind = "所得財產"
        if kind and kind in seen_kinds:
            continue
        if kind:
            seen_kinds.add(kind)
        kept.append(part)
    return "；".join(kept).strip(" ，。；")


def mental_rest_period(period: str) -> str:
    if not period:
        return ""
    for value in [part.strip(" ，。；") for part in period.split("；")]:
        if "需休養" in value:
            return value
    return ""


def draft_damage_sentence(fact: dict) -> str:
    claimant = fact["claimant"]
    label = fact["item"]
    amount = fact["amount"]
    amount_raw = amount.replace("元", "")
    amount_value = fact.get("amount_value") or normalize_amount_value(amount_raw)
    injury = fact.get("injury") or ""
    basis = fact.get("basis") or ""
    evidence = fact.get("evidence") or ""
    calculation = fact.get("calculation") or ""
    period = fact.get("period") or ""
    care_need = fact.get("care_need") or ""
    personal = fact.get("personal_context") or ""
    medical = fact.get("medical_process") or ""
    damage_object = fact.get("damage_object") or ""
    local_claim = fact.get("local_claim") or ""
    source_line = fact.get("source_line") or ""
    source_span = fact.get("source_span") or ""
    subclaims = fact.get("subclaims") or []
    style_level = int(fact.get("style_level") or 0)

    if claimant in {"原告", "未特定原告"}:
        subject = "原告"
    elif claimant == "原告2人":
        subject = claimant
    else:
        subject = f"原告{claimant}"
    injury_text = f"因本件事故受有{injury}" if injury else "因本件事故受有損害"
    if label == "醫療費用":
        if len(subclaims) > 1:
            subclaim_text = format_subclaim_details(subclaims)
            parts = [subject + injury_text]
            if subclaim_text:
                parts.append(f"分別支出{subclaim_text}")
            evidence_value = filter_joined_terms_against(evidence, "，".join(parts))
            evidence_phrase = format_evidence_phrase(evidence_value)
            if evidence_phrase:
                parts.append(evidence_phrase)
            parts.append(f"合計醫療費用{amount}")
            return "，".join(parts) + "。"
        claim = ensure_claim_verb(clean_source_fragment(local_claim or basis or f"支出醫療費用{amount}"))
        parts = [subject + injury_text]
        for value in [medical]:
            value = clean_source_fragment(value)
            if value and value not in parts:
                parts.append(value)
        evidence_value = filter_joined_terms_against(evidence, "，".join(parts))
        evidence_phrase = format_evidence_phrase(evidence_value)
        if evidence_phrase:
            parts.append(evidence_phrase)
        return "，".join(parts) + f"，{claim}。"
    if label == "交通費用":
        if len(subclaims) > 1:
            subclaim_text = format_traffic_subclaim_details(subclaims)
            return f"{subject}因本件事故後有交通往返或代步需要，分別支出{subclaim_text}，合計交通費用{amount}。"
        tail = evidence_tail(evidence)
        claim = ensure_claim_verb(clean_source_fragment(local_claim or f"支出交通費用{amount}"))
        reason = clean_traffic_reason(basis or "因本件事故後有交通往返需要", subject)
        if subject != "原告":
            reason = re.sub(r"^原告", subject, reason)
        reason = re.sub(r"，?支出交通費用$", "", reason)
        if amount_present_in_text(amount_raw, amount_value, reason):
            return ensure_sentence_punctuation(reason if reason.startswith(subject) else f"{subject}{reason}")
        return ensure_sentence_punctuation(f"{reason if reason.startswith(subject) else subject + reason}，{claim}{tail}")
    if label == "看護費用":
        if len(subclaims) > 1:
            subclaim_text = format_subclaim_details(subclaims)
            return f"{subject}{injury_text}，分別支出{subclaim_text}，合計看護費用{amount}。"
        if amount_present_in_text(amount_raw, amount_value, source_span):
            care_source = source_line if source_line and clean_source_fragment(source_span).endswith("共") else source_span
            sentence = clean_labeled_source_sentence(care_source, label)
            if injury_text and not re.search(r"(受有|受傷|傷害)", sentence):
                sentence = f"{subject}{injury_text}，{sentence}"
            elif injury_text and re.match(r"^[0-9]+(?:\.[0-9]+)?月", sentence):
                sentence = f"{subject}{injury_text}，{sentence}"
            elif subject != "原告" and not sentence.startswith(subject) and claimant not in sentence[:80]:
                sentence = f"{subject}{sentence}"
            return ensure_sentence_punctuation(sentence)
        parts = [subject + injury_text]
        for value in [medical, period, care_need, calculation, basis]:
            if value and value not in parts:
                parts.append(value)
        return "，".join(parts) + f"，請求看護費用{amount}。"
    if label in {"工作損失", "勞動能力減損"}:
        if len(subclaims) > 1:
            subclaim_text = format_subclaim_details(subclaims)
            return f"{subject}{injury_text}，分別主張{subclaim_text}，合計{label}{amount}。"
        parts = []
        detail_values = [basis] if basis else [evidence, phrase_list(period), phrase_list(calculation)]
        for value in detail_values:
            value = clean_source_fragment(value).replace(")", "").replace("）", "")
            if value and value not in "，".join(parts):
                parts.append(ensure_work_loss_subject(value, subject, claimant))
        if not parts and should_include_injury_for_work_loss(source_line or source_span):
            parts.append(subject + injury_text)
        claim = clean_source_fragment(local_claim or f"請求{label}{amount}")
        base = "，".join(parts)
        if amount_present_in_text(amount_raw, amount_value, base):
            return ensure_sentence_punctuation(base)
        if base:
            return ensure_sentence_punctuation(f"{base}，{claim}")
        return ensure_sentence_punctuation(f"{subject}{claim}")
    if label == "精神慰撫金":
        return draft_mental_damage_sentence(subject, injury_text, medical, period, care_need, personal, amount, style_level)
    if label == "車輛修復費用":
        if len(subclaims) > 1:
            subclaim_text = format_subclaim_details(subclaims)
            return f"{format_damage_object_text(damage_object) or '車輛'}因本件事故受損，分別支出{subclaim_text}，合計車輛修復費用{amount}。"
        object_text = format_damage_object_text(damage_object) or "車輛"
        claim = ensure_claim_verb(clean_source_fragment(local_claim or f"修理費用{amount}"))
        tail = labeled_evidence_tail(label, evidence)
        reason = basis if basis and basis not in claim and not re.search(r"(證據|單據|收據|發票|為證|可證)", basis) else f"{object_text}因本件事故受損而有修理必要"
        reason = clean_vehicle_reason(reason, object_text)
        if claim.startswith(object_text) and re.search(r"(?:修繕|修理|修復|維修)費用", claim):
            return ensure_sentence_punctuation(f"{claim}{tail}")
        if amount_present_in_text(amount_raw, amount_value, reason):
            return ensure_sentence_punctuation(reason)
        if reason.startswith(object_text) or re.match(r"^(?:原告騎乘之)?(?:車號[^，。；\n]{0,40})?(?:系爭)?(?:汽車|機車|車輛)", reason):
            return f"{reason}，{claim}{tail}"
        return f"{object_text}因本件事故受損，{reason}，{claim}{tail}"
    if label == "財物損失":
        if len(subclaims) > 1:
            object_text = format_damage_object_text(damage_object) or "上開財物"
            subclaim_text = format_subclaim_details(subclaims)
            return f"{object_text}因本件事故毀損，分別支出{subclaim_text}，合計財物損失{amount}。"
        object_text = format_damage_object_text(damage_object) or "上開財物"
        tail = labeled_evidence_tail(label, evidence)
        claim = ensure_claim_verb(clean_source_fragment(local_claim or basis or f"財物損失{amount}"))
        return f"{object_text}因本件事故毀損，{claim}{tail}"
    if label == "其他必要費用":
        if len(subclaims) > 1:
            object_text = format_damage_object_text(damage_object) or "相關必要支出"
            subclaim_text = format_subclaim_details(subclaims)
            return f"{object_text}因本件事故受損或有支出必要，分別支出{subclaim_text}，合計其他必要費用{amount}。"
        object_text = format_damage_object_text(damage_object) or "相關必要支出"
        tail = labeled_evidence_tail(label, evidence)
        claim = ensure_claim_verb(clean_source_fragment(local_claim or basis or f"其他必要費用{amount}"))
        return f"{object_text}因本件事故受損或有支出必要，{claim}{tail}"
    return f"{subject}{injury_text}，{basis or label}，計{amount}。"


def format_subclaim_details(subclaims: list[dict]) -> str:
    details = []
    for subclaim in subclaims:
        amount_raw = subclaim.get("amount_raw", "")
        amount_value = subclaim.get("amount_value", normalize_amount_value(amount_raw))
        amount = f"{normalize_amount_display(amount_value)}元" if amount_value else f"{amount_raw}元"
        text = simplify_subclaim_text(subclaim.get("text", ""), amount_raw, amount_value)
        mentioned_amounts = {normalize_amount_value(match.group(1)) for match in re.finditer(AMOUNT_PATTERN, text)}
        if mentioned_amounts and amount_value not in mentioned_amounts:
            continue
        text = re.sub(r"^分別支出", "", text).strip(" ，。；、")
        if not text:
            text = amount
        elif not amount_present_in_text(amount_raw, amount_value, text):
            text = f"{text}{amount}"
        if text and text not in details:
            details.append(text)
    return join_chinese_phrases(details)


def format_traffic_subclaim_details(subclaims: list[dict]) -> str:
    details = []
    for subclaim in subclaims:
        amount_raw = subclaim.get("amount_raw", "")
        amount_value = subclaim.get("amount_value", normalize_amount_value(amount_raw))
        amount = f"{normalize_amount_display(amount_value)}元" if amount_value else f"{amount_raw}元"
        text = clean_traffic_reason(simplify_subclaim_text(subclaim.get("text", ""), amount_raw, amount_value), "原告")
        text = re.sub(r"^因本件事故後有交通往返(?:或代步)?需要，", "", text)
        if re.search(r"(受傷|傷害)", text) and not re.search(r"(交通|車資|計程車|往返|來回|代步|停車|過路|油資)", text):
            text = ""
        if not text:
            text = amount
        elif not amount_present_in_text(amount_raw, amount_value, text):
            text = f"{text}{amount}"
        if text and text not in details:
            details.append(text)
    return join_chinese_phrases(details)


def clean_traffic_reason(text: str, subject: str) -> str:
    text = clean_source_fragment(text)
    text = re.sub(r"^(?:原告)?因本件(?:事故|車禍)[^，。；\n]{0,80}受傷，?", "因本件事故後行動不便，", text)
    text = re.sub(
        rf"^(?:{re.escape(subject)})?因本件(?:事故|車禍)[^，。；\n]{{0,120}}?(?:等)?傷害[，,]?",
        "",
        text,
    )
    text = re.sub(
        r"^(?:原告)?[^，。；\n]{0,8}因本件(?:事故|車禍)[^，。；\n]{0,120}?(?:等)?傷害[，,]?",
        "",
        text,
    )
    text = re.sub(r"^因傷", "因本件事故後", text)
    text = re.sub(r"^且因本件車禍足部受傷，", "", text)
    text = re.sub(r"^對照其因系爭車禍事故[^，。；\n]{0,120}?(?:等)?傷害，?", "", text)
    text = re.sub(r"對照其因系爭車禍事故[，,]?受有[^，。；\n]{0,120}?(?:等)?傷害，?", "", text)
    text = re.sub(r"對照其因系爭車禍事故[^，。；\n]{0,120}?(?:等)?傷害，?", "", text)
    text = re.sub(r"[^，。；\n]{0,30}因(?:本件|系爭)車禍事故受傷至醫院治療，?", "", text)
    text = re.sub(r"[^，。；\n]{0,30}因上開傷害而不良於行，?", "因本件事故後行動不便，", text)
    text = re.sub(r"^(?:分別支出)?(?:每次|單趟|來回|往返)", r"因本件事故後往返", text)
    text = text.strip(" ，。；")
    if not text:
        return "因本件事故後有交通往返需要"
    if not text.startswith(("因", "原告", subject)):
        text = f"因本件事故後有交通往返需要，{text}"
    return text


def clean_vehicle_reason(text: str, object_text: str) -> str:
    text = clean_source_fragment(text)
    if re.search(AMOUNT_PATTERN + r"\s*[:：]", text):
        text = re.sub(r"^.*?" + AMOUNT_PATTERN + r"\s*[:：]\s*", "", text)
    text = re.sub(r"^(?:車號[^，。；\n]{0,40})?車輛修復費用\s*", "", text)
    text = re.sub(r"^(?:系爭汽車|系爭機車|汽車|機車|車輛)[；、](?:車輛|汽車|機車)因本件事故受損，", f"{object_text}因本件事故受損，", text)
    text = re.sub(rf"^{re.escape(object_text)}因本件事故受損，{re.escape(object_text)}因本件車禍受損，", f"{object_text}因本件車禍受損，", text)
    text = re.sub(r"^(機車|汽車|車輛)因本件事故受損，(原告騎乘之車號[^，。；\n]{0,40}\1因本件車禍受損，)", r"\2", text)
    if not text:
        text = f"{object_text}因本件事故受損而有修理必要"
    return text


def simplify_subclaim_text(text: str, amount_raw: str, amount_value: str) -> str:
    text = clean_source_fragment(text)
    if not text:
        return ""
    text = re.sub(r"([0-9][0-9,]*元)([0-9][0-9,]*元)", r"\1", text)
    amount_match = None
    for match in re.finditer(AMOUNT_PATTERN, text):
        if normalize_amount_value(match.group(1)) == amount_value:
            amount_match = match
            break
    if amount_match:
        start = max(
            text.rfind("、", 0, amount_match.start()),
            text.rfind("；", 0, amount_match.start()),
            text.rfind("，", 0, amount_match.start() - 12),
        ) + 1
        candidate = text[start:amount_match.end()]
        if "）" in candidate and "（" not in candidate:
            vehicle_start = max(
                text.rfind("系爭", 0, amount_match.start()),
                text.rfind("車輛", 0, amount_match.start()),
                text.rfind("汽車", 0, amount_match.start()),
                text.rfind("機車", 0, amount_match.start()),
            )
            if vehicle_start >= 0:
                start = vehicle_start
        text = text[start:amount_match.end()]
    text = re.sub(r"^(?:原告)?(?:主張)?(?:自系爭車禍發生)?(?:已)?支出", "", text)
    text = re.sub(r"^原告(?=至少|因有|需|另|前往|主張)", "", text)
    text = re.sub(r"^(?:以及|並且|且|故|另|另外)", "", text)
    return text.strip(" ，。；、")


def should_include_injury_for_work_loss(source: str) -> bool:
    return bool(re.search(r"(因傷|受傷|傷害|本件車禍所受傷害|本件事故所受傷害)", source or ""))


def clean_labeled_source_sentence(text: str, label: str) -> str:
    text = clean_source_fragment(text)
    label_heads = {
        "看護費用": ["看護費用", "看護費", "照護費用"],
        "醫療費用": ["醫療費用", "醫藥費用", "醫藥費", "醫療復健費用"],
    }.get(label, [label])
    for head in label_heads:
        text = re.sub(rf"^{re.escape(head)}\s*{AMOUNT_PATTERN}\s*[:：]\s*", "", text)
        text = re.sub(rf"^{re.escape(head)}\s*{AMOUNT_PATTERN}\s+", "", text)
        text = re.sub(rf"^{re.escape(head)}\s+", "", text)
    return text


def ensure_work_loss_subject(text: str, subject: str, claimant: str) -> str:
    text = clean_source_fragment(text)
    text = re.sub(r"^之薪資損失\s+(?=按|原告|於|依|自|因)", "", text)
    text = re.sub(r"^原告之薪資損失\s+(?=按|原告|於|依|自|因)", "", text)
    text = re.sub(r"^(?:原告(?:之)?)?(?:工作損失|薪資損失|不能工作之損失|無法工作損失|收入損失)\s+(?=按|原告|於|依|自|因)", "", text)
    if not text:
        return ""
    if claimant and claimant not in {"原告", "未特定原告", "原告2人"}:
        pos = text.find(f"原告{claimant}")
        if pos > 0:
            text = text[pos:]
        elif pos < 0:
            bare_pos = text.find(claimant)
            if bare_pos > 0:
                text = text[bare_pos:]
    if subject != "原告" and text.startswith(subject):
        return text
    if claimant and claimant not in {"原告", "未特定原告", "原告2人"} and text.startswith(f"原告{claimant}"):
        return text
    if text.startswith("原告") and subject != "原告":
        return re.sub(r"^原告", subject, text, count=1)
    if text.startswith(("原告", "因", "依", "以", "參以", "自")):
        return text
    if claimant and claimant not in {"原告", "未特定原告", "原告2人"} and text.startswith(claimant):
        return f"原告{text}"
    return f"{subject}{text}"


def extract_party_names(raw: str) -> list[str]:
    raw = (raw or "").replace("被告", "")
    return split_plaintiff_names(raw)


def extract_scoped_damage_items(comp_facts: str, plaintiff_names: list[str], context_text: str = "") -> list[dict]:
    comp_facts = unicodedata.normalize("NFC", comp_facts or "")
    context_text = unicodedata.normalize("NFC", context_text or "")
    items = []
    seen: set[tuple[str, str, str, str]] = set()
    current_plaintiff = ""
    pending: list[str] = []

    for sentence in split_damage_source_sentences(comp_facts):
        sentence = unicodedata.normalize("NFC", sentence)
        mentioned = mentioned_plaintiffs(sentence, plaintiff_names)
        if "原告2人" in sentence or "原告二人" in sentence or "原告二名" in sentence:
            current_plaintiff = "原告2人"
            pending = [sentence]
        elif mentioned:
            if current_plaintiff == mentioned[0]:
                pending.append(sentence)
                pending = pending[-4:]
            else:
                current_plaintiff = mentioned[0]
                pending = [sentence]
        elif current_plaintiff:
            pending.append(sentence)
            pending = pending[-4:]

        for amount_match in re.finditer(AMOUNT_PATTERN, sentence):
            if is_rate_or_reference_amount(sentence, amount_match):
                continue
            amount_raw = amount_match.group(1)
            amount_value = normalize_amount_value(amount_raw)
            amount_display = normalize_amount_display(amount_raw)
            label = infer_damage_label_for_amount(sentence, amount_match)
            if label == "一般損害項目":
                continue
            for label in [label]:
                source_span = select_relevant_source_span(pending, label) if pending else sentence
                plaintiff = infer_plaintiff_for_amount(sentence, amount_match, plaintiff_names)
                if not plaintiff:
                    plaintiff = infer_plaintiff_from_text(source_span, plaintiff_names)
                if not plaintiff and label == "車輛修復費用":
                    plaintiff = infer_vehicle_claimant_from_context(source_span, context_text or comp_facts, plaintiff_names)
                if not plaintiff and label != "車輛修復費用":
                    plaintiff = current_plaintiff
                if not plaintiff:
                    plaintiff = "原告"
                amount_context = extract_amount_context(sentence, amount_match)
                if label == "工作損失":
                    amount_context = extract_work_loss_amount_context(sentence, amount_match) or amount_context
                amount_window = sentence[max(0, amount_match.start() - 12):amount_match.end() + 4]
                each_claim = "各" in f"{amount_context}{amount_window}"
                specific_plaintiffs = [name for name in plaintiff_names if is_specific_plaintiff_name(name)]
                if each_claim and len(mentioned) > 1:
                    target_plaintiffs = mentioned
                elif each_claim and len(specific_plaintiffs) > 1 and re.search(r"原告(?:兩人|二人|2人|各)", sentence):
                    target_plaintiffs = specific_plaintiffs
                else:
                    target_plaintiffs = [plaintiff]
                for target_plaintiff in target_plaintiffs:
                    key = (target_plaintiff, label, amount_value, normalize_required_source_key(amount_context))
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append({
                        "plaintiff": target_plaintiff,
                        "label": label,
                        "amount_raw": amount_display,
                        "amount_value": amount_value,
                        "amount_context": amount_context,
                        "source_line": sentence,
                        "source_span": amount_context if label == "工作損失" else source_span,
                    })
                    if is_specific_plaintiff_name(target_plaintiff):
                        current_plaintiff = target_plaintiff
    return items


def extract_work_loss_amount_context(sentence: str, amount_match: re.Match[str]) -> str:
    left = max(
        sentence.rfind("。", 0, amount_match.start()),
        sentence.rfind("此外", 0, amount_match.start()),
        sentence.rfind("另外", 0, amount_match.start()),
        sentence.rfind("另", 0, amount_match.start()),
    )
    start = left + (2 if left >= 0 and sentence.startswith(("此外", "另外"), left) else 1)
    context = sentence[start:amount_match.end()].strip(" ，。；;")
    if len(context) > 180:
        context = context[-180:]
    return clean_source_fragment(context)


def extract_amount_context(sentence: str, amount_match: re.Match[str]) -> str:
    start, end = amount_match.span()
    subitem_markers = ["⑴", "⑵", "⑶", "⑷", "⑸", "⑹", "⑺", "⑻", "⑼"]
    subitem_left = max([sentence.rfind(marker, 0, start) for marker in subitem_markers] + [-1])
    left = max(
        sentence.rfind("。", 0, start),
        sentence.rfind("；", 0, start),
        sentence.rfind(";", 0, start),
        sentence.rfind("、", 0, start),
        sentence.rfind("，", 0, start),
        sentence.rfind("\n", 0, start),
        subitem_left,
    ) + 1
    right_candidates = [
        pos for pos in [
            sentence.find("。", end),
            sentence.find("；", end),
            sentence.find(";", end),
            sentence.find("、", end),
            sentence.find("，", end),
            sentence.find("\n", end),
        ] if pos >= 0
    ]
    right_candidates.extend(
        pos for pos in [sentence.find(marker, end) for marker in subitem_markers] if pos >= 0
    )
    right = min(right_candidates) if right_candidates else len(sentence)
    context = sentence[left:right].strip(" ，。；;")
    if len(context) > 120:
        relative_amount_start = max(0, start - left)
        short_left = max(
            context.rfind("。", 0, relative_amount_start),
            context.rfind("；", 0, relative_amount_start),
            context.rfind("、", 0, relative_amount_start),
            context.rfind("，", 0, max(0, relative_amount_start - 18)),
        )
        if short_left >= 0:
            context = context[short_left + 1:].strip(" ，。；;")
    if "）" in context and "（" not in context:
        vehicle_start = max(
            sentence.rfind("系爭", 0, start),
            sentence.rfind("車輛", 0, start),
            sentence.rfind("汽車", 0, start),
            sentence.rfind("機車", 0, start),
        )
        if vehicle_start >= 0:
            context = sentence[vehicle_start:right].strip(" ，。；;")
    return clean_source_fragment(context)


def mentioned_plaintiffs(text: str, plaintiff_names: list[str]) -> list[str]:
    text = unicodedata.normalize("NFC", text or "")
    positions = []
    for name in sorted(plaintiff_names, key=len, reverse=True):
        if not is_specific_plaintiff_name(name):
            continue
        pos = text.find(name)
        if pos >= 0:
            positions.append((pos, -len(name), name))
    return [name for _, _, name in sorted(positions)]


def infer_plaintiff_from_text(text: str, plaintiff_names: list[str]) -> str:
    mentioned = mentioned_plaintiffs(text, plaintiff_names)
    return mentioned[0] if mentioned else ""


def infer_plaintiff_for_amount(sentence: str, amount_match: re.Match[str], plaintiff_names: list[str]) -> str:
    prefix = unicodedata.normalize("NFC", sentence[:amount_match.start()])
    positions = [(prefix.rfind(name), len(name), name) for name in plaintiff_names if prefix.rfind(name) >= 0]
    if not positions:
        return ""
    return max(positions, key=lambda item: (item[0], item[1]))[2]


def infer_vehicle_claimant_from_context(source_span: str, context_text: str, plaintiff_names: list[str]) -> str:
    vehicle_terms = extract_damage_object_terms(source_span)
    vehicle_aliases = [term for term in vehicle_terms.split("；") if term.startswith("系爭")]
    if not vehicle_aliases and "系爭汽車" in source_span:
        vehicle_aliases = ["系爭汽車"]
    if not vehicle_aliases and "系爭機車" in source_span:
        vehicle_aliases = ["系爭機車"]
    if not vehicle_aliases:
        vehicle_aliases = ["系爭汽車", "系爭機車", "車輛", "汽車", "機車"]

    candidates = []
    for name in plaintiff_names:
        name_patterns = [
            rf"(?:原告)?{re.escape(name)}[^。；\n]{{0,45}}(?:所有|駕駛|騎乘)[^。；\n]{{0,45}}(?:{'|'.join(map(re.escape, vehicle_aliases))}|車牌號碼|自用小客車|普通重型機車)",
            rf"(?:{'|'.join(map(re.escape, vehicle_aliases))})[^。；\n]{{0,45}}(?:為|係|屬於)(?:原告)?{re.escape(name)}[^。；\n]{{0,20}}(?:所有|駕駛|使用)",
            rf"(?:原告)?{re.escape(name)}[^。；\n]{{0,45}}所駕駛[^。；\n]{{0,20}}(?:{'|'.join(map(re.escape, vehicle_aliases))})",
            rf"(?:原告)?{re.escape(name)}駕駛車牌號碼[^。；\n]{{0,45}}(?:{'|'.join(map(re.escape, vehicle_aliases))}|自用小客車|普通重型機車)",
        ]
        if any(re.search(pattern, context_text) for pattern in name_patterns):
            candidates.append(name)

    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else ""


def select_relevant_source_span(sentences: list[str], label: str) -> str:
    selected = []
    for sentence in reversed(sentences):
        labels = infer_damage_labels(sentence)
        has_other_amount = bool(re.search(AMOUNT_PATTERN, sentence)) and label not in labels
        has_other_label = labels != ["一般損害項目"] and label not in labels
        if selected and (has_other_amount or has_other_label):
            break
        if label in labels or labels == ["一般損害項目"] or not selected:
            selected.append(sentence)
    selected.reverse()
    return "；".join(selected)


def clean_damage_section(text: str, constraints: dict | None = None) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        if should_drop_damage_line(line):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = strip_case_borrowing_markers(cleaned)
    if constraints is not None:
        cleaned = remove_unsupported_sensitive_lines(cleaned, constraints)
        cleaned = remove_empty_damage_blocks(cleaned)
        cleaned = ensure_required_damage_items(cleaned, constraints["required_items"])
        cleaned = merge_repeated_damage_titles(cleaned)
        cleaned = remove_duplicate_damage_blocks(cleaned)
        cleaned = enrich_plaintiff_injury_mentions(cleaned, constraints.get("plaintiff_injuries", {}))
        cleaned = enrich_grouped_section_claimants(cleaned, constraints.get("plaintiff_injuries", {}))
        cleaned = renumber_flat_damage_blocks(cleaned)
    cleaned = remove_repeated_injury_leadins(cleaned)
    cleaned = dedupe_sentences_within_damage_blocks(cleaned)
    cleaned = remove_duplicate_total_phrases(cleaned)
    return cleaned


def remove_repeated_injury_leadins(text: str) -> str:
    pattern = re.compile(
        r"(原告(?:[" + NAME_CHARS + r"]{1,5})?因本件事故受有(?P<inj>[^，。；\n]{3,80}?)(?:等)?傷害)，"
        r"\s*原告(?:[" + NAME_CHARS + r"]{1,5})?因本件事故受有(?P=inj)(?:之)?傷害，"
    )
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"\1，", text)
    return text


def dedupe_sentences_within_damage_blocks(text: str) -> str:
    block_pattern = re.compile(
        r"(?ms)(^（[一二三四五六七八九十0-9]+）[^\n]+\n)(.*?)(?=^（[一二三四五六七八九十0-9]+）|^[一二三四五六七八九十0-9]+、|\Z)"
    )

    def replace_block(match: re.Match[str]) -> str:
        heading, body = match.group(1), match.group(2)
        sentences = re.findall(r".*?[。！？](?:\s*|$)", body, flags=re.S)
        if not sentences:
            return match.group(0)
        seen = set()
        kept = []
        for sentence in sentences:
            compact = re.sub(r"\s+", "", sentence)
            if len(compact) > 20 and compact in seen:
                continue
            seen.add(compact)
            kept.append(sentence.strip())
        trailing = body
        for sentence in sentences:
            trailing = trailing.replace(sentence, "", 1)
        tail = trailing.strip()
        if tail:
            kept.append(tail)
        joiner = "" if "精神慰撫金" in heading else "\n"
        return heading + joiner.join(part for part in kept if part).strip() + "\n\n"

    return block_pattern.sub(replace_block, text)


def should_drop_damage_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.startswith(("綜上所述", "綜上所陳")):
        return True
    if any(token in line for token in FORBIDDEN_DAMAGE_LINES):
        return True
    if re.match(r"^(?:總計|合計|共計)[:：]?[0-9,]+元[。.]?$", line):
        return True
    if re.match(r"^請求被告賠償[0-9,]+元[。.]?$", line):
        return True
    return False


def merge_repeated_damage_titles(text: str) -> str:
    section_pattern = r"(?m)^[一二三四五六七八九十0-9]+、(?:原告[^\n]*部分|未特定[^\n]*|共同[^\n]*)$"
    matches = list(re.finditer(section_pattern, text))
    if not matches:
        return merge_repeated_titles_in_flat_section(text)

    prefix = text[:matches[0].start()].strip()
    sections = []
    for idx, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(0).rstrip()
        body = text[body_start:body_end].strip()
        merged_body = merge_repeated_titles_in_flat_section(body)
        section = heading
        if merged_body:
            section += f"\n\n{merged_body}"
        sections.append(section)
    output = []
    if prefix:
        output.append(prefix)
    output.extend(sections)
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(output)).strip()


def merge_repeated_titles_in_flat_section(text: str) -> str:
    raw_blocks = [b for b in re.split(r"\n\n(?=（[一二三四五六七八九十\d]+）)", text.strip()) if b.strip()]
    if len(raw_blocks) <= 1:
        return text
    merged: list[dict] = []
    index_by_title: dict[str, int] = {}
    prefix_blocks = []
    for block in raw_blocks:
        match = re.match(r"^（[一二三四五六七八九十\d]+）([^\n]+)\n?(.*)$", block, re.S)
        if not match:
            prefix_blocks.append(block)
            continue
        title = match.group(1).strip()
        body = match.group(2).strip()
        if title not in index_by_title:
            index_by_title[title] = len(merged)
            merged.append({"title": title, "bodies": [body] if body else []})
            continue
        target = merged[index_by_title[title]]
        if body and not any(body == old or body in old for old in target["bodies"]):
            target["bodies"] = [old for old in target["bodies"] if old not in body]
            target["bodies"].append(body)

    output = prefix_blocks[:]
    for idx, item in enumerate(merged, start=1):
        body = "\n".join(item["bodies"]).strip()
        block = f"{to_chinese_item_marker(idx)}{item['title']}"
        if body:
            block += f"\n{body}"
        output.append(block)
    return "\n\n".join(output).strip()


def remove_empty_damage_blocks(text: str) -> str:
    blocks = [b for b in re.split(r"\n\n(?=（[一二三四五六七八九十\d]+）)", text.strip()) if b.strip()]
    if not blocks:
        return text
    kept = []
    for block in blocks:
        title_match = re.match(r"^(（[一二三四五六七八九十\d]+）[^\n]+)\n?(.*)$", block, re.S)
        if title_match and not title_match.group(2).strip():
            continue
        kept.append(block)
    return "\n\n".join(kept)


def strip_case_borrowing_markers(text: str) -> str:
    text = re.sub(r"^相似案例.*$", "", text, flags=re.M)
    text = re.sub(r"^案例\s*\d+.*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_unsupported_sensitive_lines(text: str, constraints: dict) -> str:
    kept_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append(raw_line)
            continue
        if line_mentions_unsupported_amount(line, constraints["allowed_amounts"]):
            continue
        if line_mentions_unsupported_hospital(line, constraints["allowed_hospitals"]):
            continue
        if line_mentions_unsupported_income(line, constraints["allowed_income_terms"]):
            continue
        kept_lines.append(raw_line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def ensure_required_damage_items(text: str, required_items: list[dict]) -> str:
    if not required_items:
        return text
    blocks = [b for b in re.split(r"\n\n(?=（[一二三四五六七八九十\d]+）)", text.strip()) if b.strip()]
    next_idx = count_damage_items(text) + 1
    appended_blocks = []

    for item in required_items:
        if required_item_present_in_text(item, text):
            continue
        if same_label_amount_block_exists(item, blocks):
            continue
        merged = False
        for idx, block in enumerate(blocks):
            if block_can_absorb_required_item(block, item):
                blocks[idx] = block.rstrip() + "\n" + required_item_sentence(item)
                merged = True
                break
        if not merged:
            prefix = to_chinese_item_marker(next_idx)
            appended_blocks.append(f"{prefix}{item['label']}\n{required_item_sentence(item)}")
            next_idx += 1
    merged_text = "\n\n".join(blocks).strip()
    if not appended_blocks:
        return merged_text
    joiner = "\n\n" if merged_text else ""
    return f"{merged_text}{joiner}" + "\n\n".join(appended_blocks)


def same_label_amount_block_exists(item: dict, blocks: list[str]) -> bool:
    label = item["label"]
    for block in blocks:
        if damage_label_matches_text(label, block) and re.search(AMOUNT_PATTERN, block):
            return True
    return False


def remove_duplicate_damage_blocks(text: str) -> str:
    section_pattern = r"(?m)^[一二三四五六七八九十0-9]+、(?:原告[^\n]*部分|未特定[^\n]*|共同[^\n]*)$"
    matches = list(re.finditer(section_pattern, text))
    if matches:
        sections = []
        prefix = text[:matches[0].start()].strip()
        for idx, match in enumerate(matches):
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            heading = match.group(0).rstrip()
            body = remove_duplicate_damage_blocks_flat(text[body_start:body_end].strip())
            sections.append(f"{heading}\n\n{body}" if body else heading)
        output = []
        if prefix:
            output.append(prefix)
        output.extend(sections)
        return renumber_grouped_damage_blocks("\n\n".join(output).strip())
    return remove_duplicate_damage_blocks_flat(text)


def remove_duplicate_damage_blocks_flat(text: str) -> str:
    blocks = [b for b in re.split(r"\n\n(?=（[一二三四五六七八九十\d]+）)", text.strip()) if b.strip()]
    if len(blocks) <= 1:
        return text
    seen = set()
    kept = []
    for block in blocks:
        title_match = re.match(r"^(（[一二三四五六七八九十\d]+）)([^\n]+)", block)
        if not title_match:
            kept.append(block)
            continue
        title = title_match.group(2).strip()
        amount = extract_claim_amount_from_block(block)
        body_key = re.sub(r"^（[一二三四五六七八九十\d]+）", "", block)
        body_key = re.sub(r"\s+", "", body_key)
        key = (title, amount, body_key)
        if key in seen:
            continue
        seen.add(key)
        kept.append(block)
    cleaned = "\n\n".join(kept)
    return renumber_flat_damage_blocks(cleaned)


def renumber_flat_damage_blocks(text: str) -> str:
    if re.search(r"(?m)^[一二三四五六七八九十0-9]+、原告", text):
        return renumber_grouped_damage_blocks(text)
    index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal index
        index += 1
        return f"{to_chinese_item_marker(index)}{match.group(1)}"

    return re.sub(r"(?m)^（[一二三四五六七八九十\d]+）([^\n]+)", replace, text)


def renumber_grouped_damage_blocks(text: str) -> str:
    section_pattern = r"(?m)^[一二三四五六七八九十0-9]+、(?:原告[^\n]*部分|未特定或共同財產損害)\n?"
    matches = list(re.finditer(section_pattern, text))
    if not matches:
        return text
    rebuilt = []
    cursor = 0
    for idx, match in enumerate(matches):
        rebuilt.append(text[cursor:match.start()])
        heading = match.group(0)
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        item_index = 0

        def replace_item(item_match: re.Match[str]) -> str:
            nonlocal item_index
            item_index += 1
            return f"{to_chinese_item_marker(item_index)}{item_match.group(1)}"

        body = re.sub(r"(?m)^（[一二三四五六七八九十\d]+）([^\n]+)", replace_item, body)
        rebuilt.append(heading)
        rebuilt.append(body)
        cursor = body_end
    rebuilt.append(text[cursor:])
    return "".join(rebuilt)


def required_item_present_in_text(item: dict, text: str) -> bool:
    blocks = [b for b in re.split(r"\n\n(?=（[一二三四五六七八九十\d]+）)", text.strip()) if b.strip()]
    amount_raw = item["amount_raw"]
    amount_value = item["amount_value"]
    label = item["label"]
    for block in blocks:
        if amount_present_in_text(amount_raw, amount_value, block) and damage_label_matches_text(label, block):
            return True
    return False


def damage_label_matches_text(label: str, text: str) -> bool:
    rules = {
        "醫療費用": ["醫療", "醫藥", "診療", "復健"],
        "交通費用": ["交通", "車資", "往返"],
        "工作損失": ["工作", "收入損失", "薪資損失", "不能工作", "無法工作"],
        "看護費用": ["看護", "照護", "幫傭"],
        "車輛修復費用": ["車輛", "修復", "修理", "維修", "機車", "汽車"],
        "精神慰撫金": ["精神", "慰撫"],
        "財物損失": ["財物", "手機", "眼鏡", "安全帽", "鞋子", "衣服", "褲子", "手錶"],
        "其他必要費用": ["其他", "營養品", "護具", "輔具", "尿布", "便器", "用品"],
    }
    return any(token in text for token in rules.get(label, [label]))


def count_damage_items(text: str) -> int:
    return len(re.findall(r"（[一二三四五六七八九十]+）", text))


def to_chinese_item_marker(index: int) -> str:
    numerals = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    if 1 <= index <= 10:
        return f"（{numerals[index]}）"
    return f"（{index}）"


def block_can_absorb_required_item(block: str, item: dict) -> bool:
    label = item["label"]
    normalized_block = block.replace(" ", "")
    if "醫療" in label and "醫療" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "交通" in label and "交通" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "工作" in label and "工作" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "修復" in label and "修復" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "慰撫" in label and "慰撫" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "財物" in label and any(token in normalized_block for token in ["財物", "手機", "眼鏡", "安全帽", "鞋子", "衣服", "褲子", "手錶"]) and not re.search(AMOUNT_PATTERN, block):
        return True
    if "看護" in label and "看護" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    if "其他" in label and "其他" in normalized_block and not re.search(AMOUNT_PATTERN, block):
        return True
    return False


def required_item_sentence(item: dict) -> str:
    source = item.get("source_line", "").strip()
    label = item["label"]
    amount = item["amount_raw"]
    if source:
        return ensure_sentence_punctuation(source)
    return f"因本次事故，產生{label}新台幣{amount}元。"


def amount_present_in_text(amount_raw: str, amount_value: str, text: str) -> bool:
    normalized_text = normalize_amount_text(text).replace(",", "").replace("臺", "台")
    variants = {amount_raw, amount_value}
    if amount_value.isdigit():
        variants.add(f"{int(amount_value):,}")
        if int(amount_value) % 10000 == 0 and int(amount_value) >= 10000:
            variants.add(f"{int(amount_value) // 10000}萬")
    for variant in variants:
        normalized_variant = variant.replace(",", "").replace("臺", "台")
        if f"{normalized_variant}元" in normalized_text:
            return True
    return False


def ensure_sentence_punctuation(text: str) -> str:
    text = clean_source_fragment(text)
    if not text:
        return ""
    if text.count("（") > text.count("）"):
        text += "）"
    if text.count("(") > text.count(")"):
        text += ")"
    return f"{text}。"


def enrich_plaintiff_injury_mentions(text: str, plaintiff_injuries: dict[str, str]) -> str:
    for name, injury in plaintiff_injuries.items():
        concrete = f"原告{name}因本件事故受有{injury}"
        text = re.sub(
            rf"原告主張因本件事故受傷，致原告{re.escape(name)}",
            concrete,
            text,
        )
        text = re.sub(
            rf"原告主張因本件車禍受傷，致原告{re.escape(name)}",
            concrete,
            text,
        )
        text = text.replace(f"原告{name}因本件事故受傷，", f"{concrete}，")
        text = text.replace(f"原告{name}因本件事故受傷。", f"{concrete}。")
        text = text.replace(f"原告{name}因本件車禍受傷，", f"{concrete}，")
        text = text.replace(f"原告{name}因本件車禍受傷。", f"{concrete}。")
    return text


def enrich_grouped_section_claimants(text: str, plaintiff_injuries: dict[str, str]) -> str:
    if not plaintiff_injuries:
        return text
    section_pattern = rf"([一二三四五六七八九十]+、原告([{NAME_CHARS}]{{2,5}})部分\n.*?)(?=\n[一二三四五六七八九十]+、原告|\n[一二三四五六七八九十]+、未特定|\Z)"

    def replace_section(match: re.Match[str]) -> str:
        block = match.group(1)
        name = match.group(2)
        injury = plaintiff_injuries.get(name)
        if not injury:
            return block
        concrete = f"原告{name}因本件事故受有{injury}"
        replacements = [
            (r"原告主張因本件事故受傷，致原告", concrete + "，"),
            (r"原告主張因本件車禍受傷，致原告", concrete + "，"),
            (r"原告因本件事故受傷", concrete),
            (r"原告因本件車禍受傷", concrete),
            (r"原告因本件事故受有", f"原告{name}因本件事故受有"),
            (r"原告因本件車禍受有", f"原告{name}因本件事故受有"),
            (r"末查原告", f"末查原告{name}"),
            (r"原告需", f"原告{name}需"),
            (r"原告工作損失", f"原告{name}工作損失"),
            (r"原告確實", f"原告{name}確實"),
            (r"原告支出", f"原告{name}支出"),
        ]
        for pattern, repl in replacements:
            block = re.sub(pattern, repl, block)
        return block

    return re.sub(section_pattern, replace_section, text, flags=re.S)


def line_mentions_unsupported_amount(line: str, allowed_amounts: set[str]) -> bool:
    if is_damage_heading_line(line):
        return False
    found = [normalize_amount_value(match.group(1)) for match in re.finditer(AMOUNT_PATTERN, line)]
    return bool(found) and any(amount not in allowed_amounts for amount in found)


def line_mentions_unsupported_hospital(line: str, allowed_hospitals: set[str]) -> bool:
    if is_damage_heading_line(line):
        return False
    if not re.search(r"(醫院|診所|中醫|紀念醫院)", line):
        return False
    if not allowed_hospitals:
        return True
    return not any(term in line or has_substantial_text_overlap(line, term) for term in allowed_hospitals)


def has_substantial_text_overlap(text: str, reference: str) -> bool:
    text = re.sub(r"\s+", "", text or "")
    reference = re.sub(r"\s+", "", reference or "")
    stop_tokens = {"醫院", "診所", "中醫", "紀念", "原告", "醫療", "費用", "證明", "收據"}
    for size in range(min(8, len(reference)), 3, -1):
        for idx in range(0, len(reference) - size + 1):
            token = reference[idx:idx + size]
            if any(stop in token for stop in stop_tokens):
                continue
            if re.search(r"\d", token):
                continue
            if token in text:
                return True
    return False


def line_mentions_unsupported_income(line: str, allowed_income_terms: set[str]) -> bool:
    if is_damage_heading_line(line):
        return False
    if re.search(r"(工作損失|不能工作|無法工作|收入損失|薪資損失)", line):
        return False
    if not re.search(r"(日薪|月薪|年收入|所得|從事|任職|工作)", line):
        return False
    if not allowed_income_terms:
        return True
    return not any(term in line for term in allowed_income_terms)


def is_damage_heading_line(line: str) -> bool:
    return bool(re.match(r"^（[一二三四五六七八九十0-9]+）[^，。；]{1,30}$", line.strip()))


def compute_total_from_damage_section(damage_section: str) -> int:
    blocks = [b.strip() for b in re.split(r"\n(?=（[一二三四五六七八九十]+）)", damage_section) if b.strip()]
    totals = []
    for block in blocks:
        title_match = re.match(r"^（[一二三四五六七八九十0-9]+）([^\n]+)", block)
        title = title_match.group(1).strip() if title_match else ""
        amount = extract_claim_amount_from_block(block, title)
        if amount is not None:
            totals.append(amount)
    return sum(totals)


def extract_claim_amount_from_block(block: str, title: str = "", claimant: str = "") -> int | None:
    label_amount = extract_claim_amount_by_title(block, title, claimant)
    if label_amount is not None:
        return label_amount
    preferred_patterns = [
        rf"(?:請求|爰請求)[^\n，。；]*?{AMOUNT_PATTERN}",
        rf"(?:共計|合計|總計)[^\n，。；]*?{AMOUNT_PATTERN}",
        rf"(?:支出)[^\n，。；]*?{AMOUNT_PATTERN}",
        rf"(?:費用|損失|慰撫金)[^\n，。；：:]*?(?:為|計|共計|合計)?[^\n，。；]*?{AMOUNT_PATTERN}",
    ]
    for pattern in preferred_patterns:
        match = re.search(pattern, block)
        if match:
            return safe_parse_amount(match.group(1))

    claim_amounts = []
    for amount_match in re.finditer(AMOUNT_PATTERN, block):
        if is_rate_or_reference_amount(block, amount_match):
            continue
        amount = safe_parse_amount(amount_match.group(1))
        if amount is not None:
            claim_amounts.append(amount)
    if claim_amounts:
        return claim_amounts[-1]
    return None


def extract_claim_amount_by_title(block: str, title: str, claimant: str = "") -> int | None:
    if not title:
        return None
    keyword_rules = [
        ("醫療費用", ["醫療費用", "醫藥費", "醫療費", "診療費", "住院費", "門診費", "醫療用品費"]),
        ("交通費用", ["交通費用", "交通費", "計程車費", "車資", "增加生活費負擔", "生活費負擔"]),
        ("看護費用", ["看護費用", "看護費", "照護費", "照顧費"]),
        ("工作損失", ["工作損失", "收入損失", "薪資損失", "不能工作之損失"]),
        ("勞動能力減損", ["勞動能力損失", "減少勞動能力", "勞動能力減損"]),
        ("車輛修復費用", ["車輛修復費", "修復費", "修理費", "維修費", "修車費"]),
        ("精神慰撫金", ["精神慰撫金", "慰撫金", "精神賠償"]),
        ("財物損失", ["財物損失", "手機", "眼鏡", "安全帽", "鞋子", "衣服", "褲子", "手錶"]),
        ("其他必要費用", ["增加生活必要費用", "生活必要費用", "醫療用品費", "用品費", "器材費", "輔具費", "營養品", "護具"]),
    ]
    title_labels = [label for label, keywords in keyword_rules if label in title or any(keyword in title for keyword in keywords)]
    if not title_labels:
        return None
    keywords = []
    for label, rule_keywords in keyword_rules:
        if label in title_labels:
            keywords.extend(rule_keywords)

    candidates: list[tuple[int, int]] = []
    body_start = block.find("\n") + 1 if "\n" in block else 0
    for amount_match in re.finditer(AMOUNT_PATTERN, block):
        if is_rate_or_reference_amount(block, amount_match):
            continue
        prefix = block[max(body_start, amount_match.start() - 36):amount_match.start()]
        local = block[max(0, amount_match.start() - 30):amount_match.end()]
        if any(keyword in prefix for keyword in keywords):
            amount = safe_parse_amount(amount_match.group(1))
            if amount is not None:
                claimant_window = block[max(body_start, amount_match.start() - 90):amount_match.start()]
                keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
                if claimant and re.search(rf"(?:原告)?{re.escape(claimant)}[^，。；\n]{{0,30}}(?:{keyword_pattern})", prefix):
                    claimant_score = 20
                elif claimant and claimant in claimant_window:
                    claimant_score = 1
                else:
                    claimant_score = 0
                label_score = 3 if re.search(r"(合計|共計|總計|總共|總共|增加生活費負擔|生活費負擔)", prefix) else 2 if re.search(r"(請求|爰請求|支出|受有)", local) else 1
                reference_penalty = 20 if re.search(r"(行情|每日|每月|每次|計算式|僅|未超出)", prefix + local) else 0
                score = claimant_score + label_score - reference_penalty
                candidates.append((score, amount))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def safe_parse_amount(raw: str) -> int | None:
    try:
        return int(normalize_amount_value(raw))
    except ValueError:
        return None


def build_conclusion_section(damage_section: str, parties: dict, style_level: int = 0) -> str:
    total_amount = compute_total_from_damage_section(damage_section)
    next_marker = conclusion_section_marker(damage_section)
    detailed_summary = build_conclusion_damage_summary(damage_section, parties, total_amount)
    if detailed_summary:
        return f"{next_marker}結論：{polish_conclusion_by_style(detailed_summary, style_level)}"

    plaintiff = parties.get("原告", "原告")
    defendant = format_party_with_role(parties.get("被告", "被告"), "被告")
    liability = "應連帶賠償" if has_multiple_parties(parties.get("被告", "")) else "應賠償"
    text = (
        f"{next_marker}結論：綜上所陳，{defendant}因前揭侵權行為，{liability}{plaintiff}所受之各項損害，"
        f"合計新台幣{total_amount:,}元，並自起訴狀繕本送達翌日起至清償日止，按年息5%計算之利息。"
    )
    return polish_conclusion_by_style(text, style_level)


def polish_conclusion_by_style(text: str, style_level: int) -> str:
    if style_level <= 0:
        return text
    if style_level == 1:
        return text.replace("所受之各項損害", "所受前述各項損害").replace("原告之損害", "原告前述損害")
    if style_level == 2:
        return text.replace("所受之各項損害", "所受前述各項損害").replace("原告之損害", "原告前述各項損害")
    if style_level == 4:
        return text.replace("所受之各項損害", "所受前述各項損害").replace("原告之損害", "原告前述各項損害").replace("前述逐項列示之損害", "前述各項損害")
    if style_level >= 5:
        return text.replace("所受之各項損害", "所受前述損害").replace("原告之損害", "原告前述損害").replace("前述逐項列示之損害", "前述損害")
    return text.replace("所受之各項損害", "所受前述逐項列示之損害").replace("原告之損害", "原告前述逐項列示之損害")


def conclusion_section_marker(damage_section: str) -> str:
    grouped_sections = re.findall(r"(?m)^[一二三四五六七八九十0-9]+、(?:原告|未特定|共同)", damage_section)
    if not grouped_sections:
        return "三、"
    return to_chinese_section_marker(max(3, len(grouped_sections) + 1))


def build_conclusion_damage_summary(damage_section: str, parties: dict, total_amount: int) -> str:
    groups = extract_conclusion_damage_groups(damage_section, parties)
    if not groups:
        return ""

    defendant = format_party_with_role(parties.get("被告", "被告"), "被告")
    liability = "應連帶賠償" if has_multiple_parties(parties.get("被告", "")) else "應賠償"
    group_sentences = []
    named_group_count = 0
    grouped_total_amount = 0
    for group in groups:
        items = group["items"]
        if not items:
            continue
        subtotal = sum(item["amount"] for item in items)
        grouped_total_amount += subtotal
        item_text = join_chinese_phrases([
            f"{item['label']}{item['amount']:,}元" for item in items
        ])
        claimant = group["claimant"]
        if claimant:
            named_group_count += 1
            group_sentences.append(f"原告{claimant}{item_text}，合計{subtotal:,}元")
        else:
            group_sentences.append(f"{item_text}，合計{subtotal:,}元")

    if not group_sentences:
        return ""
    if grouped_total_amount > 0:
        total_amount = grouped_total_amount
    if len(group_sentences) == 1:
        return (
            f"綜上所陳，{defendant}因前揭侵權行為，{liability}原告之損害，"
            f"包含{group_sentences[0]}，並自起訴狀繕本送達翌日起至清償日止，按年息5%計算之利息。"
        )

    total_subject = f"{chinese_count_prefix(named_group_count)}原告損害" if named_group_count else "原告損害"
    return (
        f"綜上所陳，{defendant}因前揭侵權行為，{liability}原告之損害，"
        f"包含{join_chinese_clauses(group_sentences)}。{total_subject}共計{total_amount:,}元，"
        f"並自起訴狀繕本送達翌日起至清償日止，按年息5%計算之利息。"
    )


def extract_conclusion_damage_groups(damage_section: str, parties: dict) -> list[dict]:
    section_pattern = rf"(?ms)^[一二三四五六七八九十0-9]+、原告([{NAME_CHARS}]{{1,5}})部分\s*(.*?)(?=^[一二三四五六七八九十0-9]+、(?:原告|未特定|共同)|\Z)"
    groups = []
    for match in re.finditer(section_pattern, damage_section):
        claimant = normalize_plaintiff_name(match.group(1))
        if not claimant:
            continue
        groups.append({
            "claimant": claimant,
            "items": extract_conclusion_damage_items(match.group(2), claimant),
        })
    if groups:
        return groups

    plaintiff = extract_party_names(parties.get("原告", ""))
    claimant = plaintiff[0] if len(plaintiff) == 1 else ""
    return [{"claimant": claimant, "items": extract_conclusion_damage_items(damage_section, claimant)}]


def extract_conclusion_damage_items(text: str, claimant: str = "") -> list[dict]:
    item_pattern = r"(?ms)^（[一二三四五六七八九十0-9]+）([^\n]+)\n(.*?)(?=^（[一二三四五六七八九十0-9]+）|\Z)"
    items = []
    for match in re.finditer(item_pattern, text.strip()):
        title = match.group(1).strip()
        body = match.group(2).strip()
        amount = extract_claim_amount_from_block(f"{title}\n{body}", title, claimant)
        if amount is None:
            continue
        items.append({
            "label": conclusion_damage_label(title, body),
            "amount": amount,
        })
    return items


def conclusion_damage_label(title: str, body: str) -> str:
    source = f"{title}\n{body}"
    if "醫療" in title:
        if "醫藥及輔具費" in source:
            return "醫藥及輔具費"
        if "醫藥費" in source:
            return "醫藥費"
        return "醫療費用"
    if "交通" in title:
        return "交通費"
    if "看護" in title:
        return "看護費"
    if "工作" in title:
        return "工作收入損失"
    if "勞動能力" in title:
        return "勞動能力減損"
    if "慰撫" in title or "精神" in title:
        return "慰撫金"
    if "安全帽" in source and "鞋子" in source:
        return "安全帽及鞋子損失"
    if "眼鏡" in source:
        return "眼鏡毀損損失"
    if "手機" in source:
        return "手機維修費用"
    if "安全帽" in source:
        return "安全帽損失"
    if "鞋子" in source:
        return "鞋子損失"
    if "財物" in title:
        return "財物損失"
    if "車輛" in title or "修復" in title or "維修" in source or "機車" in source or "汽車" in source:
        return "車輛修理費"
    return title


def format_party_with_role(raw: str, role: str) -> str:
    names = extract_party_names(raw)
    if not names:
        return role
    if len(names) == 1:
        return f"{role}{names[0]}"
    return f"{role}{join_party_names(names)}"


def has_multiple_parties(raw: str) -> bool:
    return len(extract_party_names(raw)) > 1


def join_chinese_phrases(parts: list[str]) -> str:
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return f"{parts[0]}及{parts[1]}"
    return "、".join(parts[:-1]) + f"及{parts[-1]}"


def join_chinese_clauses(parts: list[str]) -> str:
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return "".join(parts)
    return "；以及".join(parts)


def join_party_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    if len(names) == 2:
        return f"{names[0]}與{names[1]}"
    return "、".join(names[:-1]) + f"與{names[-1]}"


def chinese_count_prefix(count: int) -> str:
    numerals = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    if 1 <= count <= 10:
        return numerals[count]
    return str(count)
