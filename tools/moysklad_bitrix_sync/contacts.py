"""Phone extraction and order/deal matching for the daily MoySklad <-> Bitrix24 sync.

The nightly sync (see README.md) reads orders from MoySklad and deals from
Bitrix24 through connectors, then uses this module for the deterministic parts:
pulling phone numbers out of free text, normalising them, and scoring whether a
MoySklad order and a Bitrix24 deal belong to the same client.

CLI:
    python3 contacts.py extract "Магомадов Адлан  8-920-092-46-42"
    python3 contacts.py normalize "8928 643 59 76"
    python3 contacts.py match order.json deals.json
"""

from __future__ import annotations

import json
import re
import sys

# Runs of digits with the separators people type inside phone numbers.
_CANDIDATE_RE = re.compile(r"\+?\d[\d \t\-\(\)\.]*\d")
_DIGIT_GROUP_RE = re.compile(r"\d+")

# Words that say nothing about who the client is.
_STOP_WORDS = {
    "клиент", "клиентка", "кл", "г", "гор", "город", "ул", "улица", "д", "дом",
    "кв", "квартира", "корп", "корпус", "с", "село", "пос", "р", "рн", "респ",
    "россия", "чеченская", "мк", "опт", "розничный", "покупатель",
}

AUTO_MATCH_SCORE = 70
REVIEW_MATCH_SCORE = 45
SUM_TOLERANCE = 0.02


def normalize_phone(raw: str | None) -> str | None:
    """Return the phone as +7XXXXXXXXXX (or +<country><number>), or None if it is not a phone."""
    if not raw:
        return None
    raw = raw.strip()
    digits = re.sub(r"\D", "", raw)
    if raw.startswith("+"):
        if digits.startswith("7"):
            return "+" + digits if len(digits) == 11 else None
        return "+" + digits if 10 <= len(digits) <= 15 else None
    if len(digits) == 11 and digits[0] in "78":
        return "+7" + digits[1:]
    if len(digits) == 10 and digits[0] == "9":
        return "+7" + digits
    return None


def extract_phones(text: str | None) -> list[str]:
    """Find every phone number in free text (order comment, counterparty name, description)."""
    if not text:
        return []
    found: list[str] = []
    for match in _CANDIDATE_RE.finditer(text):
        candidate = match.group(0)
        if candidate.startswith("+"):
            phone = normalize_phone(candidate)
            if phone:
                found.append(phone)
                continue
        # Several phones can be glued together ("89204222289  89288888389"),
        # so collect digit groups until they form a valid number.
        buffer = ""
        for group in _DIGIT_GROUP_RE.findall(candidate):
            buffer += group
            phone = normalize_phone(buffer)
            if phone is None and len(buffer) > 11:
                buffer = group
                phone = normalize_phone(buffer)
            if phone:
                found.append(phone)
                buffer = ""
    unique: list[str] = []
    for phone in found:
        if phone not in unique:
            unique.append(phone)
    return unique


def name_tokens(text: str | None) -> set[str]:
    """Lower-cased meaningful words of a name or address, without digits and filler words."""
    if not text:
        return set()
    text = text.lower().replace("ё", "е")
    words = re.findall(r"[a-zа-я]+", text)
    return {w for w in words if len(w) >= 2 and w not in _STOP_WORDS}


def address_tokens(text: str | None) -> set[str]:
    """Words and house numbers of an address, so "Ключевая, 142" matches "ул Ключевая 142"."""
    if not text:
        return set()
    return name_tokens(text) | set(re.findall(r"\d+[а-яa-z]?", text.lower()))


def _phones_of(record: dict) -> set[str]:
    phones = set()
    for raw in record.get("phones") or []:
        phone = normalize_phone(raw) or None
        phones.update([phone] if phone else extract_phones(raw))
    for field in ("name", "description"):
        phones.update(extract_phones(record.get(field)))
    return phones


def match_score(order: dict, deal: dict) -> dict:
    """Score how likely a MoySklad order and a Bitrix24 deal are the same client.

    Both records are dicts with optional keys: phones (list), name, address, sum,
    description. A shared phone is decisive; otherwise name, address and amount
    add up, and the caller decides with AUTO_MATCH_SCORE / REVIEW_MATCH_SCORE.
    """
    reasons = []
    shared_phones = _phones_of(order) & _phones_of(deal)
    if shared_phones:
        return {"score": 100, "reasons": ["телефон " + ", ".join(sorted(shared_phones))]}

    score = 0
    common_names = name_tokens(order.get("name")) & name_tokens(deal.get("name"))
    if len(common_names) >= 2:
        score += 40
        reasons.append("ФИО " + " ".join(sorted(common_names)))
    elif common_names:
        score += 20
        reasons.append("имя " + " ".join(sorted(common_names)))

    common_address = address_tokens(order.get("address")) & address_tokens(deal.get("address"))
    if len(common_address) >= 2:
        score += 25
        reasons.append("адрес " + " ".join(sorted(common_address)))

    order_sum, deal_sum = order.get("sum") or 0, deal.get("sum") or 0
    if order_sum > 0 and deal_sum > 0 and abs(order_sum - deal_sum) <= SUM_TOLERANCE * max(order_sum, deal_sum):
        score += 25
        reasons.append(f"сумма {order_sum:.0f} ≈ {deal_sum:.0f}")

    return {"score": score, "reasons": reasons}


def best_match(order: dict, deals: list[dict]) -> dict:
    """Pick the best deal for an order and say whether it is safe to sync automatically."""
    scored = sorted(
        ({"deal": deal, **match_score(order, deal)} for deal in deals),
        key=lambda item: item["score"],
        reverse=True,
    )
    if not scored or scored[0]["score"] < REVIEW_MATCH_SCORE:
        return {"decision": "no_match", "candidates": scored[:3]}
    top = scored[0]
    tied = len(scored) > 1 and scored[1]["score"] == top["score"]
    if top["score"] >= AUTO_MATCH_SCORE and not tied:
        return {"decision": "auto", "match": top, "candidates": scored[:3]}
    return {"decision": "review", "match": top, "candidates": scored[:3]}


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "extract":
        print(json.dumps(extract_phones(" ".join(argv[1:])), ensure_ascii=False))
    elif len(argv) >= 2 and argv[0] == "normalize":
        print(json.dumps(normalize_phone(" ".join(argv[1:])), ensure_ascii=False))
    elif len(argv) == 3 and argv[0] == "match":
        with open(argv[1], encoding="utf-8") as f:
            order = json.load(f)
        with open(argv[2], encoding="utf-8") as f:
            deals = json.load(f)
        print(json.dumps(best_match(order, deals), ensure_ascii=False, indent=2))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
