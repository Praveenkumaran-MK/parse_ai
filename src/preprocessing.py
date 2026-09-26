"""
preprocessing.py
----------------
Normalization utilities for business_name and business_address fields.

The goal is NOT to destroy information — we keep raw + normalized versions
side by side, because some similarity features (e.g. raw edit distance)
work better on the original string, while others (token overlap, TF-IDF)
work better on a cleaned/canonicalized version.

Everything here is pure string processing — no external lookups, no APIs,
no internet augmentation. Fully compliant with the "no external data" rule.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# Legal-suffix / abbreviation canonicalization for business names.
# Keys are regex patterns (word-boundary matched), values are canonical forms.
NAME_SUFFIX_MAP = {
    r"\bcorp\b\.?": "corporation",
    r"\bco\b\.?": "company",
    r"\bltd\b\.?": "limited",
    r"\bpvt\b\.?": "private",
    r"\bincorporated\b": "inc",
    r"\binc\b\.?": "inc",
    r"\bllc\b\.?": "llc",
    r"\bllp\b\.?": "llp",
    r"\bplc\b\.?": "plc",
    r"\bindustries\b": "ind",
    r"\bind\b\.?": "ind",
    r"\benterprises\b": "ent",
    r"\bent\b\.?": "ent",
    r"\bassociates\b": "assoc",
    r"\bassoc\b\.?": "assoc",
    r"\bbros\b\.?": "brothers",
    r"\bmfg\b\.?": "manufacturing",
    r"\bintl\b\.?": "international",
    r"\bgrp\b\.?": "group",
}

# Tokens that carry near-zero discriminative value for blocking purposes.
# We strip these out when building the "core name" used for token blocking,
# but we keep them in the normalized name used for string-similarity features.
GENERIC_NAME_TOKENS = {
    "the", "and", "of", "a", "an",
    "corporation", "company", "limited", "private", "inc", "llc", "llp",
    "plc", "ind", "ent", "assoc", "brothers", "manufacturing",
    "international", "group", "co",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics (helps with transliteration variants)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(raw_name: str) -> str:
    """
    Canonical normalized form of a business name.
    - lowercase
    - strip accents (transliteration robustness)
    - replace '&' with 'and'
    - expand common legal-suffix abbreviations
    - strip punctuation
    - collapse whitespace
    """
    if raw_name is None or (isinstance(raw_name, float)):
        return ""
    name = str(raw_name).strip().lower()
    name = _strip_accents(name)
    name = name.replace("&", " and ")
    # Remove punctuation except internal alphanumerics/spaces
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    for pattern, canonical in NAME_SUFFIX_MAP.items():
        name = re.sub(pattern, canonical, name)

    name = re.sub(r"\s+", " ", name).strip()
    return name


def core_name_tokens(normalized_name: str) -> set:
    """
    Token set used for blocking: drop generic/legal tokens so blocking keys
    on the distinctive part of the business name (e.g. "sri lakshmi" out of
    "sri lakshmi enterprises pvt ltd").
    """
    tokens = normalized_name.split()
    core = {t for t in tokens if t not in GENERIC_NAME_TOKENS and len(t) > 1}
    # Fallback: if everything got stripped (e.g. name IS just "abc ltd"),
    # keep the original tokens so we don't lose blocking signal entirely.
    return core if core else set(tokens)


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

ADDRESS_ABBREV_MAP = {
    r"\brd\b\.?": "road",
    r"\bst\b\.?": "street",
    r"\bave\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bapt\b\.?": "apartment",
    r"\bbldg\b\.?": "building",
    r"\bfl\b\.?": "floor",
    r"\bste\b\.?": "suite",
    r"\bdr\b\.?": "drive",
    r"\bln\b\.?": "lane",
    r"\bcir\b\.?": "circle",
    r"\bct\b\.?": "court",
    r"\bhwy\b\.?": "highway",
    r"\bnr\b\.?": "near",
    r"\bopp\b\.?": "opposite",
    r"\bsec\b\.?": "sector",
    r"\bnagar\b": "nagar",  # kept explicit for clarity / future extension
}

# US ZIP: 5 digits optionally + 4. India PIN: 6 digits.
US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
INDIA_PIN_RE = re.compile(r"\b(\d{6})\b")

# Landmark phrases we pull out into a separate field rather than discard.
LANDMARK_RE = re.compile(
    r"\b(?:near|opposite|opp\.?|behind|next to|adjacent to)\s+([a-z0-9 ]{3,40})",
    re.IGNORECASE,
)


def normalize_address(raw_address: str) -> str:
    """Canonical normalized form of an address string."""
    if raw_address is None or (isinstance(raw_address, float)):
        return ""
    addr = str(raw_address).strip().lower()
    addr = _strip_accents(addr)
    addr = addr.replace("&", " and ")
    addr = re.sub(r"[^\w\s,]", " ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()

    for pattern, canonical in ADDRESS_ABBREV_MAP.items():
        addr = re.sub(pattern, canonical, addr)

    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def extract_postal_code(raw_address: str, country: str = "") -> str:
    """
    Extract a postal code if present. Country-aware but falls back
    gracefully for unseen countries (e.g. France) by trying both patterns.
    """
    if not raw_address:
        return ""
    text = str(raw_address)
    country_norm = (country or "").strip().lower()

    if country_norm == "india":
        m = INDIA_PIN_RE.search(text)
        return m.group(1) if m else ""
    if country_norm == "us":
        m = US_ZIP_RE.search(text)
        return m.group(1) if m else ""

    # Unknown country (e.g. France) or ambiguous: try both, prefer the
    # longer/more specific match.
    m6 = INDIA_PIN_RE.search(text)
    m5 = US_ZIP_RE.search(text)
    if m6:
        return m6.group(1)
    if m5:
        return m5.group(1)
    return ""


def extract_landmark(raw_address: str) -> str:
    """Pull out landmark references (e.g. 'Near SBI ATM') into their own field."""
    if not raw_address:
        return ""
    m = LANDMARK_RE.search(str(raw_address))
    return m.group(1).strip() if m else ""


def preprocess_dataframe(df, name_col="business_name", addr_col="business_address",
                          country_col="country"):
    """
    Adds normalized columns in-place-style (returns a copy) to a records dataframe:
      - name_norm, name_core_tokens (set)
      - addr_norm, postal_code, landmark
    """
    out = df.copy()
    out["name_norm"] = out[name_col].apply(normalize_name)
    out["name_core_tokens"] = out["name_norm"].apply(core_name_tokens)
    out["addr_norm"] = out[addr_col].apply(normalize_address)
    out["postal_code"] = [
        extract_postal_code(a, c) for a, c in zip(out[addr_col], out[country_col])
    ]
    out["landmark"] = out[addr_col].apply(extract_landmark)
    return out
