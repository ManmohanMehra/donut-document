"""
Unified schema (Path B) — one shared field set used for every document type,
with `card_type` acting as the routing token that tells the decoder which
document it's looking at. This replaces the old approach of hand-writing a
full field list per type, which doesn't scale once you're past a handful of
types — see STATUS.md for the migration note (105+ types as of 2026-07-13,
sourced from ~/Downloads/passport_data via scripts/rename.py's naming
convention: bare 3-letter alpha code = passport, hyphenated code = other
document type e.g. national ID / driving licence / residence permit).

Tradeoff accepted deliberately: this is the "pure unified" option, not
"unified core + per-type extras" — a handful of fields that only existed for
specific old types (place_of_issue, type P/S/D/O, country_code,
emirates_id/issuing_authority) are dropped rather than kept as optional
extras. If a specific type turns out to need a field the unified set doesn't
cover, add it here — every type gets it, always-null where not applicable,
same as mrz_line1/2 already are for non-MRZ document types.
"""

# 1. The one field set every document type uses. Any field not present /
#    not visible / not applicable to a given document type is left None —
#    e.g. mrz_line1/mrz_line2 are null for a driving licence, sex may be
#    null for a residence permit that doesn't print it, etc.
UNIFIED_FIELDS = {
    "document_number": None,   # passport no / national ID no / licence no / permit no
    "surname": None,
    "given_names": None,
    "dob": None,               # DD/MM/YYYY
    "sex": None,                # M / F
    "nationality": None,
    "date_of_issue": None,      # DD/MM/YYYY
    "date_of_expiry": None,     # DD/MM/YYYY
    "place_of_birth": None,
    "mrz_line1": None,          # null for non-MRZ document types
    "mrz_line2": None,
}

# 2. Every supported card_type, one per document type actually being
#    collected. Kept as a dict (not a flat list) so downstream code
#    (get_schema, add_tokens, vision_label, inference) doesn't need to
#    change — SCHEMAS[type] still returns a field dict, it's just the
#    *same* field dict for every type now.
#
# Naming: bare 3-letter ISO alpha code -> "{code}_passport" (e.g. ind, usa,
# zwe). Everything else -> the source folder name lowercased with hyphens
# replaced by underscores (e.g. ARE-FED-ID -> are_fed_id,
# QAT-RESIDENT-PERMIT -> qat_resident_permit).
#
# "indian_passport" is kept as-is (not renamed to "ind_passport") because
# the 19 already-verified samples in data/real/metadata.jsonl use that name,
# and renaming it would silently orphan verified ground truth. "cod_passport"
# and "foreign_passport" are kept as forward-looking / fallback entries even
# though no raw images exist for them yet.
_CARD_TYPE_NAMES = [
    "indian_passport", "foreign_passport", "cod_passport",
    "afg_passport", "are_dl", "are_fed_id", "are_id", "aus_passport", "aus_dl",
    "bel_passport", "bgd_passport", "bgd_national_id", "bhr_passport", "bhr_dl",
    "bhr_id", "can_passport", "chn_passport", "col_passport", "deu_passport",
    "deu_id", "dnk_passport", "egy_passport", "esp_passport", "esp_id",
    "eth_passport", "fra_passport", "gbr_passport", "gbr_dl", "geo_passport",
    "hkg_passport", "hkg_permanent_id", "idn_passport", "ind_addar", "ind_pan",
    "irn_passport", "irq_passport", "isr_passport", "ita_passport", "ita_id",
    "jor_passport", "jor_id", "jpn_passport", "kaz_passport", "ken_passport",
    "ken_id", "kwt_passport", "kwt_civil_id", "kwt_dl", "lby_passport",
    "lux_passport", "lva_passport", "mex_passport", "mmr_passport",
    "moz_passport", "mys_passport", "mys_dl", "mys_national_id", "nld_passport",
    "npl_passport", "nzl_passport", "omn_dl", "omn_resident_card",
    "pak_passport", "pak_id", "phl_passport", "phl_dl", "phl_health_id",
    "phl_national_id", "phl_ofw", "phl_pnid", "phl_postal_id",
    "phl_senior_citizen_id", "phl_tin", "phl_umid", "phl_voter_id", "qat_dl",
    "qat_resident_permit", "rou_id", "sau_passport", "sau_dl",
    "sau_national_id", "sau_resident_id", "sdn_passport", "syr_passport",
    "tha_passport", "tha_national_id", "tur_passport", "tur_dl", "tur_id",
    "tur_resident_permit", "uk_brp", "uk_dl", "usa_passport", "usa_dl",
    "usa_pr", "uzb_passport", "ven_passport", "vnm_passport", "vnm_id",
    "yem_passport", "zaf_passport", "zaf_dl", "zaf_id", "zwe_passport",
    "zwe_dl", "zwe_id",
]

SCHEMAS = {name: {"card_type": name, **UNIFIED_FIELDS} for name in _CARD_TYPE_NAMES}

# 3. Quick-access constant (used by FastAPI for input validation)
SUPPORTED_CARD_TYPES = list(SCHEMAS.keys())


# 4. Safe lookup helper (used in inference.py / app.py)
def get_schema(card_type: str) -> dict:
    """Return the schema dict for a card type, or raise ValueError if unknown."""
    if card_type not in SCHEMAS:
        raise ValueError(
            f"Unknown card type: '{card_type}'. "
            f"Supported types: {SUPPORTED_CARD_TYPES}"
        )
    return SCHEMAS[card_type].copy()  # .copy() so callers can't mutate the master schema


# 5. Token generator (used by add_tokens.py to avoid duplication)
def get_all_tokens() -> list[str]:
    """
    Generate all structural XML-style tokens needed for Donut's vocabulary:
    one <s_type>/</s_type> wrapper pair per card_type, plus one pair per
    shared field (deduplicated automatically since every type uses the same
    field dict).
    """
    tokens = []
    for card_type, fields in SCHEMAS.items():
        tokens.append(f"<s_{card_type}>")
        tokens.append(f"</s_{card_type}>")
        for field in fields:
            if field == "card_type":
                continue  # card_type is the wrapper tag, already handled above
            tokens.append(f"<s_{field}>")
            tokens.append(f"</s_{field}>")
    return list(dict.fromkeys(tokens))  # deduplicate, preserve order
