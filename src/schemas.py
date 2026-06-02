SCHEMAS = {

    "indian_passport": {
        "card_type": "indian_passport",
        "type": None,              # P=Regular, S=Service, D=Diplomatic, O=Official
        "country_code": None,  # 3-letter ISO code e.g. GBR, USA, DEU
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "nationality": None,
        "dob": None,               # DD/MM/YYYY
        "sex": None,               # M / F
        "place_of_birth": None,
        "place_of_issue": None,    # City where passport was issued — e.g. BENGALURU
        "date_of_issue": None,
        "date_of_expiry": None,
        "mrz_line1": None,         # Raw MRZ string — used for checksum validation
        "mrz_line2": None,
    },

    "foreign_passport": {
        "card_type": "foreign_passport",
        "issuing_country": None,   # 3-letter ISO code e.g. GBR, USA, DEU
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "nationality": None,
        "dob": None,
        "sex": None,
        "date_of_expiry": None,
        "mrz_line1": None,
        "mrz_line2": None,
    },

    # Add future card types here following the same pattern
    # "aadhaar": { ... }
    # "pan": { ... }
}

# 1. Quick-access constant (used by FastAPI for input validation)
SUPPORTED_CARD_TYPES = list(SCHEMAS.keys())

# 2. Safe lookup helper (used in inference.py / app.py)
def get_schema(card_type: str) -> dict:
    """Return the schema dict for a card type, or raise ValueError if unknown."""
    if card_type not in SCHEMAS:
        raise ValueError(
            f"Unknown card type: '{card_type}'. "
            f"Supported types: {SUPPORTED_CARD_TYPES}"
        )
    return SCHEMAS[card_type].copy()  # .copy() so callers can't mutate the master schema

# 3. Token generator (used by add_tokens.py to avoid duplication)
def get_all_tokens() -> list[str]:
    """
    Generate all structural XML-style tokens needed for Donut's vocabulary.
    Called once by add_tokens.py.
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

