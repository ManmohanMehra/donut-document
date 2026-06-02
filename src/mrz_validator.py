# src/mrz_validator.py

MRZ_WEIGHTS = [7, 3, 1]
MRZ_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"

def mrz_check_digit(s: str) -> int:
    """Compute ICAO MRZ check digit for a string."""
    total = 0
    for i, ch in enumerate(s):
        if ch not in MRZ_CHARS:
            return -1   # Invalid character
        val = int(ch) if ch.isdigit() else (0 if ch == "<" else ord(ch) - 55)
        total += val * MRZ_WEIGHTS[i % 3]
    return total % 10

def validate_mrz(line1: str, line2: str) -> dict:
    """
    Validate key check digits in an ICAO TD3 (passport) MRZ.
    Returns a dict of field → pass/fail.
    """
    # Standardize length and remove common OCR noise
    line1 = line1.strip().upper().replace(" ", "")
    line2 = line2.strip().upper().replace(" ", "")
    
    if len(line1) != 44 or len(line2) != 44:
        return {"valid": False, "reason": f"MRZ lines must be 44 characters each (Got {len(line1)} and {len(line2)})"}

    checks = {}

    try:
        # Passport number (chars 1–9, check digit at 10)
        checks["passport_number"] = mrz_check_digit(line2[0:9]) == int(line2[9])

        # Date of birth (chars 14–19, check at 20)
        checks["dob"] = mrz_check_digit(line2[13:19]) == int(line2[19])

        # Date of expiry (chars 22–27, check at 28)
        checks["expiry"] = mrz_check_digit(line2[21:27]) == int(line2[27])

        # Overall composite check (chars 1–10, 14–20, 22–43 of line2)
        composite = line2[0:10] + line2[13:20] + line2[21:43]
        checks["composite"] = mrz_check_digit(composite) == int(line2[43])

        checks["valid"] = all(checks.values())
    except (ValueError, IndexError):
        return {"valid": False, "reason": "Structural error in MRZ format"}

    return checks
