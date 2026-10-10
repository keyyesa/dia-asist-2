"""
Report Analysis Agent: extracts diabetes-relevant lab values from a report's raw text and
classifies each value against standard reference ranges using a deterministic rule table
(so classification never depends on the LLM's arithmetic).

Extraction is RULE-BASED FIRST: tolerant regular expressions find each parameter by its common
names/abbreviations (including OCR-style typos such as "HbAlC") and read the value, unit and the
lab's own reference range that follow. Rules are instant, free and cannot hallucinate. The LLM is
only used as a fallback when the rules find nothing (unusual layouts).

Safeguards:
  * values returned by the LLM are dropped if the number does not literally appear in the report;
  * common unit variants (mmol/L, umol/L, mmol/mol, mg/mmol) are converted to the canonical unit;
  * thresholds in "interpretation legends" (">8%", "7-8%") are never mistaken for results;
  * ranges use exclusive upper bounds, so decimal values never fall into gaps.

Labels say "..._range" on purpose: a single lab value is not a diagnosis.
"""
import math
import re
from src import llm_client
from src.agents.json_utils import extract_json

INF = math.inf

# Ordered bands per parameter: a value belongs to the first band whose (exclusive)
# upper bound is greater than the value.
REFERENCE_BANDS = {
    "HbA1c": {"unit": "%", "bands": [(4.0, "low"), (5.7, "normal"), (6.5, "prediabetes_range"), (INF, "diabetes_range")]},
    # Estimated average glucose (derived from HbA1c): 5.7% ~ 117 mg/dL, 6.5% ~ 140 mg/dL.
    "eAG": {"unit": "mg/dL", "bands": [(117, "normal"), (140, "prediabetes_range"), (INF, "diabetes_range")]},
    "FBG": {"unit": "mg/dL", "bands": [(70, "low"), (100, "normal"), (126, "prediabetes_range"), (INF, "diabetes_range")]},
    # Random glucose has no prediabetes category; >=200 mg/dL counts only together with classic symptoms.
    "RBG": {"unit": "mg/dL", "bands": [(70, "low"), (200, "below_diagnostic_threshold"), (INF, "diabetes_range_if_symptomatic")]},
    "OGTT_2hr": {"unit": "mg/dL", "bands": [(140, "normal"), (200, "prediabetes_range"), (INF, "diabetes_range")]},
    # Creatinine ranges differ by sex/lab: use the lab's own reference interval when available.
    "Serum_Creatinine": {"unit": "mg/dL", "bands": [(1.3, "normal"), (INF, "elevated")]},
    # ADA: UACR 30-299 mg/g moderately increased, >=300 mg/g severely increased.
    "Urine_Albumin": {"unit": "mg/g", "bands": [(30, "normal"), (300, "microalbuminuria"), (INF, "macroalbuminuria")]},
    "LDL": {"unit": "mg/dL", "bands": [(100, "optimal"), (160, "borderline"), (INF, "high")]},
    # HDL goal is sex-specific (men >=40, women >=50 mg/dL); 40 is used as the general cut-off.
    "HDL": {"unit": "mg/dL", "bands": [(40, "low"), (INF, "acceptable")]},
    "Triglycerides": {"unit": "mg/dL", "bands": [(150, "normal"), (200, "borderline"), (500, "high"), (INF, "very_high")]},
}

PARAM_LABELS = {
    "HbA1c": "HbA1c", "eAG": "estimated average glucose (eAG)", "FBG": "fasting blood glucose (FBG)",
    "RBG": "random blood glucose (RBG)", "OGTT_2hr": "2-hour OGTT glucose",
    "Serum_Creatinine": "serum creatinine", "Urine_Albumin": "urine albumin",
    "LDL": "LDL cholesterol", "HDL": "HDL cholesterol", "Triglycerides": "triglycerides",
}


def _same(v):
    return v


_GLUCOSE = {"mg/dl": _same, "mmol/l": lambda v: v * 18.016}
_CHOL = {"mg/dl": _same, "mmol/l": lambda v: v * 38.67}

# unit (normalised: lowercase, no spaces) -> converter into the canonical unit
CONVERSIONS = {
    "HbA1c": {"%": _same, "mmol/mol": lambda v: v / 10.929 + 2.15},
    "eAG": _GLUCOSE,
    "FBG": _GLUCOSE,
    "RBG": _GLUCOSE,
    "OGTT_2hr": _GLUCOSE,
    "Serum_Creatinine": {"mg/dl": _same, "umol/l": lambda v: v / 88.4},
    "Urine_Albumin": {"mg/g": _same, "mg/gcreat": _same, "mg/gcr": _same, "ug/mg": _same,
                      "mg/mmol": lambda v: v * 8.84},
    "LDL": _CHOL,
    "HDL": _CHOL,
    "Triglycerides": {"mg/dl": _same, "mmol/l": lambda v: v * 88.57},
}

# ----------------------------------------------------------------------------- rule-based extraction
_ALIASES = {
    # "HbAlC" / "HBA1C" / "Hb A1c" / "A1c" / "glycosylated haemoglobin"
    "HbA1c": r"hb\s*a\s*[1il]\s*c|\ba[1il]c\b|glyc\w*\s*h(?:a)?emoglobin",
    "eAG": r"estimated\s+average\s+glucose|\be\.?a\.?g\b|average\s+glucose",
    "FBG": r"fasting\s+(?:blood\s+|plasma\s+)?(?:glucose|sugar)|\bfbg\b|\bfbs\b|\bfpg\b",
    "RBG": r"random\s+(?:blood\s+|plasma\s+)?(?:glucose|sugar)|\brbg\b|\brbs\b|\brpg\b",
    "OGTT_2hr": r"\bogtt\b|2[\s-]*h(?:ou)?rs?\s*(?:post[\s-]*load|plasma)?\s*glucose",
    "Serum_Creatinine": r"(?<!urine )(?<!urinary )creatinine",
    "Urine_Albumin": r"urine\s+albumin|microalbumin\w*|albumin\s*/\s*creatinine|\buacr\b|\bacr\b",
    "LDL": r"\bldl\b",
    "HDL": r"\bhdl\b",
    "Triglycerides": r"triglycerides?|\btg\b",
}
_ALIAS_RX = {p: re.compile(rx, re.IGNORECASE) for p, rx in _ALIASES.items()}

_WINDOW = 80  # characters after a test name in which its result is searched
_NUMBER_RX = re.compile(r"(?<![\w.,<>\u2264\u2265=\-\u2013\u2014])(\d{1,5}(?:[.,]\d{1,3})?)(?![\w.,]?\d)")
_RANGE_RX = re.compile(r"(\d+(?:[.,]\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:[.,]\d+)?)")
_UNIT_RX = re.compile(
    r"%|mg\s*/\s*d[lL]|mmol\s*/\s*mol|mmol\s*/\s*[lL]|[u\u00b5\u03bc]mol\s*/\s*[lL]"
    r"|mg\s*/\s*mmol|mg\s*/\s*g(?:\s*creat\w*)?|[u\u00b5\u03bc]g\s*/\s*mg", re.IGNORECASE)


def _first_value(window: str):
    """First plausible RESULT in the window -> (number_text, end_offset). Skips thresholds such as
    '>8%' and range limits such as '7-8%' / '4.0 - 6.0' (those are not the patient's result)."""
    for m in _NUMBER_RX.finditer(window):
        if re.match(r"\s*[-\u2013\u2014]\s*\d", window[m.end():]):  # low end of a range
            continue
        if re.search(r"\d\s*[-\u2013\u2014]\s*$", window[:m.start()]):  # high end of a range
            continue
        return m.group(1), m.end()
    return None


def extract_values_rules(text: str) -> dict:
    """{param: {"value": float, "unit": str|None, "lab_range": str|None}} found by pattern matching."""
    matches = sorted((m.start(), m.end(), p) for p, rx in _ALIAS_RX.items() for m in rx.finditer(text))
    found = {}
    for i, (start, end, param) in enumerate(matches):
        if param in found:
            continue
        limit = min(end + _WINDOW, len(text))
        for s2, _e2, p2 in matches[i + 1:]:
            if s2 < end or (p2 == param and s2 - end <= 15):  # overlapping / same test named twice in a row
                continue
            limit = min(limit, s2)  # never read the next test's value
            break
        hit = _first_value(text[end:limit])
        if not hit:
            continue
        number, vend = hit
        try:
            value = float(number.replace(",", "."))
        except ValueError:
            continue
        rest_start = end + vend
        stop = next((s2 for s2, _e, _p in matches if s2 >= rest_start), len(text))
        rest = text[rest_start:min(stop, rest_start + _WINDOW)]
        unit_m, range_m = _UNIT_RX.search(rest), _RANGE_RX.search(rest)
        found[param] = {
            "value": value,
            "unit": re.sub(r"\s+", "", unit_m.group(0)) if unit_m else None,
            "lab_range": f"{range_m.group(1)} - {range_m.group(2)}" if range_m else None,
        }
    return found


# ----------------------------------------------------------------------------- LLM fallback
EXTRACTION_SYSTEM_PROMPT = """You are the Report Analysis Agent for diabetes laboratory reports.
Extract these parameters if present in the report text:
HbA1c, eAG, FBG, RBG, OGTT_2hr, Serum_Creatinine, Urine_Albumin, LDL, HDL, Triglycerides
Return a flat JSON object where each key maps to {"value": <number>, "unit": "<unit exactly as printed in the report, or null>"}.
Copy numbers exactly as printed. Never estimate, infer or calculate a value that is not printed.
Omit any parameter that is not present. Return ONLY valid JSON, no markdown fences, no explanation."""


def _norm_unit(unit):
    if not unit:
        return None
    return str(unit).strip().lower().replace(" ", "").replace("\u00b5", "u").replace("\u03bc", "u")


def _appears_in_text(value: float, text: str) -> bool:
    """True if the number is literally present in the report text (hallucination guard)."""
    candidates = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    candidates |= {c.replace(".", ",") for c in candidates}
    for c in candidates:
        if re.search(rf"(?<![\d.,]){re.escape(c)}(?!\d|[.,]\d)", text):
            return True
    return False


def _split_value_unit(raw):
    """Accepts {"value": 7.2, "unit": "%"} or a bare number/string. Returns (float|None, unit|None)."""
    unit = None
    if isinstance(raw, dict):
        unit = raw.get("unit")
        raw = raw.get("value")
    try:
        return float(str(raw).replace(",", ".").strip().rstrip("%")), unit
    except (TypeError, ValueError):
        return None, unit


def convert_to_canonical(param: str, value: float, unit):
    """Returns the value in the canonical unit, or None if the unit is not supported."""
    unit_n = _norm_unit(unit)
    if unit_n is None:
        return value  # no unit printed: assume the canonical unit
    converter = CONVERSIONS.get(param, {}).get(unit_n)
    return converter(value) if converter else None


def classify_value(param: str, value: float) -> str:
    spec = REFERENCE_BANDS.get(param)
    if not spec:
        return "unknown"
    for upper, label in spec["bands"]:
        if value < upper:
            return label
    return spec["bands"][-1][1]


def extract_values(report_text: str) -> dict:
    """LLM extraction (fallback). Returns {} if the model is unavailable or answers badly."""
    try:
        result = llm_client.generate(report_text, system_instruction=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001
        return {}
    return extract_json(result)


def _interpret(param: str, value: float, unit, lab_range=None) -> dict:
    canonical_unit = REFERENCE_BANDS[param]["unit"]
    converted = convert_to_canonical(param, value, unit)
    if converted is None:
        entry = {"value": value, "unit": unit, "classification": "unit_not_supported"}
    else:
        entry = {"value": round(converted, 2), "unit": canonical_unit,
                 "classification": classify_value(param, converted)}
        if _norm_unit(unit) not in (None, _norm_unit(canonical_unit)):
            entry["reported_as"] = f"{value:g} {unit}"
    if lab_range:
        entry["lab_range"] = lab_range
    return entry


def analyze_report(report_text: str) -> dict:
    """Returns {param: {value, unit, classification[, reported_as, lab_range]}} for every trustworthy
    parameter. Rules first (fast, exact); the LLM is only asked when the rules find nothing."""
    interpretation = {}
    for param, item in extract_values_rules(report_text).items():
        interpretation[param] = _interpret(param, item["value"], item["unit"], item["lab_range"])
    if interpretation:
        return interpretation

    for param, raw in extract_values(report_text).items():
        if param not in REFERENCE_BANDS:
            continue
        value, unit = _split_value_unit(raw)
        if value is None or not _appears_in_text(value, report_text):
            continue  # missing, unparseable, or not actually printed in the report
        interpretation[param] = _interpret(param, value, unit)
    return interpretation


def missing_parameters(interpretation: dict) -> list:
    """Friendly names of the standard parameters that were NOT found in the report."""
    return [label for key, label in PARAM_LABELS.items() if key not in (interpretation or {})]
