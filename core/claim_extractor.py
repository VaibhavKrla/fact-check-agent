"""
Claim extraction for the Fact-Check Agent.

Sends the full parsed document text to Mistral in a single call (Phase
1's ParsedDocument.full_text is already page-tagged and table-aware),
parses the structured JSON response into Claim objects, deduplicates
repeated claims, and caps the result to MAX_CLAIMS.

One call over the whole document -- rather than one call per page -- is
the approved approach: it stays fast under the free-tier 1 RPS limit and
lets the model see cross-page context when classifying claims.
"""

import json
import logging
from typing import List, Optional

from config import Settings
from core.api_clients import ApiClientError, call_mistral_chat
from core.models import Claim, ClaimType
from utils.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_message

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "mistral-small-latest"
MAX_CLAIMS = 30

_VALID_TYPES = {t.value for t in ClaimType}


class ClaimExtractionError(RuntimeError):
    """Raised when claim extraction fails outright: an API error, or a
    response that can't be parsed into the expected JSON shape."""


def extract_claims(
    document_text: str,
    settings: Settings,
    model: str = EXTRACTION_MODEL,
    max_claims: int = MAX_CLAIMS,
) -> List[Claim]:
    """Extract, validate, and deduplicate claims from document text.

    Args:
        document_text: page-tagged text, typically
            ParsedDocument.full_text from core/pdf_parser.py.
        settings: loaded API settings (needs mistral_api_key).
        model: Mistral model id, overridable for testing.
        max_claims: hard cap on returned claims after dedup -- protects
            verification cost downstream in Phase 3.

    Returns:
        A list of unique Claim objects, capped at max_claims. Returns an
        empty list with no API call made if document_text is blank --
        callers (Phase 4 UI) should check ParsedDocument.has_extractable_text
        first, but this guard avoids wasting a call either way.

    Raises:
        ClaimExtractionError: on API failure or an unparseable response.
    """
    if not document_text.strip():
        logger.info("extract_claims called with empty text -- skipping API call.")
        return []

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_extraction_user_message(document_text)},
    ]

    try:
        response = call_mistral_chat(settings, messages=messages, model=model, json_mode=True)
    except ApiClientError as exc:
        raise ClaimExtractionError(f"Mistral extraction call failed: {exc}") from exc

    raw_claims = _parse_response(response)
    claims = [c for c in (_to_claim(item) for item in raw_claims) if c is not None]

    deduped = _dedupe(claims)

    if len(deduped) > max_claims:
        logger.warning(
            "Extracted %d unique claims; truncating to max_claims=%d.",
            len(deduped),
            max_claims,
        )
        deduped = deduped[:max_claims]

    return deduped


def _parse_response(response: dict) -> list:
    """Pull the JSON-mode content out of a Mistral chat completion
    response and return the raw "claims" list, defensively."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClaimExtractionError(
            f"Unexpected Mistral response shape: {exc}. Raw response: {response!r}"
        ) from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ClaimExtractionError(
            f"Mistral did not return valid JSON despite json_mode: {exc}. "
            f"Content: {content!r}"
        ) from exc

    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        raise ClaimExtractionError(f"Expected 'claims' to be a list, got: {type(claims)}")
    return claims


def _to_claim(item) -> Optional[Claim]:
    """Convert one raw claim dict into a validated Claim, or None if it's
    malformed enough to skip. Logged, not fatal -- one bad item shouldn't
    discard every other valid claim in the batch."""
    if not isinstance(item, dict):
        logger.warning("Skipping malformed claim item (not a dict): %r", item)
        return None

    text = str(item.get("text", "")).strip()
    if not text:
        logger.warning("Skipping claim with empty text: %r", item)
        return None

    raw_type = str(item.get("type", "other")).strip().lower()
    claim_type = ClaimType(raw_type) if raw_type in _VALID_TYPES else ClaimType.OTHER

    try:
        page = int(item.get("page", 0))
    except (TypeError, ValueError):
        page = 0

    return Claim(text=text, claim_type=claim_type, page=page)


def _dedupe(claims: List[Claim]) -> List[Claim]:
    """Collapse claims that are identical once normalized (the same stat
    repeated verbatim elsewhere in the doc), preserving first-seen order."""
    seen = set()
    deduped = []
    for claim in claims:
        if claim.dedupe_key in seen:
            continue
        seen.add(claim.dedupe_key)
        deduped.append(claim)
    return deduped
