"""
Claim verification for the Fact-Check Agent.

For each unique claim: search Tavily for live evidence (run in parallel --
Tavily's free tier has no meaningful per-second cap, so this is purely
network-bound), then judge the claim against that evidence using Mistral
Large, batched several claims per call to stay within Mistral's free-tier
1 RPS limit.

Grounding is enforced twice: the judge prompt requires the model to only
cite a source_url that was actually in the search results it was given,
and a second programmatic check after parsing discards/downgrades any
verdict that cites a URL not present in that claim's own search results.
This is the hallucination guard called for in the approved blueprint.

A single claim's search or judge failure never aborts the whole run --
it degrades to a "False" verdict with an explanatory message, so one bad
claim never blocks the rest of the report.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from config import Settings
from core.api_clients import ApiClientError, call_mistral_chat, call_tavily_search
from core.models import Claim, Verdict, VerificationResult
from utils.prompts import VERIFICATION_SYSTEM_PROMPT, build_verification_user_message

logger = logging.getLogger(__name__)

JUDGE_MODEL = "mistral-large-latest"
SEARCH_WORKERS = 4
BATCH_SIZE = 5
RESULTS_PER_CLAIM = 3
MAX_CONTENT_CHARS = 500  # truncate each search snippet to control token use


class ClaimVerificationError(RuntimeError):
    """Raised internally when a judge call can't be parsed. Always caught
    within this module -- callers of verify_claims never see it, since a
    failed batch degrades to per-claim 'False' results instead."""


def verify_claims(claims: List[Claim], settings: Settings) -> List[VerificationResult]:
    """Verify a list of claims against live web evidence.

    Returns one VerificationResult per input claim, in the same order
    claims were given. Never raises for an individual claim's search or
    judge failure.
    """
    if not claims:
        return []

    search_results = _search_all(claims, settings)

    # Claims with zero search evidence skip the judge call entirely --
    # there's nothing to ground a verdict in, so the answer is mechanical
    # rather than something worth spending a Mistral call on.
    judgeable: List[Claim] = []
    results_by_key: Dict[str, VerificationResult] = {}

    for claim in claims:
        hits = search_results.get(claim.dedupe_key, [])
        if not hits:
            results_by_key[claim.dedupe_key] = VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                explanation="No search results were found to support this claim.",
                source_url=None,
            )
        else:
            judgeable.append(claim)

    for batch in _chunk(judgeable, BATCH_SIZE):
        results_by_key.update(_judge_batch(batch, search_results, settings))

    # Re-assemble in original order; defensively fall back if a claim
    # somehow has no result yet (e.g. judge silently dropped it).
    ordered: List[VerificationResult] = []
    for claim in claims:
        result = results_by_key.get(claim.dedupe_key)
        if result is None:
            result = VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                explanation="Verification could not be completed for this claim.",
                source_url=None,
            )
        ordered.append(result)
    return ordered


def _search_all(claims: List[Claim], settings: Settings) -> Dict[str, list]:
    """Run Tavily search for every unique claim in parallel. A failed
    search for one claim logs a warning and yields an empty result list
    for it rather than aborting the whole batch."""
    unique_claims = list({c.dedupe_key: c for c in claims}.values())
    results: Dict[str, list] = {}

    def _search_one(claim: Claim):
        try:
            response = call_tavily_search(
                settings, query=claim.text, max_results=RESULTS_PER_CLAIM
            )
            return claim.dedupe_key, response.get("results", []) or []
        except ApiClientError as exc:
            logger.warning("Tavily search failed for claim %r: %s", claim.text[:60], exc)
            return claim.dedupe_key, []

    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        futures = [pool.submit(_search_one, c) for c in unique_claims]
        for future in as_completed(futures):
            key, hits = future.result()
            results[key] = hits

    return results


def _chunk(items: List, size: int) -> List[List]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _judge_batch(
    batch: List[Claim], search_results: Dict[str, list], settings: Settings
) -> Dict[str, VerificationResult]:
    """Send one batch of claims + their search evidence to Mistral Large
    and return verdicts keyed by claim.dedupe_key, with the hallucination
    guard applied to every result before it's returned."""
    batch_payload = []
    claim_by_id: Dict[int, Claim] = {}
    for i, claim in enumerate(batch, start=1):
        claim_by_id[i] = claim
        batch_payload.append(
            {
                "claim_id": i,
                "claim_text": claim.text,
                "claim_type": claim.claim_type.value,
                "search_results": [
                    {
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "content": str(r.get("content", ""))[:MAX_CONTENT_CHARS],
                    }
                    for r in search_results.get(claim.dedupe_key, [])
                ],
            }
        )

    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_verification_user_message(batch_payload)},
    ]

    try:
        response = call_mistral_chat(
            settings, messages=messages, model=JUDGE_MODEL, json_mode=True
        )
        raw_results = _parse_judge_response(response)
    except (ApiClientError, ClaimVerificationError) as exc:
        logger.warning("Judge call failed for a batch of %d claims: %s", len(batch), exc)
        return {
            claim.dedupe_key: VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                explanation="Verification judge call failed; could not assess this claim.",
                source_url=None,
            )
            for claim in batch
        }

    output: Dict[str, VerificationResult] = {}
    for item in raw_results:
        claim = claim_by_id.get(item.get("claim_id"))
        if claim is None:
            logger.warning("Judge response referenced unknown claim_id: %r", item)
            continue
        output[claim.dedupe_key] = _build_grounded_result(claim, item, search_results)

    # Any claim the judge silently skipped in its response still needs a result.
    for claim in batch:
        if claim.dedupe_key not in output:
            output[claim.dedupe_key] = VerificationResult(
                claim=claim,
                verdict=Verdict.FALSE,
                explanation="Judge did not return a verdict for this claim.",
                source_url=None,
            )

    return output


def _parse_judge_response(response: dict) -> list:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClaimVerificationError(f"Unexpected Mistral response shape: {exc}") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ClaimVerificationError(f"Judge did not return valid JSON: {exc}") from exc

    results = parsed.get("results", [])
    if not isinstance(results, list):
        raise ClaimVerificationError(f"Expected 'results' to be a list, got: {type(results)}")
    return results


def _build_grounded_result(
    claim: Claim, item: dict, search_results: Dict[str, list]
) -> VerificationResult:
    """Apply the hallucination guard: a verdict's source_url is only kept
    if it's literally one of the URLs Tavily returned for this claim.
    Otherwise the verdict is downgraded to False, since an unverifiable
    or invented source is treated the same as having no evidence at all.
    """
    raw_verdict = str(item.get("verdict", "")).strip()
    try:
        verdict = Verdict(raw_verdict)
    except ValueError:
        logger.warning(
            "Judge returned unrecognized verdict %r, defaulting to False.", raw_verdict
        )
        verdict = Verdict.FALSE

    explanation = str(item.get("explanation", "")).strip() or "No explanation provided."
    source_url = item.get("source_url")

    valid_urls = {r.get("url") for r in search_results.get(claim.dedupe_key, [])}
    if source_url and source_url not in valid_urls:
        logger.warning(
            "Judge cited an unverifiable source_url for claim %r -- downgrading to False.",
            claim.text[:60],
        )
        return VerificationResult(
            claim=claim,
            verdict=Verdict.FALSE,
            explanation=(
                "The verification model cited a source that wasn't in the search "
                "results actually retrieved, so this claim could not be grounded "
                "in real evidence."
            ),
            source_url=None,
        )

    return VerificationResult(
        claim=claim,
        verdict=verdict,
        explanation=explanation,
        source_url=source_url or None,
    )
