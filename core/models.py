"""
Shared data models for the Fact-Check Agent.

Claim is produced by core/claim_extractor.py (Phase 2) and consumed by
core/verifier.py (Phase 3), which attaches a VerificationResult to each
one. Both dataclasses live in this single module -- per the approved
architecture -- so the extractor and verifier can share types without a
circular import between them.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ClaimType(str, Enum):
    STATISTIC = "statistic"
    DATE = "date"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    OTHER = "other"


@dataclass(frozen=True)
class Claim:
    """A single factual claim extracted from a document."""

    text: str
    claim_type: ClaimType
    page: int  # 1-indexed page the claim was found on

    @property
    def dedupe_key(self) -> str:
        """Normalized key used to collapse near-identical repeated claims
        (the same stat quoted twice in a marketing PDF) before spending
        API budget verifying the same thing more than once."""
        return " ".join(self.text.lower().split())


class Verdict(str, Enum):
    VERIFIED = "Verified"
    INACCURATE = "Inaccurate"
    FALSE = "False"


@dataclass(frozen=True)
class VerificationResult:
    """Result of checking one Claim against live web data.

    Populated by core/verifier.py starting Phase 3 -- defined here now so
    that module and claim_extractor.py share one model file with no
    circular imports.
    """

    claim: Claim
    verdict: Verdict
    explanation: str
    source_url: Optional[str] = None
