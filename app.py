"""
Fact-Check Agent -- Streamlit entrypoint.

Phase 4 wires the full pipeline together: PDF upload -> parsing -> claim
extraction -> claim verification -> results report, with staged progress
feedback and session-level caching so re-running the script (Streamlit
reruns the whole file on every widget interaction, including clicking the
download button below) never re-spends API calls on a document that has
already been processed in this session.
"""

import hashlib
import io
import logging

import pandas as pd
import streamlit as st

from config import ConfigError, load_settings, require_settings
from core.claim_extractor import ClaimExtractionError, extract_claims
from core.models import Verdict
from core.pdf_parser import PdfParsingError, parse_pdf
from core.verifier import verify_claims

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Fact-Check Agent",
    page_icon=":mag:",
    layout="centered",
)


def render_setup_status() -> bool:
    """Show whether required API keys are configured, without leaking
    them. Returns True only if both are present, so the caller can gate
    the uploader on it."""
    settings = load_settings()

    st.subheader("Environment status")
    col1, col2 = st.columns(2)
    with col1:
        if settings.mistral_api_key:
            st.success("Mistral API key: configured")
        else:
            st.error("Mistral API key: missing")
    with col2:
        if settings.tavily_api_key:
            st.success("Tavily API key: configured")
        else:
            st.error("Tavily API key: missing")

    if not settings.is_complete:
        st.info(
            "Add the missing key(s) to `.streamlit/secrets.toml` locally, "
            "or in the app's Secrets settings once deployed. "
            "See `.streamlit/secrets.toml.example` for the expected format."
        )
    return settings.is_complete


def _file_hash(file_bytes: bytes) -> str:
    """Stable identity for an uploaded file's contents, used as the
    session cache key so re-uploading the same PDF (or a Streamlit rerun
    triggered by an unrelated widget) doesn't re-run extraction/verification."""
    return hashlib.sha256(file_bytes).hexdigest()


def _process_document(file_bytes: bytes, settings) -> dict:
    """Run the full pipeline once: parse -> extract -> verify.

    Renders staged progress as it goes. Returns a dict the report
    renderer needs. Raises PdfParsingError / ClaimExtractionError on
    unrecoverable failure -- the caller handles those with st.error.
    """
    status = st.status("Processing document...", expanded=True)

    with status:
        st.write("Parsing PDF...")
        parsed = parse_pdf(io.BytesIO(file_bytes))

        if parsed.truncated:
            st.warning(
                f"This PDF has {parsed.total_pages_in_file} pages; only the "
                f"first {len(parsed.pages)} were processed."
            )
        if parsed.likely_scanned or not parsed.has_extractable_text:
            st.warning(
                "This PDF appears to be scanned or image-only -- little to "
                "no text could be extracted, so few or no claims may be found."
            )

        st.write("Extracting claims...")
        claims = extract_claims(parsed.full_text, settings)
        st.write(f"Found {len(claims)} checkable claim(s).")

        results = []
        if claims:
            st.write("Verifying claims against live web data...")
            results = verify_claims(claims, settings)

        status.update(label="Done", state="complete", expanded=False)

    return {"parsed": parsed, "claims": claims, "results": results}


def _results_to_dataframe(results) -> pd.DataFrame:
    rows = [
        {
            "Claim": r.claim.text,
            "Type": r.claim.claim_type.value,
            "Page": r.claim.page,
            "Verdict": r.verdict.value,
            "Explanation": r.explanation,
            "Source": r.source_url or "",
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def render_results(results) -> None:
    if not results:
        st.info("No checkable claims were found in this document.")
        return

    counts = {v: 0 for v in Verdict}
    for r in results:
        counts[r.verdict] += 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Verified", counts[Verdict.VERIFIED])
    col2.metric("Inaccurate", counts[Verdict.INACCURATE])
    col3.metric("False", counts[Verdict.FALSE])

    df = _results_to_dataframe(results)
    st.dataframe(df, width="stretch", hide_index=True)

    st.download_button(
        "Download report as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fact_check_report.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Fact-Check Agent")
    st.caption(
        "Upload a document, extract its factual claims, and verify them "
        "against live web sources."
    )

    keys_ready = render_setup_status()

    st.divider()
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF to fact-check",
        type=["pdf"],
        disabled=not keys_ready,
        help=None if keys_ready else "Configure both API keys above before uploading.",
    )

    if not keys_ready:
        st.caption("Upload is disabled until both API keys above are configured.")
        return

    if uploaded_file is None:
        st.caption("Upload a PDF to extract and verify its factual claims.")
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = _file_hash(file_bytes)

    cache = st.session_state.setdefault("processed_documents", {})

    if file_hash not in cache:
        try:
            settings = require_settings()
            cache[file_hash] = _process_document(file_bytes, settings)
        except ConfigError as exc:
            st.error(str(exc))
            return
        except PdfParsingError as exc:
            st.error(f"Could not read this PDF: {exc}")
            return
        except ClaimExtractionError as exc:
            st.error(f"Claim extraction failed: {exc}")
            return
    else:
        st.caption("Using cached results for this file from earlier in this session.")

    render_results(cache[file_hash]["results"])


if __name__ == "__main__":
    main()
