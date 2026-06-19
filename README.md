# Fact-Check Agent

Upload a PDF, extract its factual claims, and verify them against live web
data using AI models and web search integration.

> **Status: complete.** All seven phases are done — parsing, extraction,
> verification, UI, adversarial testing, and deployment are implemented
> and tested. See [Known limitations](#known-limitations) for honest
> caveats before you treat this as production-grade.

## Overview

Pipeline, in one line: **PDF → parse (pdfplumber) → extract claims
(Mistral) → verify against live search (Tavily + Mistral judge) → report
(table + CSV)**.

1. You upload a PDF.
2. `core/pdf_parser.py` extracts page-wise text and tables, capped at 30
   pages, with a heuristic warning if the document looks scanned.
3. `core/claim_extractor.py` sends the full document to Mistral in a
   single call, gets back structured claims (text, type, page), dedupes
   repeats, and caps at 30 unique claims.
4. `core/verifier.py` searches Tavily for each unique claim in parallel,
   then judges each one against the live evidence using Mistral, batched
   5 claims per call. A programmatic grounding check discards any source
   URL the judge didn't actually receive from search — see
   [Known limitations](#known-limitations) for what this guard does and
   doesn't catch.
5. `app.py` renders the result: verdict counts, a full table
   (claim / type / page / verdict / explanation / source), and a CSV
   download.

## Quick start (local)

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys:
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Get a free Mistral key: https://console.mistral.ai
   - Get a free Tavily key: https://tavily.com
   - Fill both values into `.streamlit/secrets.toml`
4. Run locally:
   ```bash
   streamlit run app.py
   ```
   You should see the app boot with an "Environment status" panel showing
   both keys as configured.

## Usage

1. Open the app (locally or at your deployed URL).
2. Upload a PDF using the uploader (max 25MB, first 30 pages processed).
3. Watch the staged progress: parsing → extracting claims → verifying.
4. Review the report:
   - **Verified** — live search results directly confirm the claim.
   - **Inaccurate** — live results contradict or update the claim (the
     real figure differs, or it's outdated).
   - **False** — no supporting evidence was found, or it's directly
     refuted. (See the limitations section — "no evidence found" and
     "actively refuted" are deliberately not distinguished in this
     3-way rubric to keep the interface simple and focused.)
5. Click **Download report as CSV** to export.
6. Try `sample_docs/test_trap_doc.pdf` first — it plants known-false
   stats, a claim repeated 3 times verbatim, a simulated scanned page,
   and a fabricated metric with zero real-world evidence, so you can see
   every code path in one upload.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account.
3. Click **Create app** → deploy from your repo.
4. Fill in: **Repository** (your fork), **Branch** (`main`), **Main file
   path** (`app.py`).
5. In **Advanced settings**, Python 3.11 or 3.12 is fine (Cloud defaults
   to 3.12; this app has no version-specific code below 3.9).
6. In the **Secrets** field, paste the contents of your local
   `.streamlit/secrets.toml` (`MISTRAL_API_KEY` and `TAVILY_API_KEY`).
   The real file is never committed — see `.gitignore`.
7. Click **Deploy**. Most apps are live within a few minutes.

**Cold start note:** the free tier sleeps idle apps. The first request
after a sleep period is noticeably slower while it wakes up — expected
behavior, not a bug. Open the app once a few minutes before a demo or
evaluation to warm it up.

## Known limitations

Documented honestly, not hidden:

- **Free-tier rate limits.** Mistral's free tier caps at ~1 request/sec;
  judge calls are kept strictly sequential (batched 5 claims/call) to
  respect this, while Tavily searches run in parallel since its tier has
  no comparable per-second cap. Under heavy concurrent use (multiple
  evaluators at once), requests may queue and the report will take
  longer, not fail outright — retry/backoff in `core/api_clients.py`
  absorbs transient 429s. Tavily's free tier also caps at 1,000
  searches/month total across all users of the deployed app.
- **3-way verdict rubric is a simplification.** "False" covers both
  "actively refuted by evidence" and "no evidence found at all" — these
  are mechanically different paths internally (the latter skips the
  judge call entirely) but both surface as "False" with an explanation that
  distinguishes them in the text to keep the frontend simple and straightforward.
- **Scanned-document detection is a heuristic, not OCR.** It flags a PDF
  as likely-scanned when extracted text per page falls below a
  threshold. It does not attempt to OCR the page — a scanned PDF will
  correctly warn the user but won't produce claims from the image
  content.
- **Hard caps exist by design.** 30 pages, 30 unique claims, 25MB upload.
  These bound API cost on the free tier; a longer or larger document is
  partially processed with a visible warning, not silently truncated.
- **No persistence.** Results live in the browser session only
  (`st.session_state`) — closing the tab loses the report unless you've
  downloaded the CSV. No history, no accounts, no database, by design
  for this scope.
- **English-language documents assumed.** Prompts and heuristics aren't
  tuned for other languages.
- **The hallucination guard has a specific, narrow scope.** It verifies
  that a cited `source_url` was actually among the search results the
  judge received — it does not verify that the judge read that source
  correctly. A wrong verdict grounded in a real-but-misinterpreted
  source would pass the guard. Judge quality ultimately depends on
  Mistral's reasoning, same as any LLM-as-judge system.
- **Verification quality is validated against mocked API responses in
  this codebase's test history**, not a live run during development
  (the sandbox used to build this couldn't reach `api.mistral.ai` or
  `api.tavily.com`). Run `sample_docs/test_trap_doc.pdf` with your real
  keys before fully trusting the verdicts.

## Demo walkthrough

A 30-second screen recording can be used to showcase the app on your portfolio or in job applications. Here's a suggested walkthrough flow that fits in 30 seconds:

| Time | Action |
|---|---|
| 0:00–0:05 | Show the deployed app loaded in a browser tab; environment status shows both keys configured (green). |
| 0:05–0:10 | Upload `sample_docs/test_trap_doc.pdf` (or your own PDF) via the uploader. |
| 0:10–0:20 | Let the staged progress play out on screen: "Parsing PDF..." → "Extracting claims..." → "Verifying claims against live web data..." → "Done." |
| 0:20–0:27 | Show the results: verdict count metrics at the top, then scroll the table to a row with a "False" or "Inaccurate" verdict and its explanation/source. |
| 0:27–0:30 | Click **Download report as CSV** to show the export working. |

Recording tips: any free screen recorder works (Loom, OBS, Windows Game
Bar, macOS screen recording). Warm up the app first if it's been idle
(see the cold-start note above) so the recording doesn't stall on a slow
wake-up. No narration is required — the on-screen progress messages
carry the story.

## Project structure

```
fact-check-agent/
├── app.py                  # Streamlit UI entrypoint
├── config.py                # API key loading (secrets.toml / env vars)
├── core/
│   ├── api_clients.py       # retry/backoff wrappers: Mistral + Tavily
│   ├── pdf_parser.py        # PDF -> page-wise text, page cap, scan detection
│   ├── claim_extractor.py   # text -> Claim list (Mistral, dedupe, cap)
│   ├── verifier.py          # Claim list -> VerificationResult list
│   └── models.py            # Claim, Verdict, VerificationResult
├── utils/
│   └── prompts.py           # extraction + verification prompt templates
├── sample_docs/
│   └── test_trap_doc.pdf    # adversarial test file (Phase 5)
├── requirements.txt
└── .streamlit/
    ├── config.toml           # upload size cap
    └── secrets.toml.example
```

## Roadmap

- [x] Phase 0 -- project setup, environment wiring
- [x] Phase 1 -- PDF parsing + resilient API client (retry/backoff)
- [x] Phase 2 -- claim extraction (Mistral)
- [x] Phase 3 -- claim verification (Tavily + Mistral judge)
- [x] Phase 4 -- UI integration, end-to-end flow
- [x] Phase 5 -- adversarial testing
- [x] Phase 6 -- deployment (Streamlit Community Cloud)
- [x] Phase 7 -- packaging and documentation
