"""
generate.py

Step 5 - Generation: local language model drafts a tailored resume + gap analysis.

INPUT (comes from retrieve.py's retrieve_relevant_chunks()):
    - a job description (plain text)
    - a list of retrieved entries, each a dictionary with the keys
      title, category, source, skills, text, and distance
      (this matches exactly what retrieve_relevant_chunks() returns)

OUTPUT:
    - a tailored resume draft, as structured data (step 6 turns this into
      a .docx file)
    - a gap analysis, as structured data (step 6 turns this into a
      markdown report)
    Both are also saved as JSON files in local/output/ so step 6 can pick
    them up without needing to re-run the language model every time.

CORE RULE the language model is instructed to follow:
    It may only rewrite and reframe content that was actually retrieved
    from your master bank. It is never allowed to invent an
    accomplishment, a number, a tool, or a responsibility that isn't
    already in the retrieved text.
"""

import json
import re
import sys
from pathlib import Path

import ollama


# ---------------------------------------------------------------------------
# Configuration - check these values before running anything
# ---------------------------------------------------------------------------

# Must match, character for character, a chat/instruct model you already
# pulled in step 1 (this is separate from EMBEDDING_MODEL in ingest.py /
# retrieve.py - those embed text, this one writes text). Run `ollama list`
# in your terminal and copy the exact name from there - if it shows
# "llama3.2:3b" or "llama3.2:1b" rather than plain "llama3.2", use the
# exact string shown.
MODEL_NAME = "llama3.2"

# Lower temperature keeps the language model close to the source material.
# This is grounded rewriting, not creative writing.
TEMPERATURE = 0.3

# How many times to give the model a chance to correct itself if it
# returns text that isn't valid JSON. Bumped up a bit here since smaller
# models (like llama3.2) are less consistent about following strict JSON
# instructions than larger ones - if you still see repeated failures,
# raise this further or try a larger model.
MAX_JSON_RETRIES = 4

# Where step 5's output gets saved.
OUTPUT_DIR = Path(__file__).parent / "output"

# A "well covered" claim gets flagged for manual review when fewer than
# this fraction of the requirement's own distinctive words actually
# appear in the excerpt being cited. Set deliberately low (0.25): a real,
# correct match still won't repeat most of a requirement's wording, so
# demanding a high overlap produces constant false alarms. What reliably
# separates a real match from a bogus one is whether ANY specific term
# lines up at all - a genuine match usually hits at least one or two, a
# bogus one typically hits zero. Raise this if bad claims slip through;
# lower it if legitimate claims keep getting flagged.
RELEVANCE_WARNING_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TAILORING_SYSTEM_PROMPT = """You are a resume-writing assistant.

You will be given a job description and a set of real, verified excerpts \
pulled from a candidate's actual career history. Your only job is to \
rewrite and reframe the SUPPLIED excerpts so they speak to the job \
description, using the job description's own terminology where it \
genuinely applies.

Hard rules, no exceptions:
- Only use facts, numbers, tools, and outcomes that appear in the supplied \
excerpts. Never invent an accomplishment, metric, tool, or responsibility \
that is not in the supplied text.
- You may rephrase, reorder, and emphasize. You may not fabricate.
- If an excerpt does not fit the job description well, still include it, \
but do not stretch its meaning to imply something it does not say.
- Output ONLY a single valid JSON object. No markdown, no commentary, no \
code fences, no text before or after the JSON object."""

GAP_ANALYSIS_SYSTEM_PROMPT = """You are a hiring-focused analyst.

You will be given a job description and a set of excerpts that were \
retrieved from a candidate's career history because they were judged the \
MOST relevant matches available. Each excerpt includes a match distance \
number - lower means a stronger semantic match. A large distance means \
this was simply the closest thing available, not necessarily a good \
match, and should usually still count as a gap.

Your job is to honestly identify which requirements in the job \
description are NOT well covered by the supplied excerpts, and then give \
an honest numeric estimate of how strong a match this candidate's real, \
retrieved experience is for this specific job.

Hard rules, no exceptions:
- Be honest, not encouraging. If nothing in the excerpts credibly \
supports a requirement, say so plainly, even if something was retrieved \
for it.
- For every single item you list as well covered, you must be able to \
name the exact excerpt title above that supports it, AND quote the \
specific phrase from that excerpt's actual text that supports it - not \
a paraphrase, the real words as they appear. A generic quote about the \
same broad topic is NOT enough - the quote must speak to this specific \
requirement, not just a nearby one. If you cannot point to real, \
specific words in an excerpt that support a requirement, that \
requirement belongs in gaps, not in well_covered_requirements. Do not \
match on a word appearing in both the job description and an excerpt \
if the underlying concepts are actually different things (for example: \
"IAM" in a candidate's background usually means AWS Identity and Access \
Management, cloud permissions - it is NOT the same thing as Red Hat \
Identity Management, a Linux directory service, even though both use \
the word "identity" or the acronym "IAM"). Never list a requirement as \
well covered just because it appeared in the job description.
- Never credit the candidate with holding a specific named \
certification (RHCSA, CCNA, a specific vendor certification, etc.) \
unless a supplied excerpt explicitly names that exact certification. A \
general training program or unrelated skill mention does NOT mean the \
candidate holds a specific certification - if the exact certification \
isn't named in an excerpt, it is a gap, full stop.
- Purely administrative or eligibility items - citizenship or work \
authorization status, language fluency, willingness to travel, on-site \
availability, physical requirements, and similar - are almost never \
addressed by a career history at all. Do not list these as well \
covered, and do not invent a source for them. Only mention one of these \
at all (in either list) if a supplied excerpt explicitly and directly \
addresses that exact item (for example, an active security clearance \
genuinely stated in an excerpt is fair game either way). Otherwise, \
leave that requirement out of both lists entirely rather than guessing.
- The fit score must be grounded in what you actually found, not in how \
the candidate might generally seem. A 9 or 10 should be rare, reserved \
for cases with no high-severity gaps at all. Each high-severity gap \
should pull the score down meaningfully. A candidate who is missing \
several core, explicitly-required skills should score low - 3 or below - \
even if they have strong experience elsewhere. Do not round up out of \
courtesy.
- Output ONLY a single valid JSON object. No markdown, no commentary, no \
code fences, no text before or after the JSON object."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks_for_prompt(chunks):
    """Turns the list of retrieved entries into readable text the language
    model can reference by title."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", f"Untitled entry {i}")
        category = chunk.get("category", "")
        source = chunk.get("source", "")
        skills = chunk.get("skills", "")
        distance = chunk.get("distance")
        text = (chunk.get("text") or "").strip()

        header = f"[{i}] {title}"
        if category:
            header += f" | Category: {category}"
        if source:
            header += f" | Source: {source}"
        if distance is not None:
            header += f" | Match distance: {distance:.3f}"

        body = text
        if skills:
            body += f"\nSkills: {skills}"

        lines.append(f"{header}\n{body}")
    return "\n\n".join(lines)


def _extract_json(text):
    """Small local models sometimes wrap JSON in markdown fences or add a
    stray sentence despite being told not to. This pulls the JSON object
    out of whatever text came back."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text


def _call_ollama_json(system_prompt, user_prompt):
    """Calls the local model and guarantees either a parsed dictionary
    back, or a clear, actionable error - never a silent failure."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = None

    for attempt in range(1, MAX_JSON_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                format="json",
                options={"temperature": TEMPERATURE},
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure the Ollama application "
                f"is running and that the model '{MODEL_NAME}' has been "
                f"pulled (`ollama pull {MODEL_NAME}`). "
                f"Underlying error: {exc}"
            ) from exc

        raw_text = response["message"]["content"]
        cleaned = _extract_json(raw_text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < MAX_JSON_RETRIES:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "That was not valid JSON. Respond again with ONLY "
                        "a single valid JSON object - no markdown fences, "
                        "no extra text before or after it."
                    ),
                })

    raise RuntimeError(
        f"The model did not return valid JSON after {MAX_JSON_RETRIES} "
        f"attempts. Last parse error: {last_error}. Try lowering "
        "TEMPERATURE, or a different model that follows JSON "
        "instructions more reliably."
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_tailored_resume(job_description, chunks):
    """Produces the tailored resume draft as structured data."""
    if not chunks:
        raise ValueError(
            "No chunks were passed in. Run retrieve_relevant_chunks() "
            "first - there is nothing to tailor without retrieved content."
        )

    user_prompt = (
        f"JOB DESCRIPTION:\n{job_description.strip()}\n\n"
        f"RETRIEVED CANDIDATE EXCERPTS:\n"
        f"{_format_chunks_for_prompt(chunks)}\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "summary": string (2-3 sentence professional summary built '
        "only from the supplied excerpts),\n"
        '  "tailored_bullets": [\n'
        "    {\n"
        '      "source_title": string (copy the exact title from the '
        "excerpt list above, e.g. the text right after the [N] marker),\n"
        '      "section": string (use the excerpt\'s Category),\n'
        '      "original_text": string (copy verbatim from the excerpt),\n'
        '      "tailored_text": string (your rewritten version)\n'
        "    }\n"
        "  ],\n"
        '  "skills_highlighted": [string, ...]\n'
        "}"
    )
    return _call_ollama_json(TAILORING_SYSTEM_PROMPT, user_prompt)


def generate_gap_analysis(job_description, chunks):
    """Produces the honest gap analysis, including a fit score, as
    structured data."""
    if not chunks:
        raise ValueError(
            "No chunks were passed in. Run retrieve_relevant_chunks() "
            "first - there is nothing to analyze without retrieved content."
        )

    user_prompt = (
        f"JOB DESCRIPTION:\n{job_description.strip()}\n\n"
        f"RETRIEVED CANDIDATE EXCERPTS:\n"
        f"{_format_chunks_for_prompt(chunks)}\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "well_covered_requirements": [\n'
        "    {\n"
        '      "requirement": string (the specific thing the job '
        "description asks for),\n"
        '      "supporting_excerpt": string (the exact title of the '
        "excerpt above that genuinely supports it - required, not "
        "optional),\n"
        '      "supporting_detail": string (a short, near-verbatim '
        "phrase copied from that excerpt's actual text - the real "
        "words, not a paraphrase - proving the connection is real)\n"
        "    }\n"
        "  ],\n"
        '  "gaps": [\n'
        "    {\n"
        '      "requirement": string (the specific thing the job '
        "description asks for),\n"
        '      "why_its_a_gap": string (why the supplied excerpts do not '
        "credibly cover it),\n"
        '      "severity": "high" | "medium" | "low"\n'
        "    }\n"
        "  ],\n"
        '  "fit_score": integer from 0 to 10 (0 = essentially no overlap '
        "with this job's requirements, 10 = an exceptionally strong, "
        "near-complete match, grounded strictly in the well_covered and "
        "gaps lists above),\n"
        '  "fit_score_reasoning": string (1-2 honest sentences explaining '
        "the number - name the specific factors that kept it from being "
        "higher)\n"
        "}"
    )
    result = _call_ollama_json(GAP_ANALYSIS_SYSTEM_PROMPT, user_prompt)
    result = _sanity_check_fit_score(result)
    result = _validate_well_covered_grounding(result, chunks)
    return result


def _sanity_check_fit_score(gap_result):
    """Small local models sometimes ignore the scoring instructions and
    hand back an inflated number regardless of the gaps it just listed.
    This doesn't silently fix the score - it flags the mismatch so you
    notice it instead of trusting a number that contradicts the model's
    own analysis."""
    score = gap_result.get("fit_score")
    gaps = gap_result.get("gaps", [])
    high_severity_count = sum(1 for g in gaps if g.get("severity") == "high")

    if isinstance(score, (int, float)) and score >= 8 and high_severity_count >= 2:
        gap_result["fit_score_warning"] = (
            f"Flagged for review: the model gave a fit score of {score}/10 "
            f"despite listing {high_severity_count} high-severity gaps. "
            "That's an internal contradiction - treat this score with "
            "skepticism and read the gaps list directly instead."
        )

    return gap_result


def _word_overlap_ratio(quote, source_text):
    """Returns the fraction of meaningful words in `quote` that actually
    appear somewhere in `source_text`. A low ratio is a strong signal
    the quote was made up rather than pulled from the real excerpt."""
    def significant_words(text):
        return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 3]

    quote_words = significant_words(quote)
    if not quote_words:
        # Nothing to check (model left it blank) - don't penalize here,
        # this gets caught separately if it matters.
        return 1.0

    source_words = set(significant_words(source_text))
    matched = sum(1 for w in quote_words if w in source_words)
    return matched / len(quote_words)


# Generic words that show up in almost every job requirement and carry
# no real distinguishing signal (e.g. "experience", "strong", "ability").
# Excluding these stops the requirement-relevance check below from being
# fooled by two sentences that share only filler words in common.
_FILLER_WORDS = {
    "experience", "experienced", "knowledge", "ability", "abilities",
    "skills", "skill", "familiarity", "understanding", "background",
    "strong", "proven", "hands", "solid", "demonstrated", "years",
    "working", "candidate", "candidates", "role", "position", "team",
    "environment", "environments", "requirements", "requirement",
    "required", "preferred", "plus", "must", "have", "having", "with",
    "and", "the", "for", "using", "including", "such", "like",
}


def _significant_terms(text):
    """Pulls out the words in `text` that actually carry meaning for
    matching purposes: recognized acronyms (kept regardless of length,
    e.g. "AWS", "SSH", "RHCSA") plus longer words that aren't generic
    filler."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+/\-]*", text)
    terms = []
    for token in tokens:
        lowered = token.lower()
        if token.isupper() and len(token) >= 2:
            terms.append(lowered)
        elif len(token) > 4 and lowered not in _FILLER_WORDS:
            terms.append(lowered)
    return terms


def _requirement_grounding_ratio(requirement, source_text):
    """Returns the fraction of the REQUIREMENT's own distinctive terms
    that actually appear in the cited excerpt's text. This is the check
    that catches a real, accurately-quoted sentence being used to
    support the wrong requirement - the quote can be 100% real and this
    ratio can still be near zero, because the specific thing being
    claimed (storage, a named certification, a specific tool) simply
    isn't in that excerpt at all."""
    req_terms = _significant_terms(requirement)
    if not req_terms:
        # Requirement was too generic/short to meaningfully check - don't
        # manufacture a false alarm out of nothing.
        return 1.0

    source_lower = source_text.lower()
    matched = sum(1 for term in req_terms if term in source_lower)
    return matched / len(req_terms)


def _looks_like_certification_claim(requirement):
    """Detects requirements that are about holding a specific named
    credential. These get a stricter check than ordinary skill claims,
    because falsely claiming a certification you don't hold is a much
    more serious error than overstating a general skill - it's a
    concrete, verifiable, checkable credential."""
    lowered = requirement.lower()
    cert_signals = [
        "certification", "certifications", "certified", "certificate",
        "rhcsa", "rhce", "ccna", "ccnp", "comptia", "security+",
        "linux+", "network+", "aws certified", "azure certified",
        "cissp", "pmp", "ckad", "cka",
    ]
    return any(signal in lowered for signal in cert_signals)


def _certification_is_genuinely_supported(requirement, source_text):
    """A certification claim only counts as supported if the excerpt
    actually names that specific credential. Sharing a generic word like
    'cloud' or 'network' with a training-program blurb is not evidence
    the candidate holds the certification."""
    source_lower = source_text.lower()

    # Distinctive terms only - strip out the generic scaffolding words
    # that appear in nearly every certification name, since matching on
    # those is exactly how a bogus claim sneaks through.
    generic_cert_words = {
        "certification", "certifications", "certified", "certificate",
        "cloud", "network", "engineer", "administrator", "associate",
        "professional", "practitioner", "data", "storage", "systems",
        "system", "specialist",
    }
    distinctive = [
        term for term in _significant_terms(requirement)
        if term not in generic_cert_words
    ]

    if not distinctive:
        # Nothing distinctive to check against - can't verify it, so
        # treat it as unsupported rather than assuming the best.
        return False

    return any(term in source_lower for term in distinctive)


def _validate_well_covered_grounding(gap_result, chunks):
    """Checks every well_covered claim two ways against the REAL text of
    the excerpt it claims to cite:
      1. Is the quoted detail actually real text from that excerpt
         (catches fabricated quotes)?
      2. Do the requirement's OWN distinctive words actually show up in
         that excerpt (catches a real, accurate quote being misapplied
         to a requirement it doesn't actually address - the far more
         common failure in practice)?
    Either check failing is enough to flag the claim for manual review.
    """
    chunks_by_title = {
        (c.get("title") or "").strip().lower(): c for c in chunks
    }
    warnings = []

    for item in gap_result.get("well_covered_requirements", []):
        requirement = item.get("requirement", "?")
        excerpt_title = (item.get("supporting_excerpt") or "").strip()
        detail = item.get("supporting_detail") or ""
        chunk = chunks_by_title.get(excerpt_title.lower())

        if chunk is None:
            warnings.append(
                f"\"{requirement}\" cites an excerpt titled "
                f"\"{excerpt_title}\", which doesn't match any excerpt "
                "that was actually retrieved. Likely fabricated - treat "
                "this claim as unverified."
            )
            continue

        source_text = " ".join([
            chunk.get("title", ""),
            chunk.get("text", ""),
            chunk.get("skills", ""),
        ])

        quote_score = _word_overlap_ratio(detail, source_text)
        relevance_score = _requirement_grounding_ratio(requirement, source_text)

        if quote_score < 0.5:
            warnings.append(
                f"\"{requirement}\" cites \"{excerpt_title}\", but the "
                f"quoted detail (\"{detail}\") doesn't clearly match that "
                "excerpt's actual text. Possibly fabricated - verify this "
                "one yourself before trusting it."
            )
        elif _looks_like_certification_claim(requirement):
            if not _certification_is_genuinely_supported(requirement, source_text):
                warnings.append(
                    f"CERTIFICATION CLAIM: \"{requirement}\" cites "
                    f"\"{excerpt_title}\", but that excerpt never names "
                    "this specific certification. Do NOT put this on a "
                    "resume unless you actually hold it - claiming a "
                    "credential you don't have is a serious problem, not "
                    "a wording issue."
                )
        elif relevance_score < RELEVANCE_WARNING_THRESHOLD:
            warnings.append(
                f"\"{requirement}\" cites \"{excerpt_title}\" with a real "
                f"quote (\"{detail}\"), but that excerpt doesn't actually "
                "seem to be about this specific requirement - the "
                "specific thing being asked for doesn't show up in that "
                "excerpt at all. Likely a real quote applied to the "
                "wrong requirement - verify this one yourself."
            )

    if warnings:
        gap_result["grounding_warnings"] = warnings

    return gap_result


def generate(job_description, chunks, save=True):
    """Runs both generation calls and, by default, saves the raw output to
    local/output/ so step 6 can build the .docx and markdown files without
    re-running the language model."""
    tailored = generate_tailored_resume(job_description, chunks)
    gaps = generate_gap_analysis(job_description, chunks)
    result = {"tailored_resume": tailored, "gap_analysis": gaps}

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "tailored_resume.json").write_text(
            json.dumps(tailored, indent=2)
        )
        (OUTPUT_DIR / "gap_analysis.json").write_text(
            json.dumps(gaps, indent=2)
        )

    return result


# ---------------------------------------------------------------------------
# Standalone test mode
# ---------------------------------------------------------------------------
# Run this file directly (`python generate.py`) to test generation on its
# own, using sample data shaped exactly like retrieve.py's real output.

if __name__ == "__main__":
    SAMPLE_JOB_DESCRIPTION = """
    Cloud Support Engineer. Looking for someone with hands-on Amazon Web
    Services experience, Linux administration skills, and the ability to
    troubleshoot production issues independently.
    """

    SAMPLE_CHUNKS = [
        {
            "title": "AWS Cloud Infrastructure Support",
            "category": "Cloud Engineering",
            "source": "Point32Health - Linux Systems Administrator",
            "skills": "AWS, EC2, Cloud Security",
            "text": (
                "Supported AWS cloud infrastructure including EC2 instance "
                "management, Linux workload administration, security "
                "configuration, and operational troubleshooting."
            ),
            "distance": 0.21,
        },
        {
            "title": "Linux Production Troubleshooting",
            "category": "Systems Administration",
            "source": "Point32Health - Linux Systems Administrator",
            "skills": "systemctl, journalctl, Log Analysis, Command-Line Diagnostics",
            "text": (
                "Diagnosed and resolved Linux production issues using "
                "command-line troubleshooting tools including systemctl, "
                "journalctl, top, df, du, grep, netstat/ss, and log "
                "analysis techniques."
            ),
            "distance": 0.24,
        },
    ]

    print("Running step 5 in standalone test mode (sample data, not your "
          "real retrieval).")
    print(f"Model: {MODEL_NAME}\n")

    try:
        result = generate(SAMPLE_JOB_DESCRIPTION, SAMPLE_CHUNKS)
    except (RuntimeError, ValueError) as exc:
        print(f"\nStep 5 failed: {exc}")
        sys.exit(1)

    print("Tailored resume summary:")
    print(result["tailored_resume"].get("summary", "(no summary returned)"))
    print(
        f"\n{len(result['tailored_resume'].get('tailored_bullets', []))} "
        "tailored bullets generated."
    )
    print(f"{len(result['gap_analysis'].get('gaps', []))} gaps identified.")
    print(
        f"Fit score: {result['gap_analysis'].get('fit_score', '?')}/10 - "
        f"{result['gap_analysis'].get('fit_score_reasoning', '')}"
    )
    print(f"\nFull output saved to: {OUTPUT_DIR}")