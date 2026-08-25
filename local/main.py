"""
run_step5.py

A way to test everything built so far, together: paste a job
description, this retrieves the most relevant entries from your
experience bank (retrieve.py) and then generates a tailored resume
draft plus a gap analysis from them (generate.py).

This is a temporary test script, not your final product -- step 7
replaces this with the real command-line tool (main.py) that also
writes out the finished .docx and markdown files. Right now, this
script exists so you can see, in your terminal, that steps 4 and 5
actually work correctly together before building anything further
on top of them.
"""

import sys

from retrieve import retrieve_relevant_chunks
from generate import generate


def main():
    print("Paste a job description below.")
    print("When finished, press Enter, then press Control-D to submit.\n")

    job_description_text = sys.stdin.read().strip()

    if not job_description_text:
        print("No job description was entered. Exiting.")
        sys.exit(1)

    print("\nSearching the experience bank ...")
    chunks = retrieve_relevant_chunks(job_description_text)
    print(f"Found {len(chunks)} relevant entries. Generating tailored content ...")
    print("(this can take a little while on a local model -- it's making two")
    print("separate calls to the language model, one per output)\n")

    try:
        result = generate(job_description_text, chunks)
    except (RuntimeError, ValueError) as exc:
        print(f"\nGeneration failed: {exc}")
        sys.exit(1)

    tailored = result["tailored_resume"]
    gaps = result["gap_analysis"]

    print("=" * 70)
    print("TAILORED RESUME DRAFT")
    print("=" * 70)
    print(f"\nSummary:\n{tailored.get('summary', '(none returned)')}\n")
    print("Tailored bullets:")
    for bullet in tailored.get("tailored_bullets", []):
        print(f"\n  - [{bullet.get('section', '?')}] {bullet.get('source_title', '?')}")
        print(f"    {bullet.get('tailored_text', '(no text returned)')}")

    skills = tailored.get("skills_highlighted", [])
    print(f"\nSkills highlighted: {', '.join(skills) if skills else '(none returned)'}")

    print("\n" + "=" * 70)
    print("GAP ANALYSIS")
    print("=" * 70)

    well_covered = gaps.get("well_covered_requirements", [])
    print("\nWell covered:")
    for item in well_covered:
        requirement = item.get("requirement", "?")
        supporting = item.get("supporting_excerpt", "(no excerpt named - suspicious)")
        detail = item.get("supporting_detail", "")
        print(f"  + {requirement}")
        print(f"    from: {supporting}")
        if detail:
            print(f"    quote: \"{detail}\"")
    if not well_covered:
        print("  (none returned)")

    grounding_warnings = gaps.get("grounding_warnings", [])
    if grounding_warnings:
        print("\n  ⚠ GROUNDING WARNINGS - the following 'well covered' claims")
        print("    look questionable and are worth checking by hand:")
        for w in grounding_warnings:
            print(f"    - {w}")

    gap_list = gaps.get("gaps", [])
    print("\nGaps:")
    for gap in gap_list:
        severity = gap.get("severity", "?").upper()
        print(f"  - [{severity}] {gap.get('requirement', '?')}")
        print(f"    {gap.get('why_its_a_gap', '')}")
    if not gap_list:
        print("  (none returned)")

    print("\n" + "-" * 70)
    fit_score = gaps.get("fit_score", "?")
    print(f"FIT SCORE: {fit_score}/10")
    print(f"  {gaps.get('fit_score_reasoning', '(no reasoning returned)')}")
    if "fit_score_warning" in gaps:
        print(f"\n  ⚠ {gaps['fit_score_warning']}")
    print(
        "\n  (This reflects how well your retrieved experience matches "
        "this job description on paper - not a guarantee about the "
        "actual hiring outcome, which depends on plenty the model can't see.)"
    )
    print("-" * 70)

    print(f"\nFull raw output saved to: output/tailored_resume.json and output/gap_analysis.json")


if __name__ == "__main__":
    main()