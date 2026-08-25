import json
import sys
from unittest.mock import patch

sys.path.insert(0, ".")
import generate as gen

SAMPLE_JD = "Cloud Support Engineer with AWS and Linux troubleshooting experience."

# These match your ACTUAL retrieve.py output shape: title, category,
# source, skills, text, distance -- no "id" field.
REAL_SHAPED_CHUNKS = [
    {
        "title": "AWS Cloud Infrastructure Support",
        "category": "Cloud Engineering",
        "source": "Point32Health - Linux Systems Administrator",
        "skills": "AWS, EC2, Cloud Security",
        "text": "Supported AWS cloud infrastructure including EC2 instance management, Linux workload administration, security configuration, and operational troubleshooting.",
        "distance": 0.21,
    },
    {
        "title": "Linux Production Troubleshooting",
        "category": "Systems Administration",
        "source": "Point32Health - Linux Systems Administrator",
        "skills": "systemctl, journalctl, Log Analysis, Command-Line Diagnostics",
        "text": "Diagnosed and resolved Linux production issues using command-line troubleshooting tools including systemctl, journalctl, top, df, du, grep, netstat/ss, and log analysis techniques.",
        "distance": 0.24,
    },
]


def fake_response(content):
    return {"message": {"content": content}}


# --- Test 1: formatting includes title, category, source, distance -----
formatted = gen._format_chunks_for_prompt(REAL_SHAPED_CHUNKS)
assert "AWS Cloud Infrastructure Support" in formatted
assert "Category: Cloud Engineering" in formatted
assert "Source: Point32Health" in formatted
assert "Match distance: 0.210" in formatted
assert "Skills: AWS, EC2, Cloud Security" in formatted
print("Test 1 passed: chunk formatting correctly uses title/category/source/distance/skills.")

# --- Test 2: tailoring call works end-to-end with real-shaped chunks ----
good_tailor_json = json.dumps({
    "summary": "Cloud support engineer with AWS and Linux troubleshooting background.",
    "tailored_bullets": [
        {
            "source_title": "AWS Cloud Infrastructure Support",
            "section": "Cloud Engineering",
            "original_text": REAL_SHAPED_CHUNKS[0]["text"],
            "tailored_text": "Supported production AWS infrastructure, managing EC2 instances and resolving operational issues.",
        }
    ],
    "skills_highlighted": ["AWS", "EC2", "Linux"],
})
with patch.object(gen.ollama, "chat", return_value=fake_response(good_tailor_json)):
    result = gen.generate_tailored_resume(SAMPLE_JD, REAL_SHAPED_CHUNKS)
    assert result["tailored_bullets"][0]["source_title"] == "AWS Cloud Infrastructure Support"
print("Test 2 passed: tailoring works correctly against real-shaped chunks, referencing titles not ids.")

# --- Test 3: gap analysis call works end-to-end ---------------------------
good_gap_json = json.dumps({
    "well_covered_requirements": ["AWS experience", "Linux troubleshooting"],
    "gaps": [
        {"requirement": "Customer-facing support experience", "why_its_a_gap": "No excerpt mentions direct customer interaction.", "severity": "medium"}
    ],
})
with patch.object(gen.ollama, "chat", return_value=fake_response(good_gap_json)):
    result = gen.generate_gap_analysis(SAMPLE_JD, REAL_SHAPED_CHUNKS)
    assert len(result["gaps"]) == 1
print("Test 3 passed: gap analysis works correctly against real-shaped chunks.")

# --- Test 4: full generate() saves both files ----------------------------
with patch.object(gen.ollama, "chat", side_effect=[fake_response(good_tailor_json), fake_response(good_gap_json)]):
    result = gen.generate(SAMPLE_JD, REAL_SHAPED_CHUNKS, save=True)
    assert (gen.OUTPUT_DIR / "tailored_resume.json").exists()
    assert (gen.OUTPUT_DIR / "gap_analysis.json").exists()
print("Test 4 passed: generate() saves both output files correctly.")

print("\nAll 4 schema-accuracy tests passed against your real chunk shape.")