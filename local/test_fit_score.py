import json
import sys
from unittest.mock import patch

sys.path.insert(0, ".")
import generate as gen

SAMPLE_JD = "Cloud Support Engineer with AWS and Linux troubleshooting experience."
SAMPLE_CHUNKS = [
    {"title": "AWS Cloud Infrastructure Support", "category": "Cloud Engineering",
     "source": "Point32Health", "skills": "AWS, EC2", "text": "Supported AWS infrastructure.", "distance": 0.21},
]


def fake_response(content):
    return {"message": {"content": content}}


# --- Test 1: honest low score, no high-severity gaps -> no warning ------
honest_low = json.dumps({
    "well_covered_requirements": [{"requirement": "AWS", "supporting_excerpt": "AWS Cloud Infrastructure Support"}],
    "gaps": [{"requirement": "5 years experience", "why_its_a_gap": "Not stated.", "severity": "medium"}],
    "fit_score": 6,
    "fit_score_reasoning": "Strong AWS overlap but experience length is unclear.",
})
with patch.object(gen.ollama, "chat", return_value=fake_response(honest_low)):
    result = gen.generate_gap_analysis(SAMPLE_JD, SAMPLE_CHUNKS)
    assert result["fit_score"] == 6
    assert "fit_score_warning" not in result
print("Test 1 passed: reasonable score with no high-severity gaps triggers no warning.")

# --- Test 2: inflated score contradicted by multiple high gaps ---------
inflated = json.dumps({
    "well_covered_requirements": [{"requirement": "AWS", "supporting_excerpt": "AWS Cloud Infrastructure Support"}],
    "gaps": [
        {"requirement": "Java", "why_its_a_gap": "Not present.", "severity": "high"},
        {"requirement": "Kubernetes", "why_its_a_gap": "Not present.", "severity": "high"},
    ],
    "fit_score": 9,
    "fit_score_reasoning": "Looks like a great fit overall.",
})
with patch.object(gen.ollama, "chat", return_value=fake_response(inflated)):
    result = gen.generate_gap_analysis(SAMPLE_JD, SAMPLE_CHUNKS)
    assert "fit_score_warning" in result
    assert "9/10" in result["fit_score_warning"]
    assert "2 high-severity" in result["fit_score_warning"]
print("Test 2 passed: inflated score with 2+ high-severity gaps correctly flagged.")

# --- Test 3: single high-severity gap with score 9 should NOT warn ------
# (the threshold is 2+, one high-severity gap alone isn't necessarily a contradiction)
one_high = json.dumps({
    "well_covered_requirements": [{"requirement": "AWS", "supporting_excerpt": "AWS Cloud Infrastructure Support"}],
    "gaps": [{"requirement": "Java", "why_its_a_gap": "Not present.", "severity": "high"}],
    "fit_score": 9,
    "fit_score_reasoning": "Strong overall match aside from one language gap.",
})
with patch.object(gen.ollama, "chat", return_value=fake_response(one_high)):
    result = gen.generate_gap_analysis(SAMPLE_JD, SAMPLE_CHUNKS)
    assert "fit_score_warning" not in result
print("Test 3 passed: single high-severity gap with high score correctly does not over-trigger.")

print("\nAll fit-score tests passed.")