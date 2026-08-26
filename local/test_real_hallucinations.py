import json
import sys
from unittest.mock import patch

sys.path.insert(0, ".")
import generate as gen

JD = "Linux system resource and package management. Network-attached storage integration. Experience with ticketing systems. RHCSA certification. Network Appliance Certified Data Administrator On-Tap. AWS Certified Cloud Practitioner. Fluent written English. Must be a U.S. Citizen or Permanent Resident."

# Your REAL master_bank entries, verbatim.
REAL_CHUNKS = [
    {
        "title": "Linux, Security, and Networking Skill Summary",
        "category": "Skills Summary",
        "source": "Aggregated technical skills",
        "skills": "Linux, RHEL, Ubuntu, Debian, DNS, HTTP/HTTPS, SSH, NAT Gateway, Routing Tables, Security Groups, Load Balancing, Firewalls, IAM Least-Privilege",
        "text": "Deep, hands-on Linux systems administration experience combined with practical networking (DNS, routing, load balancing, firewalls) and a security-first mindset applied specifically to IAM least-privilege design and enforcement.",
        "distance": 0.30,
    },
    {
        "title": "Cloud and DevOps Engineer Training Program",
        "category": "Training",
        "source": "Utrains, 2025-Present",
        "skills": "AWS, Linux, Terraform, Docker, Kubernetes, Ansible, GitHub Actions, CI/CD",
        "text": "Completed hands-on training focused on AWS cloud architecture, Linux administration, Terraform, Docker, Kubernetes, Ansible, GitHub Actions, CI/CD pipelines, networking, monitoring, and production DevOps workflows. Built and deployed cloud infrastructure projects using Infrastructure as Code, containerization, and automation as part of the program.",
        "distance": 0.32,
    },
    {
        "title": "Cloud and Infrastructure-as-Code Skill Summary",
        "category": "Skills Summary",
        "source": "Aggregated technical skills",
        "skills": "AWS, EC2, VPC, IAM, Route 53, S3, EBS, EFS, RDS, DynamoDB, ECS, CloudFront, CloudWatch, CloudTrail, API Gateway, SQS, SNS, Auto Scaling, Elastic Load Balancer, Terraform, Terraform Modules, Remote State Management, Ansible",
        "text": "Broad hands-on experience across core AWS services (compute, storage, networking, database, messaging) and Infrastructure as Code tooling, with an emphasis on Terraform module design and remote state management for team-based infrastructure workflows.",
        "distance": 0.35,
    },
]

# EXACTLY what your model produced in the real run.
real_bad_output = json.dumps({
    "well_covered_requirements": [
        {   # LEGIT - should NOT be flagged
            "requirement": "Linux system resource and package management",
            "supporting_excerpt": "Linux, Security, and Networking Skill Summary",
            "supporting_detail": "Linux systems administration experience combined with practical networking",
        },
        {   # BAD - storage never mentioned in that excerpt
            "requirement": "Network-attached storage integration",
            "supporting_excerpt": "Linux, Security, and Networking Skill Summary",
            "supporting_detail": "Linux systems administration experience combined with practical networking",
        },
        {   # BAD - ticketing systems never mentioned
            "requirement": "Experience with ticketing systems",
            "supporting_excerpt": "Cloud and Infrastructure-as-Code Skill Summary",
            "supporting_detail": "Broad hands-on experience across core AWS services and Infrastructure as Code tooling",
        },
        {   # BAD - fabricated source
            "requirement": "Fluent in spoken and written English",
            "supporting_excerpt": "None",
            "supporting_detail": "None",
        },
        {   # BAD - fabricated source
            "requirement": "Must be a U.S. Citizen or Permanent Resident",
            "supporting_excerpt": "None",
            "supporting_detail": "None",
        },
        {   # BAD - RHCSA is not in the bank
            "requirement": "Redhat certifications: RHCSA (RH294)",
            "supporting_excerpt": "Cloud and DevOps Engineer Training Program",
            "supporting_detail": "Completed hands-on training focused on AWS cloud architecture, Linux administration, Terraform, Docker, Kubernetes, Ansible, GitHub Actions, CI/CD pipelines, networking, monitoring, and production DevOps workflows",
        },
        {   # BAD - NetApp cert is not in the bank
            "requirement": "Network Appliance: Certified Data Administrator - On-Tap, Cloud and Storage Engineer",
            "supporting_excerpt": "Cloud and DevOps Engineer Training Program",
            "supporting_detail": "Completed hands-on training focused on AWS cloud architecture, Linux administration, Terraform, Docker, Kubernetes, Ansible, GitHub Actions, CI/CD pipelines, networking, monitoring, and production DevOps workflows",
        },
    ],
    "gaps": [],
    "fit_score": 6,
    "fit_score_reasoning": "Some relevant experience.",
})

with patch.object(gen.ollama, "chat", return_value={"message": {"content": real_bad_output}}):
    result = gen.generate_gap_analysis(JD, REAL_CHUNKS)

warnings = result.get("grounding_warnings", [])
print(f"Caught {len(warnings)} of the 6 known-bad claims:\n")
for w in warnings:
    print(f"  - {w}\n")

flagged_text = " ".join(warnings)

checks = {
    "Network-attached storage (real quote, wrong topic)": "storage" in flagged_text.lower(),
    "Ticketing systems (real quote, wrong topic)": "ticketing" in flagged_text.lower(),
    "Fluent English (fabricated source)": "English" in flagged_text,
    "US Citizen (fabricated source)": "Citizen" in flagged_text,
    "RHCSA cert (not in bank)": "RHCSA" in flagged_text,
    "NetApp cert (not in bank)": "Network Appliance" in flagged_text,
}

print("=" * 60)
for name, caught in checks.items():
    print(f"  {'CAUGHT' if caught else 'MISSED'}  {name}")
print("=" * 60)

legit_flagged = "Linux system resource and package management" in flagged_text
print(f"\n  {'FALSE ALARM' if legit_flagged else 'CORRECT'}  Legitimate Linux claim not flagged")

# A certification the candidate ACTUALLY holds must not be flagged.
REAL_CERT_CHUNK = [{
    "title": "AWS Certified Cloud Practitioner",
    "category": "Certification",
    "source": "Amazon Web Services, 2026",
    "skills": "AWS Fundamentals, Cloud Concepts",
    "text": "Earned the AWS Certified Cloud Practitioner certification, validating foundational knowledge of AWS Cloud services, architecture, security, and pricing.",
    "distance": 0.15,
}]
real_cert_output = json.dumps({
    "well_covered_requirements": [{
        "requirement": "AWS Certified Cloud Practitioner",
        "supporting_excerpt": "AWS Certified Cloud Practitioner",
        "supporting_detail": "Earned the AWS Certified Cloud Practitioner certification",
    }],
    "gaps": [], "fit_score": 8, "fit_score_reasoning": "Good.",
})
with patch.object(gen.ollama, "chat", return_value={"message": {"content": real_cert_output}}):
    cert_result = gen.generate_gap_analysis("AWS Certified Cloud Practitioner required.", REAL_CERT_CHUNK)
real_cert_flagged = "grounding_warnings" in cert_result
print(f"  {'FALSE ALARM' if real_cert_flagged else 'CORRECT'}  Genuinely-held AWS cert not flagged")

assert all(checks.values()), "Some known-bad claims were missed"
assert not legit_flagged, "The legitimate claim was incorrectly flagged"
assert not real_cert_flagged, "A genuinely-held certification was incorrectly flagged"
print("\nALL PASSED: every real hallucination caught, no false alarms.")