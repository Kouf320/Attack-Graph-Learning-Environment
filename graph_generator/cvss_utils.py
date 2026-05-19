"""
CVSS v3.1 vector parsing helpers.

This is a self-contained copy of the helper used by the original ViolenceLang
``violence_model_generator.py`` so the integrated generator has no dependency on
files living deep inside ``mal/ViolenceLang/ViolenceGenerator``.
"""

from __future__ import annotations

CVSS_METRICS = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "S": "Scope",
    "C": "Confidentiality",
    "I": "Integrity",
    "A": "Availability",
    "E": "Exploit Code Maturity",
    "RL": "Remediation Level",
    "RC": "Report Confidence",
    "CR": "Confidentiality Requirement",
    "IR": "Integrity Requirement",
    "AR": "Availability Requirement",
    "MAV": "Modified Attack Vector",
    "MAC": "Modified Attack Complexity",
    "MPR": "Modified Privileges Required",
    "MUI": "Modified User Interaction",
    "MS": "Modified Scope",
    "MC": "Modified Confidentiality",
    "MI": "Modified Integrity",
    "MA": "Modified Availability",
}


def parse_cvss(cvss_string: str) -> dict:
    """Parse a CVSS 3.1 vector string into ``{full metric name: value}``.

    Example
    -------
    >>> parse_cvss("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")["Attack Vector"]
    'N'
    """
    if cvss_string is None:
        return {}

    if cvss_string.startswith("CVSS:3.1/"):
        cvss_string = cvss_string[len("CVSS:3.1/"):]
    elif cvss_string.startswith("CVSS:3.0/"):
        cvss_string = cvss_string[len("CVSS:3.0/"):]

    parsed = {}
    for pair in cvss_string.split("/"):
        if ":" not in pair:
            continue
        metric, value = pair.split(":", 1)
        parsed[CVSS_METRICS.get(metric, "Unknown Metric")] = value
    return parsed


def is_valid_cvss31(cvss_string: str) -> bool:
    """Light structural validation of a CVSS 3.x base vector.

    Checks the eight mandatory base metrics are present with allowed values.
    Does not attempt to score; the RL environment recomputes severities from
    its own database, so we only guard against malformed strings.
    """
    if not isinstance(cvss_string, str) or not cvss_string:
        return False
    metrics = parse_cvss(cvss_string)
    required = {
        "Attack Vector": {"N", "A", "L", "P"},
        "Attack Complexity": {"L", "H"},
        "Privileges Required": {"N", "L", "H"},
        "User Interaction": {"N", "R"},
        "Scope": {"U", "C"},
        "Confidentiality": {"N", "L", "H"},
        "Integrity": {"N", "L", "H"},
        "Availability": {"N", "L", "H"},
    }
    for name, allowed in required.items():
        if metrics.get(name) not in allowed:
            return False
    return True
