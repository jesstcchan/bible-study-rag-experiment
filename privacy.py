"""Small, local safeguards for anonymous free-text collection.

The checks intentionally run before text is written to the oTree database or
sent to the language-model provider. They catch common direct identifiers,
but they do not replace clear participant instructions or a disclosure-risk
review of the final dataset.
"""

from __future__ import annotations

import re


IDENTIFIER_PATTERNS = (
    (
        "an email address",
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "a web address",
        re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE),
    ),
    (
        "a phone number",
        re.compile(r"(?<![\w:])(?:\+?\d[\s().-]*){7,}(?!\w)"),
    ),
    (
        "a social-media handle",
        re.compile(r"(?<!\w)@[A-Z0-9_]{2,}\b", flags=re.IGNORECASE),
    ),
    (
        "a phrase that may introduce identifying information",
        re.compile(
            r"\b(?:my\s+(?:full\s+)?name\s+is|i\s+am\s+called|"
            r"my\s+(?:e-?mail|phone(?:\s+number)?|address)\s+is|"
            r"contact\s+me\s+(?:at|on))\b",
            flags=re.IGNORECASE,
        ),
    ),
)


def potential_identifier_type(value: object) -> str | None:
    """Return a short description if text contains an obvious identifier."""
    text = str(value or "")
    for description, pattern in IDENTIFIER_PATTERNS:
        if pattern.search(text):
            return description
    return None


def privacy_error(value: object) -> str | None:
    """Return a participant-facing validation error, or ``None``."""
    identifier_type = potential_identifier_type(value)
    if identifier_type is None:
        return None
    return (
        f"Your response appears to contain {identifier_type}. "
        "Please remove names, contact details, links, usernames, addresses, "
        "and other information that could identify you or another person."
    )
