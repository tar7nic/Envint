"""Input sanitisation and output validation. Retrieved text is DATA, never instructions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"
CITE_RE = re.compile(r"\[([A-Z]+-\d+#\d+(?:\.\d+)*)\]")
INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore (all |any )?(previous|prior|above)", r"disregard .{0,30}(context|instruction)",
    r"system (override|prompt)", r"you are now", r"reveal .{0,20}prompt",
)]


def sanitize_chunk_text(text: str) -> str:
    """Neutralise angle brackets so a chunk cannot close the <document> tag or fake a new one."""
    return " ".join(text.replace("<", "(").replace(">", ")").split())


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


@dataclass
class ParsedAnswer:
    text: str
    refused: bool
    cited: list[str] = field(default_factory=list)    # valid chunk ids, in order of appearance
    invalid: list[str] = field(default_factory=list)  # ids not present in the retrieved context


def parse_answer(text: str, allowed_ids: set[str]) -> ParsedAnswer:
    text = (text or "").strip()
    if REFUSAL_TOKEN in text.upper().replace(" ", "_"):
        return ParsedAnswer(text=text, refused=True)
    cited, invalid = [], []
    for cid in CITE_RE.findall(text):
        bucket = cited if cid in allowed_ids else invalid
        if cid not in bucket:
            bucket.append(cid)
    return ParsedAnswer(text=text, refused=False, cited=cited, invalid=invalid)