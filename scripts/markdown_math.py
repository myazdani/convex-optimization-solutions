"""The sole permitted baseline documentation normalization for GitHub math."""
from __future__ import annotations
import hashlib

BASELINE_MARKDOWN_EXCEPTIONS = frozenset({
    'docs/solutions/9.31.md', 'docs/solutions/10.16.md', 'docs/solutions/11.1.md',
})


def normalize_fenced_less_than(document: str) -> str:
    """Escape only literal less-than signs inside fenced math, preserving all else."""
    fence = None
    result = []
    for line in document.splitlines(keepends=True):
        if line.startswith('```'):
            fence = line.strip()[3:] if fence is None else None
        elif fence == 'math':
            line = line.replace('<', r'\lt ')
        result.append(line)
    return ''.join(result)


def validate_baseline_markdown_exception(relative, expected, actual_bytes, evidence):
    """Require independent acceptance and prove an exact, narrow text transform."""
    assert relative in BASELINE_MARKDOWN_EXCEPTIONS, relative
    assert evidence['accepted'] is True and evidence['reviewer'] == 'root_independent_format_review'
    assert evidence['original_sha256'] == expected
    original = evidence['original_text'].encode('utf-8')
    assert hashlib.sha256(original).hexdigest() == expected
    assert evidence['corrected_sha256'] == hashlib.sha256(actual_bytes).hexdigest()
    assert normalize_fenced_less_than(original.decode('utf-8')).encode('utf-8') == actual_bytes
