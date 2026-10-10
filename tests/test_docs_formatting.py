"""Guard the Markdown conventions used by GitHub's math renderer.

These source checks supplement, rather than replace, visual GitHub Preview QA.
"""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
INLINE_MATH = re.compile(r"\$`([^`\n]+)`\$")


def prose_lines(path):
    """Skip fenced code/math while rejecting accidentally unclosed fences."""
    fence = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("```"):
            if fence is None:
                fence = number
            else:
                assert line == "```", f"{path}:{number}: nested/unclosed fence"
                fence = None
            continue
        if fence is None:
            yield number, line
    assert fence is None, f"{path}:{fence}: unclosed fenced block"


def test_github_math_delimiters():
    for path in MARKDOWN:
        assert r"\operatorname" not in path.read_text(), (
            f"{path}: GitHub rejects operatorname; use upright math notation"
        )
        for number, line in prose_lines(path):
            # Math uses $`...`$ inline and ```math blocks. Backticks prevent
            # underscores, asterisks, and backslashes becoming Markdown syntax.
            remaining = INLINE_MATH.sub("", line)
            remaining = re.sub(r"`[^`]*`", "", remaining)
            assert not re.search(r"\\[()\[\]]", remaining), (
                f"{path}:{number}: unsupported escaped math delimiter"
            )
            assert "$" not in remaining, (
                f"{path}:{number}: unmatched or nonstandard math delimiter"
            )


def test_github_tables_preserve_columns():
    for path in MARKDOWN:
        expected = None
        for number, line in prose_lines(path):
            if not line.startswith("|"):
                expected = None
                continue
            # Even code spans do not protect raw pipes from GFM table parsing.
            # Use \\vert / \\Vert inside table math instead.
            for formula in INLINE_MATH.findall(line):
                assert "|" not in formula, f"{path}:{number}: pipe in table math"
            columns = len(re.split(r"(?<!\\)\|", line)) - 2
            if expected is None:
                expected = columns
            assert columns == expected, f"{path}:{number}: inconsistent table columns"


def test_markdown_contains_no_hidden_control_characters():
    for path in MARKDOWN:
        for offset, char in enumerate(path.read_text()):
            assert ord(char) >= 32 or char in '\n\r\t', (
                f'{path}: hidden control byte {ord(char)} at character {offset}; '
                'check accidentally interpreted LaTeX escapes'
            )
            assert ord(char) != 127, f'{path}: DEL control byte at character {offset}'
