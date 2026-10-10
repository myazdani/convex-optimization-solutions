# Adding solutions and reproductions

## Choose the category and deliverable

Start with the [canonical catalog](../coverage/catalog.json) and the official
source page. A pending catalog entry is not an invitation to create a placeholder
solution page.

- For a worked example in the text, add original reproduction notes under
  `docs/examples/`, standalone implementation under `examples/`, and executed
  artifacts under `artifacts/examples/`. Preserve its source identity and describe
  whether the book's exact inputs are available. Update `coverage/example_reproductions.json`
  only with tested implementation evidence. Do not relabel a synthetic illustration
  as an exact reproduction of unknown textbook data.
- For a numbered theoretical exercise, give an original, complete proof or derivation
  with all hypotheses and subparts checked. Code and plots are required only when
  they support the requested mathematics; they do not replace a proof.
- For a numbered computational exercise, follow the implementation contract below.
  The existing 16 solution paths remain under `docs/solutions/`.

After adding verified work, update the relevant source/status records, regenerate
with `python scripts/build_catalog.py`, and run `python scripts/validate_catalog.py`.
The catalog's current 16/321 organization is a delivery snapshot; adding a solution
requires deliberately updating its status and coverage checks rather than claiming
that a new file automatically completes an entry.

## Computational exercise contract

1. Verify the exact exercise in the official book and its category/source record.
2. Create `src/convex_solutions/chapters/chNN/ex_N_NN.py` and chapter `__init__.py`.
3. Expose `solve(output_dir: Path | None = None, seed: int = 0) -> dict`. Return
   serializable metrics/data (NumPy values are supported by the CLI serializer).
   Save generated plots under `output_dir` only when it is provided. The function
   must not write elsewhere, access the network, or modify global random state.
4. Add original notes in `docs/solutions/N.NN.md`: source, concise task
   description, mathematics, implementation choices, results, validation, and
   limitations. Do not copy full problem statements or the official solutions.
5. Add `tests/test_ex_N_NN.py`. Test the mathematical claim and relevant residuals,
   not just the return type or solver status. Include a deterministic reference
   or independent formulation wherever possible. Bound runtime; mark long tests.
6. Report requested parts that remain open. An illustrative experiment is not a
   complete solution if data, constraints, dimensions, or required subparts differ.

Shared helpers:

- `convex_solutions.numerics.solve_cvxpy(problem, ...)` selects an installed
  open-source solver and rejects failed/nonoptimal status.
- `convex_solutions.plotting.pyplot()` configures a headless plotting backend.
- `convex_solutions.io.write_json(path, value)` serializes NumPy arrays/scalars.
- `convex_solutions.provenance.file_sha256(path)` checks stored data.

Keep shared CLI/solver helpers and provenance checks consistent. The historical
computational inventory and `src/convex_solutions/` source tree are hashed in the
16 existing execution envelopes. Worked-example pilots live outside that package
so adding examples does not invalidate the delivered exercises' provenance.
If the existing numerical source changes, rerun the affected reproduction and
validation workflow before publishing new results.

## Executed results in the notes

Each solution includes one `BEGIN GENERATED RESULTS` / `END GENERATED RESULTS`
HTML-comment pair. Run `python scripts/reproduce_results.py` to execute the full
seed-0 profile, record its provenance, and fill those blocks from
`artifacts/*/results.json`. To refresh only the tables from existing outputs,
run `python scripts/render_results.py`.
Use `python scripts/render_results.py --check` before publishing: it compares the
entire displayed result section with freshly rendered tables. Extend the chapter
renderers in `scripts/result_sections_early.py` or `scripts/result_sections_later.py`
when adding metrics. They must compute their tables from the execution envelope,
not copy expected numeric values into text. Preserve the artifact JSON alongside
the Markdown so readers can inspect unrounded values and execution provenance.

## GitHub Markdown mathematics

Use GitHub's dollar/backtick pairs for inline math, as in $`x^2`$, and fenced
`math` blocks for display equations. Keep
inline formulas on one source line. Escaped parenthesis/bracket math delimiters
are not supported by GitHub's Markdown renderer. In table formulas, use
`\vert` or `\Vert` instead of pipe characters so they cannot split columns.
Use `\mathrm{diag}`, `\mathrm{rank}`, and similar upright labels for named
operators: GitHub currently rejects the `operatorname` macro.

See [GitHub's math formatting guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
Run `python -m pytest tests/test_docs_formatting.py -q`, then inspect the rendered
GitHub Preview, including matrices, aligned equations, inline formulas, and
tables. The source checks do not replace a rendering check.
