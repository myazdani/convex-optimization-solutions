# Implementing an exercise

1. Verify the exact exercise in the official book and its entry in the inventory.
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

Do not edit another worker's chapter without coordination. Integration owns the
CLI, shared helpers, manifest schema, README, and aggregate tests. Inventory
research owns the initial `coverage/inventory.json` source audit.

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
