# Official numerical inputs and licensing

These JSON files contain numeric matrices converted from the official CVXOPT
book-example datasets, or numeric constants extracted from its example pages.
Each has a sibling provenance file identifying the original source URL, original
SHA-256 when applicable, transformed SHA-256, and the precise conversion.

The upstream CVXOPT notice credits M. Andersen and L. Vandenberghe (2012–2026),
L. Vandenberghe (2010–2011), and J. Dahl and L. Vandenberghe (2004–2009).
The reused inputs are provided under GPL-3.0-or-later, with the license preserved
in LICENSE-CVXOPT-DATA.txt. See https://cvxopt.org/copyright.html.
New Python implementations are independently authored; upstream program bodies
are not included. Data licensing does not assert a different license for other
repository files.

Run `scripts/prepare_worked_example_data.py` with the directory of downloaded,
hash-pinned official data files to repeat the binary conversions. The restricted
reader permits only numeric cvxopt.base.matrix construction, forbids other pickle
globals and persistent IDs, preserves column-major shape, and emits plain JSON.
It never executes arbitrary downloaded Python code.

Important source differences:

- regsel.py and placement.py explicitly say their inputs differ from the book.
- placement.bin contains 28 edges; the book's instance has 27.
- logreg.bin contains 100 observations; both official scripts also specify 100.
- robls.bin is reused unchanged, but the book's Tikhonov delta is 1 whereas both
  official scripts use 0.1. The implementation executes both.
- Newly generated random arrays are separately identified by their NumPy PCG64
  seed in each result. Equal integer seeds do not imply MATLAB RNG equivalence.
