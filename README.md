# Convex Optimization: examples and exercises

A source-linked study companion to Stephen Boyd and Lieven Vandenberghe,
*Convex Optimization* (Cambridge University Press, 2004), organized in three categories.
The official [book page](https://web.stanford.edu/~boyd/cvxbook/) and
[textbook PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) are the reference.
Explanations and implementations here are original; the book and its solution manual
are not reproduced.

## Start here

1. **[Examples](docs/examples/README.md)**: numerical worked examples in the text,
   with the book's supplied results or plots to reproduce. **2 of 62 catalogued
   studies are reproduced**: [Example 7.4](docs/examples/7.4.md) and
   [Example 8.3](docs/examples/8.3.md); 60 remain pending, with data limits recorded.
2. **[Computational exercises](docs/computational-exercises/README.md)**: the **16
   already-delivered numerical exercise solutions**, with derivations, executable
   Python/CVXPY code, actual result tables, and 19 generated figures.
3. **[All other exercises](docs/other-exercises/README.md)**: the remaining **321
   numbered exercises**, organized by chapter and linked to their source pages.
   Their solutions are **pending**, including theoretical and proof-based work.

The [book index](docs/README.md) provides the same three entry points. The
[canonical category manifest](coverage/catalog.json) accounts for all **337 numbered
exercises exactly once** across categories 2 and 3. Worked examples are tracked
separately in category 1. Complete inventory coverage is not a claim that all
exercise solutions or example reproductions are complete.

## Verified scope

The 16 computational solutions comprise 11 exercises that explicitly request
numerical execution or implementation (**9.10, 9.30–9.32, 10.14–10.16,
11.21–11.24**) and five numerical extensions (**4.3, 5.29, 5.33, 5.34, 11.1**).
Those five remain end-of-chapter exercises in category 2; they are not worked
examples from the text.

Every delivered exercise page retains its existing URL, mathematics, and **actual
executed results in Markdown tables**, generated from recorded JSON. The
[execution record](coverage/results-refresh.json) captures commands, timestamps,
package versions, output checksums, and the full-size seed-0 experiments, including
the 100-variable partitioning SDP. The [verification report](coverage/verification.json)
and [115-test numerical-results regression log](coverage/results-tests.log) document
that executed snapshot. See [coverage documentation](coverage/README.md) for the
current catalog checks and the distinction between inventory and solution coverage.

The [historical source audit](coverage/source_audit.json) and
[computational implementation inventory](coverage/inventory.json) preserve the
original numerical selection and its provenance. Their old exclusions describe
that delivery only; the three-category catalog is now canonical. All 55 previously
unimplemented optional numerical extensions are included among the 321 pending
exercises. The separate supplementary exercise collection is not part of this
catalog of the textbook itself.

## Setup

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Python 3.11+ is supported. Open-source solvers bundled with CVXPY are sufficient;
no commercial solver or MATLAB license is required. For consistent, efficient
runtimes on small dense systems, set `OPENBLAS_NUM_THREADS=1` before running. The validated Linux/Python 3.12 snapshot
records all 35 relevant package versions in `requirements-lock.txt`.
For exact reproduction of that snapshot, install that file before the editable
project. Run in a fresh virtual environment to avoid unrelated package conflicts.

## Run

```sh
python scripts/validate_catalog.py
convex-solutions list
convex-solutions audit
convex-solutions run 9.30 --output artifacts/9.30 --seed 0
convex-solutions run all --output artifacts --seed 0
python scripts/render_results.py
python scripts/render_results.py --check
```

For the computational-exercise reproduction workflow, use `python scripts/reproduce_results.py`
(or `make reproduce`). It executes all 16 full-size experiments, records the
actual command and output checksums, and updates every Markdown result table.

The CLI runs category 2 only. `run all` and `audit` report the historical
required/selected computational subset; their completion flags do not mean that
all book exercises or worked examples are solved. Use `validate_catalog.py` for
the full category inventory. Pending exercises stay visible in category 3. Every run records the seed,
package versions, solver availability, numeric results, execution time, and
paths to generated figures. Individual exercise notes live in `docs/solutions/`.
All randomness uses a local seeded NumPy generator. Source and inventory SHA-256
fingerprints are recorded in each CLI output. Figures use a noninteractive
backend, so scripts also run on headless machines.

`render_results.py` updates only the marked results block in each solution page;
derivations stay intact. Its `--check` mode fails if any displayed table differs
from the recorded run. Run it after rerunning the experiments. The table
renderers never substitute expected values when output is missing. Numerical
residuals, runtimes, and final solver digits can vary across environments.

## Repository layout

- `coverage/catalog.json`: canonical three-category manifest and all 337 numbered exercises
- `coverage/worked_examples.json`: source research for numerical examples in the text
- `coverage/`: historical computational provenance, schemas, and verification evidence
- `docs/examples/`: worked-example catalog and verified reproduction notes
- `docs/computational-exercises/`: index of the 16 delivered numerical exercises
- `docs/other-exercises/`: chapter-indexed catalog of 321 pending exercises
- `examples/`: standalone worked-example reproduction code
- `src/convex_solutions/`: CLI, numerical utilities, chapter implementations
- `docs/solutions/`: derivations, modeling choices, results, and exercise caveats
- `data/`: source provenance and permitted data; generated data labeled clearly
- `tests/`: mathematical checks, numerical regressions, and CLI tests
- `artifacts/`: checked-in executed snapshot, figures, and generated datasets
- `scripts/`: validation and reproducibility helpers

## Numerical standards

Each solution states its convex formulation and checks feasibility or stationarity
as appropriate. Where possible, a separate derivation, closed form, or independent
solver supplies a cross-check. Tests use tolerances appropriate to the solver and
problem conditioning; they do not require bitwise equality across platforms.
Prescribed textbook datasets must be used when available. Synthetic substitutes
must be explicitly labeled and never counted as completing a prescribed-data task.

The upstream textbook, figures, and external datasets retain their original rights. These original code
and notes are provided as a study companion, without an implied license to any
third-party material.

## Rebuild the portable package

```sh
python scripts/validate_catalog.py
python scripts/validate_inventory.py
python scripts/verify_outputs.py
python scripts/package_inventory.py
python scripts/package_repository.py
```

The ZIP includes source, tests, original notes, provenance, figures, and numerical
results. It excludes virtual environments, build caches, Git internals, and the
textbook PDF. The package script also writes a SHA-256 checksum beside the ZIP.
