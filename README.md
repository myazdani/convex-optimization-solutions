# Convex Optimization: computational solutions

Original Python/CVXPY implementations of the computational exercises in
Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (Cambridge University
Press, 2004). The official [book page](https://web.stanford.edu/~boyd/cvxbook/)
and [textbook PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) are the
reference. This repository refers to exercise numbers and gives original
explanations; it does not reproduce the textbook or its solution manual.

## Verified scope

**16 runnable solutions, 95 passing tests, 19 generated figures.** The full
source audit covers all 337 numbered exercises and identifies 11 that explicitly
request numerical algorithm execution or implementation. All 11 are implemented
and verified, together with five concrete-data or visualization extras:

- Required: **9.10, 9.30–9.32, 10.14–10.16, 11.21–11.24**
- Extras: **4.3, 5.29, 5.33, 5.34, 11.1**

Start with the [solution index](docs/README.md), which links each original
explanation and recorded result. This snapshot includes the full-size seed-0
experiments, including the 100-variable partitioning SDP, and all required plots.

The [source audit](coverage/source_audit.json) and
[inventory](coverage/inventory.json) make exclusions explicit. **55 additional
optional computational/plotting extensions are catalogued but unimplemented**;
purely analytical exercises, worked examples, and the separate supplementary
exercise collection are outside this scope. This is not a solution set for all
337 textbook exercises.

The [verification report](coverage/verification.json) records the clean-environment
installation, numerical tests, independent mathematical checks, full reproduction,
and figure review. No known failures remain within the selected scope.

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
convex-solutions list
convex-solutions audit
convex-solutions run 9.30 --output artifacts/9.30 --seed 0
convex-solutions run all --output artifacts --seed 0
```

Only implemented modules run. `run all` reports both executed modules and
required/selected coverage; optional unimplemented extensions stay visible in
the inventory. Every run records the seed,
package versions, solver availability, numeric results, execution time, and
paths to generated figures. Individual exercise notes live in `docs/solutions/`.
All randomness uses a local seeded NumPy generator. Source and inventory SHA-256
fingerprints are recorded in each CLI output. Figures use a noninteractive
backend, so scripts also run on headless machines.

## Repository layout

- `coverage/`: source-backed scope audit, manifest schema, completion evidence
- `src/convex_solutions/`: CLI, numerical utilities, chapter implementations
- `docs/solutions/`: derivations, modeling choices, results, and exercise caveats
- `data/`: source provenance and permitted data; generated data labeled clearly
- `tests/`: mathematical checks, numerical regressions, and CLI tests
- `artifacts/`: regenerated outputs, excluded from source control
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
python scripts/validate_inventory.py
python scripts/verify_outputs.py
python scripts/package_inventory.py
python scripts/package_repository.py
```

The ZIP includes source, tests, original notes, provenance, figures, and numerical
results. It excludes virtual environments, build caches, Git internals, and the
textbook PDF. The package script also writes a SHA-256 checksum beside the ZIP.
