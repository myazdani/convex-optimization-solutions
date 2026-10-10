# Coverage contract

## Canonical three-category catalog

[`catalog.json`](catalog.json), validated against [`catalog.schema.json`](catalog.schema.json),
is the canonical organization of this repository:

1. [Examples](../docs/examples/README.md): numerical worked examples from the text,
   separately inventoried with source results and reproduction/data limitations.
2. [Computational exercises](../docs/computational-exercises/README.md): the 16
   existing verified numerical exercise solutions, including five extensions.
3. [All other exercises](../docs/other-exercises/README.md): the remaining 321
   numbered exercises, including theoretical exercises, with independent review required for completion.

The catalog contains all **337 numbered textbook exercises exactly once** across
categories 2 and 3. This is complete numbered-exercise inventory coverage, not
complete solution coverage. Worked examples are a separate content type and do
not affect that count. Their source inventory has its own coverage statement in
[`worked_examples.json`](worked_examples.json).

Statuses are evidence-based:

- `pending`: identified, but no complete reproduction or solution is delivered
- `implemented`: code/notes exist, but verification is incomplete
- `verified`: the stated task, data, and outputs have been checked
- `blocked`: a specific missing input or access issue prevents completion

Do not create an empty solution page for a pending entry or mark a task verified
from importability alone. The manifest links delivered files only when they exist.
The worked-example research inventory is supplemented by
[`example_reproductions.json`](example_reproductions.json) when executed reproductions
are delivered. This keeps research evidence and implementation evidence distinct.

## Completion and independent review

[`solution_progress.json`](solution_progress.json) tracks all 399 items in the
request: the 18 prior verified deliverables and 381 newly requested items. A draft
or a complete author submission stays open until a different reviewer checks the
source, all subparts, assumptions, and correctness against the saved file hash.

[`exercise_solutions.json`](exercise_solutions.json) is the accepted-proof overlay.
The 321 exercises retain their category when completed. Proofs need original,
complete arguments; they do not need fabricated numerical artifacts. When an
exercise requires computation, the executed code/output must accompany its proof.

Each new worked example is classified as `exact` or `method`. The user approved
method reproductions on documented inputs when original textbook inputs are
unavailable. These count as completed reproductions after execution and independent
review, but must set `output_match_verified` to false and explicitly explain the
data differences. Exact reproductions must verify the source's actual result.
The catalog reports these totals separately.

Review records under `reviews/` bind the exact solution bytes (and code/results
bytes for examples) to the reviewer, source/subpart checks, and mathematical
findings. The registration scripts reject self-review, changed files, missing
subparts, and unsubstantiated completion. The
[`preserved numerical baseline`](preserved_numerical_baseline.json) protects the
115 original implementation, proof, output, and source files byte for byte.

## Build and validate

```sh
python scripts/build_catalog.py
python scripts/build_catalog.py --check
python scripts/validate_catalog.py
python -m pytest tests/test_catalog.py tests/test_docs_formatting.py -q
```

The generator reads the source audit's complete numbered-exercise list, the
historical computational inventory's delivered metadata, and the worked-example
source/reproduction inventories, plus the verified exercise-solution overlay. It writes the unified manifest and three catalog
pages. It does not solve exercises, run examples, or create solution stubs.
Validation rejects numbering gaps, duplicate entries, category overlap, broken
local links, unsubstantiated completion statuses, and stale generated indexes.

The [organization verification record](category-verification.json) and
[147-test regression log](category-tests.log) record this catalog/pilot snapshot,
including byte-for-byte preservation of 103 existing numerical files.

## Incremental publication evidence

The [`publications/`](publications/) directory records each verified GitHub batch,
its exact commit, accepted-solution totals, local test log, preserved baseline, and
GitHub math/image rendering checks. These records distinguish the published
snapshot from any newer work still awaiting review.

## Historical computational inventory

`inventory.json`, `inventory.schema.json`, and `source_audit.json` preserve the
original numerical selection and recorded provenance. Their former `excluded`
and `optional` classifications describe that historical selection only. They do
not exclude theoretical work or worked examples from the current catalog. The
55 unimplemented optional numerical extensions are among category 3's 321 entries.
These historical inputs are unchanged so the recorded execution hashes remain valid.

The original CLI `convex-solutions audit` and `run all` concern category 2's
computational implementation subset. `required_complete` means all 11 explicitly
computational exercises were verified; `selected_complete` means all 16 selected
modules were verified. Its `complete` flag combines those two historical checks.
It does not mean the book's 337 exercises or all worked examples are solved.

## Existing numerical verification

`verification.json` records commands, outcomes, environment, and known limits for
the initial numerical delivery. The execution log initially reported pending
verification because status was updated only after tests and review. The appended
post-run line, `status.json`, and the final `run_summary.json` record those checks.
Individual result envelopes retain the inventory hash at their execution time.

`results-refresh.json` and `results-rerun.log` document the subsequent full seed-0
execution that populated each of the 16 exercise pages' result tables. The recorded
output checksums bind tables to that run. `results-tests.log` and `results-tests.xml`
record its 115-test suite; the organization update's checks are separate from this
historical evidence. The new category navigation preserves the original solution
pages, formulas, plots, and numerical artifacts.
