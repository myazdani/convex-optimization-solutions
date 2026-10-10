# Coverage contract

## Canonical three-category catalog

[`catalog.json`](catalog.json), validated against [`catalog.schema.json`](catalog.schema.json),
is the canonical organization of this repository:

1. [Examples](../docs/examples/README.md): numerical worked examples from the text,
   separately inventoried with source results and reproduction/data limitations.
2. [Computational exercises](../docs/computational-exercises/README.md): the 16
   existing verified numerical exercise solutions, including five extensions.
3. [All other exercises](../docs/other-exercises/README.md): the remaining 321
   numbered exercises, including theoretical exercises; all are pending.

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

## Build and validate

```sh
python scripts/build_catalog.py
python scripts/build_catalog.py --check
python scripts/validate_catalog.py
python -m pytest tests/test_catalog.py tests/test_docs_formatting.py -q
```

The generator reads the source audit's complete numbered-exercise list, the
historical computational inventory's delivered metadata, and the worked-example
source/reproduction inventories. It writes the unified manifest and three catalog
pages. It does not solve exercises, run examples, or create solution stubs.
Validation rejects numbering gaps, duplicate entries, category overlap, broken
local links, unsubstantiated completion statuses, and stale generated indexes.

The [organization verification record](category-verification.json) and
[147-test regression log](category-tests.log) record this catalog/pilot snapshot,
including byte-for-byte preservation of 103 existing numerical files.

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
