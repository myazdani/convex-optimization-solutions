# Coverage contract

The canonical inventory is `inventory.json`, validated against
`inventory.schema.json`. A source audit is `in_progress` until the entire book
has been checked. Exercises are `required` when the book explicitly requests
computation and `optional` when a computational illustration is a useful addition.

Statuses are evidence-based:

- `pending`: identified but no complete implementation
- `implemented`: code and original explanation exist; verification pending
- `verified`: requested numerical work and data match the exercise, tests pass,
  and figures/results have been inspected
- `blocked`: a concrete missing input/access issue prevents completion
- `not_applicable`: reviewed and excluded, with a stated reason

Verification evidence is recorded in `coverage/verification.json` separately
from the source audit. It includes command, outcome, environment, and known
limitations. Do not mark a task verified from importability alone. Run
`convex-solutions audit` to compare claimed inventory status with available code.

## Verified snapshot

This snapshot verifies 11 required exercises and five selected extras. The 55
unselected optional records retain pending status and are explicitly outside the
implemented scope. `required_complete` concerns the 11 required exercises;
`selected_complete` concerns all 16 selected modules; `complete` requires both.

The execution log initially reports pending verification because status is only
updated after numerical tests and review. Its appended post-run line,
`verification.json`, `status.json`, and the final `run_summary.json` record the
completed checks. Individual result envelopes retain the inventory hash at the
time of execution.

## Fresh results embedded in Markdown

`results-refresh.json` and `results-rerun.log` document the later full seed-0
execution used to populate every solution page's generated results section.
This record is separate from the historical initial verification above. The
recorded JSON output checksums bind the tables to their source run; the tests
regenerate every section and reject stale Markdown. `results-tests.log` and
`results-tests.xml` record the corresponding final test suite.
