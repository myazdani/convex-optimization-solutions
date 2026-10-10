# Data provenance

Numerical exercises either use constants stated by the exercise, deterministic
generated instances when the exercise allows them, or an upstream dataset.
Never silently substitute a generated dataset for prescribed upstream data.

For each stored external dataset create a sibling `<filename>.provenance.json`:

```json
{
  "id": "exercise-6.xx-example",
  "kind": "upstream",
  "source_url": "https://verified-original-source.example/data",
  "source_title": "Original dataset title",
  "retrieved_at": "2026-10-10T00:00:00Z",
  "original_filename": "example.csv",
  "sha256": "64 hexadecimal characters",
  "rights": "Link to license or note that redistribution has not been authorized",
  "transformations": []
}
```

Generated instances record `kind: "generated"`, a generator description, seed,
and dimensions in the solution output. Fixed numbers taken from an exercise
record `kind: "textbook_constants"`, exercise ID, and source page. Do not save
textbook scans or solution-manual content here. Keep runtime offline; data
acquisition is an explicit preparation step with verified provenance.
