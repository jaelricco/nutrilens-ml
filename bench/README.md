# Versioned Benchmarks

Every release runs `nutrilens-ml evaluate --bench vN` and publishes a
scorecard under `docs/scorecards/`. Benchmarks are frozen — adding samples
means bumping to the next version, never editing `vN`.

## Layout

```
bench/
├── README.md          # this file
└── v0/
    ├── manifest.json  # list of {sample_id, path, labels}
    └── README.md      # dataset description, collection dates, known gaps
```

`manifest.json` conforms to `docs/schemas/` (same schema as the training
manifests). The bench runner pulls referenced files from the object store
using the content-hash cache in `nutrilens_ml.data.ingest`.

## Adding a benchmark version

1. Create `bench/vN/` with `manifest.json` and a short `README.md`.
2. Commit it — the bench must live in git so release runs are reproducible.
3. Run `nutrilens-ml evaluate --bench vN` once locally to validate shape.
4. Cite the new version in the PR that promotes it.
