# Labeled street-scene reference index

Inference-only sidecar index for `POST /predict` that compares a query image
against an explicit, user-licensed corpus of labeled street scenes. There is
**no real corpus in this repository**: no uploads, backups or caches are ever
ingested, and the example manifest is empty on purpose. Without a built index
the endpoint keeps its existing six-key contract unchanged.

## Manifest schema (`schema_version: 1`)

See `data/reference-manifest.example.json` (empty). Each entry:

| field | required | notes |
| --- | --- | --- |
| `reference_id` | yes | unique, `[A-Za-z0-9._-]+`, becomes the stable match id |
| `iso` | yes | two-letter code; must be in the **loaded country catalog** (dynamic, no fixed list here) |
| `region` | yes | first-level region label used for grouping |
| `city` | no | optional second-level city label; empty/absent means country/region-level reference |
| `city_id` | no | optional stable city identifier (e.g. GeoNames id); requires `city` |
| `place` | yes | human-readable place label |
| `viewpoint` | no | optional camera/viewpoint label; part of the prototype group key |
| `lat`, `lng` | yes | finite, bounded (`\u00b190`, `\u00b1180`) labeled reference coordinates |
| `source` | yes | where the image came from (provenance for reviewers) |
| `license` | yes | explicit license/permission statement |
| `image_path` | yes | local **relative** path under the manifest directory |

Validation rules: unknown keys, duplicate `reference_id`, missing/blank
labels, non-finite or out-of-range coordinates, ISOs outside the loaded
catalog, and any absolute path or `..`/symlink escape out of the manifest
directory are hard errors. Image bytes are decoded as RGB with EXIF
orientation applied and all metadata dropped; labels and coordinates are
**never** read from image EXIF/GPS.

## Building

```powershell
# repository root, inference interpreter (CPU works)
inference\streetclip\.venv\Scripts\python.exe -m inference.streetclip.build_reference `
  --manifest data\reference-manifest.json --output inference\streetclip\storage\reference
```

- `--manifest` and `--output` are required. Nothing is downloaded.
- Default refuses to overwrite existing artifacts. `--force` only rewrites a
  validated directory containing exactly `reference.json`/`reference.npz`;
  unexpected files abort and nothing is recursively deleted.
- `--max-prototypes N` (default 3) bounds prototypes per
  `(iso, region, city, viewpoint)` group; groups are clustered with
  deterministic farthest-first selection over normalized image vectors.
- Artifacts: `reference.json` (canonical metadata + `index_revision`) and
  `reference.npz` (normalized `vectors` per image + cluster `prototypes`).
  Arrays are always loaded with `allow_pickle=False`.

Load-time validation (`ReferenceIndex.load`) checks model id, pinned model
revision, catalog revision, preprocessing version, vector dimension,
finiteness, L2-normalization, prototype/support layout and the full
content hash; any mismatch or tampering raises instead of loading.

## Wiring (later stage, not this stage)

Only after the index loads cleanly should the service attach the optional
`reference` sidecar (key absent otherwise). Contract consumed by the backend:

```json
"reference": {
  "index_revision": "<sha256>", "model_revision": "<pinned>",
  "catalog_revision": "<sha256>", "aggregation": "max",
  "matches": [{"reference_id": "...", "iso": "JP", "region": "kanto",
               "city": "tokyo", "city_id": "1392685764", "place": "...",
               "viewpoint": "", "lat": 35.68, "lng": 139.76,
               "raw_cosine": 0.21}],
  "regions": [{"iso": "JP", "region": "kanto", "city": "tokyo",
               "raw_cosine": 0.21, "prototype_count": 3, "support": 12}],
  "coverage": ["JP"]
}
```

- `matches` are real per-image neighbors; `regions` aggregate prototypes per
  `(iso, region, city)`. Bounds: at most 10 matches, 20 region rows.
- `aggregation` is `max` (default) or `top2mean` (explicit opt-in).
- `lat`/`lng` are labeled **reference** coordinates, never a claim about the
  photo's GPS. No filesystem paths or embedding values are ever exposed.
- Ordering is deterministic: score descending, then id/iso/region/city.

## Tests

`inference/streetclip/.venv/Scripts/python.exe -m pytest inference/streetclip/tests/test_reference.py -q`

All fixtures are synthetic temporary images and stub vectors used only to
exercise index math and validation; they are not a fabricated production
corpus.