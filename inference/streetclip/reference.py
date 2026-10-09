"""Labeled street-scene reference index (inference-only, NumPy + JSON).

The index is built offline from an explicit user-provided manifest of
licensed labeled reference images (schema: ``docs/reference-index.md``).
A build writes two artifacts into one output directory:

- ``reference.json``: canonical metadata (pinned model/revisions, per-image
  labels, prototype groups, ``index_revision``).
- ``reference.npz``: normalized float32 ``vectors`` (one row per reference
  image) plus deterministic ``prototypes`` (farthest-first cluster means).

The module only depends on NumPy and never reads EXIF metadata for labels:
coordinates come exclusively from the licensed manifest, and the caller
supplies image vectors through ``predictor.encode_image`` (see
``build_reference.py``). Country ISO membership is validated dynamically
against whatever catalog the caller loaded; nothing here hardcodes a
country list.

``ReferenceIndex.load`` validates model/revision/catalog/preprocessing
compatibility, dimension, finiteness, L2 norms, label integrity and the
content hash, and always loads arrays with ``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODEL_ID, MODEL_REVISION

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = "streetclip-image-preprocess-v1"
DEFAULT_MAX_PROTOTYPES = 3
NORM_TOLERANCE = 2e-3
MAX_MATCHES = 10
MAX_REGIONS = 20
AGGREGATIONS = ("max", "top2mean")
ARTIFACT_NAMES = ("reference.json", "reference.npz")

_REF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REFERENCE_KEYS = {
    "reference_id", "iso", "region", "city", "city_id", "place", "viewpoint",
    "lat", "lng", "source", "license", "image_path",
}


class ReferenceIndexError(ValueError):
    """Raised when a manifest, vector set or stored index is invalid."""


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceIndexError(f"{label} must be a non-empty string")
    return value.strip()


def _require_coordinate(value: Any, label: str, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceIndexError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < -bound or number > bound:
        raise ReferenceIndexError(f"{label} must be finite within [-{bound}, {bound}]")
    return number


@dataclass(frozen=True)
class ReferenceRecord:
    """One labeled reference image entry from the licensed manifest."""

    reference_id: str
    iso: str
    region: str
    city: str
    city_id: str | None
    place: str
    viewpoint: str
    lat: float
    lng: float
    source: str
    license: str
    image_path: str

    @classmethod
    def from_dict(cls, item: Any, *, index: int) -> "ReferenceRecord":
        if not isinstance(item, dict):
            raise ReferenceIndexError(f"reference {index} is not an object")
        unknown = sorted(set(item) - _REFERENCE_KEYS)
        if unknown:
            raise ReferenceIndexError(f"reference {index} has unknown keys: {unknown}")
        ref_id = _require_text(item.get("reference_id"), f"reference {index} reference_id")
        if not _REF_ID_RE.match(ref_id):
            raise ReferenceIndexError(f"reference {index} reference_id has unsupported characters")
        iso = _require_text(item.get("iso"), f"reference {ref_id} iso").upper()
        if not re.fullmatch(r"[A-Z]{2}", iso):
            raise ReferenceIndexError(f"reference {ref_id} iso must be a two-letter code")
        region = _require_text(item.get("region"), f"reference {ref_id} region")
        place = _require_text(item.get("place"), f"reference {ref_id} place")
        city_value = item.get("city", "")
        if city_value is None:
            city_value = ""
        city = _require_text(city_value, f"reference {ref_id} city") if str(city_value).strip() else ""
        city_id_value = item.get("city_id")
        city_id: str | None = None
        if city_id_value is not None:
            city_id = _require_text(city_id_value, f"reference {ref_id} city_id")
            if not city:
                raise ReferenceIndexError(f"reference {ref_id} city_id requires a city")
        viewpoint_value = item.get("viewpoint", "")
        viewpoint = ""
        if viewpoint_value is not None and str(viewpoint_value).strip():
            viewpoint = _require_text(viewpoint_value, f"reference {ref_id} viewpoint")
        lat = _require_coordinate(item.get("lat"), f"reference {ref_id} lat", 90.0)
        lng = _require_coordinate(item.get("lng"), f"reference {ref_id} lng", 180.0)
        source = _require_text(item.get("source"), f"reference {ref_id} source")
        license_text = _require_text(item.get("license"), f"reference {ref_id} license")
        image_path = _require_text(item.get("image_path"), f"reference {ref_id} image_path")
        if Path(image_path).is_absolute() or ".." in Path(image_path).parts:
            raise ReferenceIndexError(f"reference {ref_id} image_path must be a contained relative path")
        return cls(
            reference_id=ref_id, iso=iso, region=region, city=city, city_id=city_id,
            place=place, viewpoint=viewpoint, lat=lat, lng=lng, source=source,
            license=license_text, image_path=image_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id, "iso": self.iso, "region": self.region,
            "city": self.city, "city_id": self.city_id, "place": self.place,
            "viewpoint": self.viewpoint, "lat": self.lat, "lng": self.lng,
            "source": self.source, "license": self.license, "image_path": self.image_path,
        }

    @property
    def group_key(self) -> tuple[str, str, str, str]:
        return (self.iso, self.region, self.city, self.viewpoint)

    @property
    def region_key(self) -> tuple[str, str, str]:
        return (self.iso, self.region, self.city)


@dataclass(frozen=True)
class PrototypeGroup:
    """Deterministic prototype cluster family for one label combination."""

    iso: str
    region: str
    city: str
    viewpoint: str
    support: int
    prototype_start: int
    prototype_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "iso": self.iso, "region": self.region, "city": self.city,
            "viewpoint": self.viewpoint, "support": self.support,
            "prototype_start": self.prototype_start,
            "prototype_count": self.prototype_count,
        }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalize_vector(value: Any, *, label: str = "vector") -> np.ndarray:
    """Return a finite 1-D float32 L2-normalized copy of ``value``."""
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ReferenceIndexError(f"{label} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ReferenceIndexError(f"{label} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ReferenceIndexError(f"{label} must contain only finite values")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm <= 0.0:
        raise ReferenceIndexError(f"{label} must have a non-zero norm")
    return np.ascontiguousarray(vector / np.float32(norm), dtype=np.float32)


def _normalize_rows(rows: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ReferenceIndexError(f"{label} must be a non-empty 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ReferenceIndexError(f"{label} must contain only finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 0.0) or np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
        raise ReferenceIndexError(f"{label} rows must be L2-normalized")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _resolve_contained(base_dir: Path, relative: str, *, ref_id: str) -> Path:
    """Resolve ``relative`` under ``base_dir`` and reject any escape."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReferenceIndexError(f"reference {ref_id} image_path must be a contained relative path")
    resolved_base = base_dir.resolve()
    resolved = (resolved_base / candidate).resolve()
    real = Path(os.path.realpath(resolved_base / candidate))
    if not resolved.is_relative_to(resolved_base) or not real.is_relative_to(resolved_base):
        raise ReferenceIndexError(f"reference {ref_id} image_path escapes the manifest directory")
    if not resolved.is_file():
        raise ReferenceIndexError(f"reference {ref_id} image_path does not exist as a file")
    return resolved


def load_manifest(
    manifest_path: str | Path, *, allowed_isos: set[str] | frozenset[str] | None = None
) -> tuple[tuple[ReferenceRecord, ...], tuple[Path, ...]]:
    """Validate an explicit licensed manifest and return records plus paths.

    ``allowed_isos`` is the dynamic country membership loaded from the
    catalog; entries outside it are rejected. Unknown keys, duplicate
    ``reference_id`` values, non-finite coordinates, empty labels/source/
    license and path traversal/symlink escapes are all hard errors.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise ReferenceIndexError("manifest file not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReferenceIndexError("manifest is unreadable") from exc
    except json.JSONDecodeError as exc:
        raise ReferenceIndexError("manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReferenceIndexError("manifest must be a JSON object")
    unknown_top = sorted(set(payload) - {"schema_version", "references", "_comment"})
    if unknown_top:
        raise ReferenceIndexError(f"manifest has unknown keys: {unknown_top}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceIndexError(f"manifest schema_version must be {SCHEMA_VERSION}")
    entries = payload.get("references")
    if not isinstance(entries, list):
        raise ReferenceIndexError("manifest references must be a list")

    base_dir = path.parent
    records: list[ReferenceRecord] = []
    resolved_paths: list[Path] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        record = ReferenceRecord.from_dict(item, index=index)
        if record.reference_id in seen:
            raise ReferenceIndexError(f"duplicate reference_id: {record.reference_id}")
        seen.add(record.reference_id)
        if allowed_isos is not None and record.iso not in allowed_isos:
            raise ReferenceIndexError(
                f"reference {record.reference_id} iso is not in the loaded country catalog"
            )
        records.append(record)
        resolved_paths.append(_resolve_contained(base_dir, record.image_path, ref_id=record.reference_id))

    order = sorted(range(len(records)), key=lambda i: records[i].reference_id)
    return tuple(records[i] for i in order), tuple(resolved_paths[i] for i in order)


def _farthest_first(vectors: np.ndarray, max_prototypes: int) -> list[np.ndarray]:
    """Deterministic farthest-first clustering returning cluster means."""
    count = vectors.shape[0]
    budget = max(1, min(int(max_prototypes), count))
    if budget >= count:
        return [normalize_vector(vectors[i], label="prototype") for i in range(count)]
    similarities = vectors @ vectors.T
    mean = normalize_vector(vectors.mean(axis=0), label="group mean")
    chosen = [int(np.argmax(similarities @ mean))]
    best = similarities[chosen[0]].copy()
    while len(chosen) < budget:
        candidate = int(np.argmin(best))
        chosen.append(candidate)
        np.maximum(best, similarities[candidate], out=best)
    prototypes = vectors[chosen]
    assignment = np.argmax(vectors @ prototypes.T, axis=1).tolist()
    clusters: list[list[int]] = [[] for _ in range(budget)]
    for row, cluster in enumerate(assignment):
        clusters[cluster].append(row)
    means: list[np.ndarray] = []
    for cluster_index, members in enumerate(clusters):
        if members:
            means.append(normalize_vector(vectors[members].mean(axis=0), label="prototype"))
        else:
            means.append(normalize_vector(vectors[chosen[cluster_index]], label="prototype"))
    return means


@dataclass(frozen=True)
class ReferenceIndex:
    """Immutable built index with deterministic search helpers."""

    references: tuple[ReferenceRecord, ...]
    groups: tuple[PrototypeGroup, ...]
    vectors: np.ndarray
    prototypes: np.ndarray
    catalog_revision: str
    index_revision: str
    model_revision: str = MODEL_REVISION

    def _metadata_core(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_ID,
            "model_revision": self.model_revision,
            "catalog_revision": self.catalog_revision,
            "preprocessing": PREPROCESSING_VERSION,
            "dimension": int(self.vectors.shape[1]),
            "references": [record.to_dict() for record in self.references],
            "groups": [group.to_dict() for group in self.groups],
        }

    def compute_revision(self) -> str:
        digest = hashlib.sha256()
        digest.update(_canonical_json(self._metadata_core()))
        digest.update(b"\x00")
        digest.update(self.vectors.tobytes())
        digest.update(b"\x00")
        digest.update(self.prototypes.tobytes())
        return digest.hexdigest()

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def reference_count(self) -> int:
        return len(self.references)

    @property
    def prototype_count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def coverage(self) -> tuple[str, ...]:
        return tuple(sorted({record.iso for record in self.references}))

    def save(self, output_dir: str | Path, *, force: bool = False) -> Path:
        """Write ``reference.json`` + ``reference.npz`` without destructive deletes."""
        target = Path(output_dir)
        if target.exists():
            if not target.is_dir():
                raise ReferenceIndexError("output path exists and is not a directory")
            entries = sorted(child.name for child in target.iterdir())
            unknown = [name for name in entries if name not in ARTIFACT_NAMES]
            if unknown:
                raise ReferenceIndexError(f"output directory contains unexpected entries: {unknown[:5]}")
            if entries and not force:
                raise ReferenceIndexError("output already exists; pass --force to overwrite reference artifacts")
        else:
            target.mkdir(parents=True, exist_ok=True)
        metadata = dict(self._metadata_core())
        metadata["index_revision"] = self.index_revision
        assets = target / ARTIFACT_NAMES[1]
        temp_assets = target / (ARTIFACT_NAMES[1] + ".tmp")
        temp_metadata = target / (ARTIFACT_NAMES[0] + ".tmp")
        with open(temp_assets, "wb") as handle:
            np.savez(handle, vectors=self.vectors, prototypes=self.prototypes)
        temp_metadata.write_bytes(_canonical_json(metadata))
        os.replace(temp_assets, assets)
        os.replace(temp_metadata, target / ARTIFACT_NAMES[0])
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_revision: str | None = None,
        catalog_revision: str | None = None,
        *,
        allowed_isos: set[str] | frozenset[str] | None = None,
        expected_dimension: int | None = None,
        expected_preprocessing: str = PREPROCESSING_VERSION,
    ) -> "ReferenceIndex":
        """Load and fully validate a stored index directory (or JSON path)."""
        source = Path(path)
        if source.is_dir():
            metadata_path, assets_path = source / ARTIFACT_NAMES[0], source / ARTIFACT_NAMES[1]
        elif source.is_file() and source.suffix == ".json":
            metadata_path, assets_path = source, source.with_name(ARTIFACT_NAMES[1])
        else:
            raise ReferenceIndexError("reference index path must be a directory or .json file")
        if not metadata_path.is_file() or not assets_path.is_file():
            raise ReferenceIndexError("reference index artifacts are incomplete")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReferenceIndexError("reference index metadata is unreadable") from exc
        if not isinstance(metadata, dict):
            raise ReferenceIndexError("reference index metadata must be an object")

        expected_model_revision = model_revision or MODEL_REVISION
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ReferenceIndexError("reference index schema_version mismatch")
        if metadata.get("model") != MODEL_ID:
            raise ReferenceIndexError("reference index model mismatch")
        if metadata.get("model_revision") != expected_model_revision:
            raise ReferenceIndexError("reference index model_revision mismatch")
        if catalog_revision is not None and metadata.get("catalog_revision") != catalog_revision:
            raise ReferenceIndexError("reference index catalog_revision mismatch")
        if metadata.get("preprocessing") != expected_preprocessing:
            raise ReferenceIndexError("reference index preprocessing version mismatch")
        dimension = metadata.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ReferenceIndexError("reference index dimension is invalid")
        if expected_dimension is not None and dimension != expected_dimension:
            raise ReferenceIndexError("reference index dimension does not match the encoder")

        try:
            with np.load(assets_path, allow_pickle=False) as data:
                keys = set(data.files)
                if keys != {"vectors", "prototypes"}:
                    raise ReferenceIndexError("reference index arrays are not the expected set")
                vectors = _normalize_rows(data["vectors"], label="index vectors")
                prototypes = _normalize_rows(data["prototypes"], label="index prototypes")
        except ReferenceIndexError:
            raise
        except Exception as exc:  # zip/format failures, object arrays without pickle
            raise ReferenceIndexError(f"reference index arrays are unreadable: {type(exc).__name__}") from exc
        if vectors.shape[1] != dimension or prototypes.shape[1] != dimension:
            raise ReferenceIndexError("reference index array dimension mismatch")

        entries = metadata.get("references")
        group_entries = metadata.get("groups")
        if not isinstance(entries, list) or not entries:
            raise ReferenceIndexError("reference index has no references")
        if not isinstance(group_entries, list) or not group_entries:
            raise ReferenceIndexError("reference index has no prototype groups")
        records = tuple(ReferenceRecord.from_dict(item, index=i) for i, item in enumerate(entries))
        ids = [record.reference_id for record in records]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ReferenceIndexError("reference index references must be sorted and unique")
        if allowed_isos is not None and any(record.iso not in allowed_isos for record in records):
            raise ReferenceIndexError("reference index contains an iso outside the loaded catalog")
        if vectors.shape[0] != len(records):
            raise ReferenceIndexError("reference index vector count does not match metadata")

        groups: list[PrototypeGroup] = []
        cursor = 0
        for item in group_entries:
            if not isinstance(item, dict):
                raise ReferenceIndexError("reference index group is not an object")
            try:
                group = PrototypeGroup(
                    iso=_require_text(item.get("iso"), "group iso").upper(),
                    region=_require_text(item.get("region"), "group region"),
                    city=str(item.get("city") or ""),
                    viewpoint=str(item.get("viewpoint") or ""),
                    support=int(item.get("support")),
                    prototype_start=int(item.get("prototype_start")),
                    prototype_count=int(item.get("prototype_count")),
                )
            except (TypeError, ValueError) as exc:
                raise ReferenceIndexError("reference index group fields are invalid") from exc
            if group.prototype_start != cursor or group.prototype_count < 1 or group.prototype_count > group.support:
                raise ReferenceIndexError("reference index prototype layout is inconsistent")
            cursor += group.prototype_count
            groups.append(group)
        if cursor != prototypes.shape[0]:
            raise ReferenceIndexError("reference index prototype count does not match metadata")
        expected_counts: dict[tuple[str, str, str, str], int] = {}
        for record in records:
            expected_counts[record.group_key] = expected_counts.get(record.group_key, 0) + 1
        stored_counts = {
            (group.iso, group.region, group.city, group.viewpoint): group.support for group in groups
        }
        if stored_counts != expected_counts or len(groups) != len(expected_counts):
            raise ReferenceIndexError("reference index groups do not match reference labels")

        core = {
            "schema_version": metadata.get("schema_version"),
            "model": metadata.get("model"),
            "model_revision": metadata.get("model_revision"),
            "catalog_revision": metadata.get("catalog_revision"),
            "preprocessing": metadata.get("preprocessing"),
            "dimension": dimension,
            "references": [record.to_dict() for record in records],
            "groups": [group.to_dict() for group in groups],
        }
        digest = hashlib.sha256()
        digest.update(_canonical_json(core))
        digest.update(b"\x00")
        digest.update(vectors.tobytes())
        digest.update(b"\x00")
        digest.update(prototypes.tobytes())
        revision = digest.hexdigest()
        if not isinstance(metadata.get("index_revision"), str) or metadata["index_revision"] != revision:
            raise ReferenceIndexError("reference index revision/hash mismatch")

        catalog_value = metadata.get("catalog_revision")
        if not isinstance(catalog_value, str) or not catalog_value:
            raise ReferenceIndexError("reference index catalog_revision is invalid")
        return cls(
            references=records,
            groups=tuple(groups),
            vectors=vectors,
            prototypes=prototypes,
            catalog_revision=catalog_value,
            index_revision=revision,
            model_revision=expected_model_revision,
        )

    def _query(self, vector: Any) -> np.ndarray:
        query = normalize_vector(vector, label="query vector")
        if query.shape[0] != self.dimension:
            raise ReferenceIndexError("query vector dimension does not match the index")
        return query

    def _region_rows(self, scores: np.ndarray, aggregation: str) -> list[dict[str, Any]]:
        rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        for group in self.groups:
            block = scores[group.prototype_start : group.prototype_start + group.prototype_count]
            if aggregation == "top2mean" and block.size >= 2:
                top = np.sort(block)[-2:]
                value = float(np.mean(top))
            else:
                value = float(np.max(block))
            key = (group.iso, group.region, group.city)
            row = rows.setdefault(
                key,
                {"iso": key[0], "region": key[1], "city": key[2], "raw_cosine": value, "prototype_count": 0, "support": 0},
            )
            row["raw_cosine"] = max(row["raw_cosine"], value)
            row["prototype_count"] += group.prototype_count
            row["support"] += group.support
        ordered = sorted(rows.values(), key=lambda row: (-row["raw_cosine"], row["iso"], row["region"], row["city"]))
        return ordered[:MAX_REGIONS]

    def search(
        self,
        vector: Any,
        top_k: int = 5,
        *,
        aggregation: str = "max",
        limit_matches: int = MAX_MATCHES,
        limit_regions: int = MAX_REGIONS,
    ) -> dict[str, Any]:
        """Return the bounded reference sidecar for one normalized query image."""
        if aggregation not in AGGREGATIONS:
            raise ReferenceIndexError(f"aggregation must be one of {AGGREGATIONS}")
        query = self._query(vector)
        scores = np.clip(self.vectors @ query, -1.0, 1.0)
        region_scores = np.clip(self.prototypes @ query, -1.0, 1.0)
        match_limit = max(0, min(int(top_k), int(limit_matches), MAX_MATCHES))
        order = sorted(range(len(self.references)), key=lambda i: (-float(scores[i]), self.references[i].reference_id))
        matches = []
        for index in order[:match_limit]:
            record = self.references[index]
            matches.append({
                "reference_id": record.reference_id,
                "iso": record.iso,
                "region": record.region,
                "city": record.city,
                "city_id": record.city_id,
                "place": record.place,
                "viewpoint": record.viewpoint,
                "lat": record.lat,
                "lng": record.lng,
                "raw_cosine": float(scores[index]),
            })
        regions = self._region_rows(region_scores, aggregation)[: max(0, min(int(limit_regions), MAX_REGIONS))]
        return {
            "index_revision": self.index_revision,
            "model_revision": self.model_revision,
            "catalog_revision": self.catalog_revision,
            "aggregation": aggregation,
            "matches": matches,
            "regions": regions,
            "coverage": list(self.coverage),
        }

    def city_scores(self, vector: Any, country_iso: str, *, aggregation: str = "max") -> list[dict[str, Any]]:
        """Optional small helper: region rows restricted to one ISO country."""
        query = self._query(vector)
        region_scores = np.clip(self.prototypes @ query, -1.0, 1.0)
        rows = [row for row in self._region_rows(region_scores, aggregation) if row["iso"] == str(country_iso).upper()]
        return rows


def build_index(
    references: tuple[ReferenceRecord, ...] | list[ReferenceRecord],
    vectors: Any,
    *,
    catalog_revision: str,
    model_revision: str = MODEL_REVISION,
    max_prototypes: int = DEFAULT_MAX_PROTOTYPES,
) -> ReferenceIndex:
    """Deterministically cluster normalized vectors into labeled prototypes."""
    catalog_revision = _require_text(catalog_revision, "catalog_revision")
    if not isinstance(max_prototypes, int) or isinstance(max_prototypes, bool) or max_prototypes < 1:
        raise ReferenceIndexError("max_prototypes must be a positive integer")
    records = tuple(references)
    if not records:
        raise ReferenceIndexError("no references to index")
    matrix = _normalize_rows(vectors, label="reference vectors")
    if matrix.shape[0] != len(records):
        raise ReferenceIndexError("vector count does not match reference count")
    order = sorted(range(len(records)), key=lambda i: records[i].reference_id)
    records = tuple(records[i] for i in order)
    matrix = np.ascontiguousarray(matrix[order], dtype=np.float32)
    if len({record.reference_id for record in records}) != len(records):
        raise ReferenceIndexError("duplicate reference_id")

    groups: list[PrototypeGroup] = []
    prototype_rows: list[np.ndarray] = []
    family_keys = sorted({record.group_key for record in records})
    for key in family_keys:
        members = [index for index, record in enumerate(records) if record.group_key == key]
        clusters = _farthest_first(matrix[members], max_prototypes)
        groups.append(
            PrototypeGroup(
                iso=key[0], region=key[1], city=key[2], viewpoint=key[3],
                support=len(members), prototype_start=len(prototype_rows), prototype_count=len(clusters),
            )
        )
        prototype_rows.extend(clusters)
    prototypes = _normalize_rows(np.stack(prototype_rows), label="prototypes")
    index = ReferenceIndex(
        references=records,
        groups=tuple(groups),
        vectors=matrix,
        prototypes=prototypes,
        catalog_revision=catalog_revision,
        index_revision="",
        model_revision=model_revision,
    )
    return ReferenceIndex(
        references=index.references,
        groups=index.groups,
        vectors=index.vectors,
        prototypes=index.prototypes,
        catalog_revision=index.catalog_revision,
        index_revision=index.compute_revision(),
        model_revision=index.model_revision,
    )


__all__ = [
    "ARTIFACT_NAMES",
    "DEFAULT_MAX_PROTOTYPES",
    "MODEL_ID",
    "MODEL_REVISION",
    "PREPROCESSING_VERSION",
    "PrototypeGroup",
    "ReferenceIndex",
    "ReferenceIndexError",
    "ReferenceRecord",
    "SCHEMA_VERSION",
    "build_index",
    "load_manifest",
    "normalize_vector",
]