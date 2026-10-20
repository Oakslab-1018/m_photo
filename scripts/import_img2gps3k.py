#!/usr/bin/env python3
"""Convert the img2gps3k CSV into an audited, eval-compatible manifest.

The importer treats the CSV ``lat``/``lon`` values as ground truth and derives
only the country label by point-in-polygon lookup against a local Natural Earth
GeoJSON file. It never assigns a city, a nearest country, or a neighbouring
label. Samples whose image/label cannot be established unambiguously stay
unresolved in the audit and are excluded from the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402
from shapely import STRtree, covers, make_valid  # noqa: E402
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, shape  # noqa: E402

try:  # script execution vs. import scripts.import_img2gps3k in tests
    from .eval_manifest import ALLOWED_IMAGE_SUFFIXES
except ImportError:  # pragma: no cover - exercised by direct CLI runs
    from eval_manifest import ALLOWED_IMAGE_SUFFIXES


SCHEMA_VERSION = 1
DEFAULT_ID_PREFIX = "img2gps3k"
ISO_FIELD_PRIORITY = ("ISO_A2_EH", "ISO_A2")
FEATURE_CODE_FIELDS = (
    "ISO_A2_EH",
    "ISO_A2",
    "ADM0_A3",
    "NAME",
    "NAME_LONG",
    "NAME_ZH",
    "SOV_A3",
)
_ISO_RE = re.compile(r"^[A-Z]{2}$")
_ID_PREFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ImportAuditError(ValueError):
    """A configuration/input problem that prevents a trustworthy audit."""


@dataclass(frozen=True)
class BoundaryFeature:
    index: int
    geometry: Any
    properties: dict[str, Any]
    geometry_repaired: bool


@dataclass(frozen=True)
class BoundaryIndex:
    features: tuple[BoundaryFeature, ...]
    tree: STRtree
    stats: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path_key(value: str) -> str:
    return value.replace("\\", "/").strip()


def _stable_sample_id(path: str, prefix: str) -> str:
    digest = hashlib.sha256(_normalize_path_key(path).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)) or hasattr(geometry, "geoms"):
        parts: list[Polygon] = []
        for child in geometry.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _as_polygonal(geometry: Any) -> Any | None:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return None if geometry.is_empty else geometry
    parts = _polygon_parts(geometry)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _catalog_isos(path: Path) -> set[str]:
    if not path.is_file():
        raise ImportAuditError(f"catalog database not found: {path}")
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise ImportAuditError(f"cannot open catalog database {path}: {exc}") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        try:
            rows = connection.execute(
                "SELECT place_id FROM geo_place WHERE level = 'country'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ImportAuditError(
                f"catalog database lacks a readable geo_place country table: {exc}"
            ) from exc
    finally:
        connection.close()

    isos = {str(row[0]).strip().upper() for row in rows if row and row[0] is not None}
    invalid = sorted(iso for iso in isos if not _ISO_RE.fullmatch(iso))
    if invalid:
        raise ImportAuditError(
            "catalog contains invalid country ISO values: " + ", ".join(invalid[:10])
        )
    if not isos:
        raise ImportAuditError("catalog contains no country ISO values")
    return isos


def _extract_iso(properties: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for field in ISO_FIELD_PRIORITY:
        value = properties.get(field)
        if not isinstance(value, str):
            continue
        code = value.strip().upper()
        if _ISO_RE.fullmatch(code):
            return code, field
    return None, None

def _feature_candidate(feature: BoundaryFeature) -> dict[str, Any]:
    properties = feature.properties
    iso, iso_field = _extract_iso(properties)
    return {
        "feature_index": feature.index,
        "name": properties.get("NAME") or properties.get("NAME_LONG"),
        "adm0_a3": properties.get("ADM0_A3"),
        "iso_code": iso,
        "iso_field": iso_field,
        "iso_fields": {field: properties.get(field) for field in ISO_FIELD_PRIORITY},
    }


def _load_boundaries(path: Path) -> BoundaryIndex:
    if not path.is_file():
        raise ImportAuditError(f"boundary GeoJSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ImportAuditError(f"boundary GeoJSON is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ImportAuditError("boundary file must be a GeoJSON FeatureCollection")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ImportAuditError("boundary FeatureCollection contains no features")

    features: list[BoundaryFeature] = []
    repaired: list[int] = []
    seen_fields: Counter[str] = Counter()
    valid_fields: Counter[str] = Counter()
    missing_iso: list[int] = []

    for index, raw_feature in enumerate(raw_features):
        if not isinstance(raw_feature, dict):
            raise ImportAuditError(f"boundary feature {index} is not an object")
        raw_geometry = raw_feature.get("geometry")
        if not isinstance(raw_geometry, dict):
            raise ImportAuditError(f"boundary feature {index} has no geometry object")
        try:
            geometry = shape(raw_geometry)
        except Exception as exc:
            raise ImportAuditError(
                f"boundary feature {index} geometry is invalid: {type(exc).__name__}: {exc}"
            ) from exc

        repaired_flag = False
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            repaired_flag = True
            repaired.append(index)
        geometry = _as_polygonal(geometry)
        if geometry is None:
            raise ImportAuditError(f"boundary feature {index} is not polygonal")

        raw_properties = raw_feature.get("properties")
        properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
        for field in FEATURE_CODE_FIELDS:
            value = properties.get(field)
            if value is not None and str(value).strip():
                seen_fields[field] += 1
        iso, iso_field = _extract_iso(properties)
        if iso is None:
            missing_iso.append(index)
        else:
            valid_fields[iso_field or ""] += 1
        features.append(
            BoundaryFeature(
                index=index,
                geometry=geometry,
                properties=properties,
                geometry_repaired=repaired_flag,
            )
        )

    try:
        tree = STRtree([feature.geometry for feature in features])
    except Exception as exc:
        raise ImportAuditError(
            f"cannot build boundary STRtree: {type(exc).__name__}: {exc}"
        ) from exc

    stats = {
        "feature_count": len(features),
        "geometry_types": dict(
            sorted(Counter(type(feature.geometry).__name__ for feature in features).items())
        ),
        "feature_code_fields_present": dict(sorted(seen_fields.items())),
        "iso_field_priority": list(ISO_FIELD_PRIORITY),
        "valid_iso_feature_count": sum(valid_fields.values()),
        "iso_field_selected_counts": dict(sorted(valid_fields.items())),
        "features_without_valid_iso_count": len(missing_iso),
        "features_without_valid_iso": missing_iso,
        "geometry_repaired_feature_indices": repaired,
        "crs_assumption": "GeoJSON coordinates are WGS84 longitude/latitude (EPSG:4326)",
    }
    return BoundaryIndex(features=tuple(features), tree=tree, stats=stats)


def _boundary_matches(boundary: BoundaryIndex, *, lat: float, lon: float) -> list[BoundaryFeature]:
    point = Point(lon, lat)
    try:
        indices = boundary.tree.query(point)
    except Exception as exc:
        raise ImportAuditError(
            f"boundary STRtree query failed: {type(exc).__name__}: {exc}"
        ) from exc
    matches: list[BoundaryFeature] = []
    for raw_index in sorted(int(index) for index in indices):
        feature = boundary.features[raw_index]
        if bool(covers(feature.geometry, point)):
            matches.append(feature)
    return matches


def _resolve_without_override(
    matches: Sequence[BoundaryFeature],
    catalog_isos: set[str],
) -> tuple[str | None, str, list[str], list[dict[str, Any]], dict[str, Any] | None]:
    candidates = [_feature_candidate(feature) for feature in matches]
    if not matches:
        return None, "unresolved", ["no_boundary_match"], candidates, None
    if len(matches) > 1:
        return None, "unresolved", ["multiple_boundary_matches"], candidates, None
    feature = matches[0]
    attempts: list[dict[str, Any]] = []
    iso: str | None = None
    iso_field: str | None = None
    for field in ISO_FIELD_PRIORITY:
        raw_value = feature.properties.get(field)
        code = raw_value.strip().upper() if isinstance(raw_value, str) else ""
        if not _ISO_RE.fullmatch(code):
            attempts.append({"field": field, "value": raw_value, "outcome": "invalid_iso"})
            continue
        in_catalog = code in catalog_isos
        attempts.append(
            {
                "field": field,
                "value": code,
                "outcome": "selected" if in_catalog else "not_in_catalog",
            }
        )
        if in_catalog:
            iso = code
            iso_field = field
            break
    if iso is None:
        reason = (
            "unknown_catalog_iso"
            if any(attempt["outcome"] == "not_in_catalog" for attempt in attempts)
            else "unknown_iso"
        )
        return None, "unresolved", [reason], candidates, None
    provenance = {
        "feature_index": feature.index,
        "feature_name": feature.properties.get("NAME")
        or feature.properties.get("NAME_LONG"),
        "adm0_a3": feature.properties.get("ADM0_A3"),
        "iso_code": iso,
        "iso_field": iso_field,
        "iso_resolution": (
            "primary_in_catalog"
            if iso_field == ISO_FIELD_PRIORITY[0]
            else "fallback_in_catalog"
        ),
        "iso_attempts": attempts,
    }
    return iso, "natural_earth_10m_covers", [], candidates, provenance


def _parse_coordinates(lat_raw: str, lon_raw: str) -> tuple[float | None, float | None, list[str]]:
    reasons: list[str] = []
    if lat_raw.strip() == "":
        reasons.append("latitude_missing")
    if lon_raw.strip() == "":
        reasons.append("longitude_missing")
    if reasons:
        return None, None, reasons

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return None, None, ["coordinates_not_numeric"]
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None, None, ["coordinates_not_finite"]

    lat_in_range = -90.0 <= lat <= 90.0
    lon_in_range = -180.0 <= lon <= 180.0
    if lat_in_range and lon_in_range:
        return lat, lon, []
    if -180.0 <= lat <= 180.0 and -90.0 <= lon <= 90.0:
        return lat, lon, ["coordinates_may_be_swapped"]
    if not lat_in_range:
        reasons.append("latitude_out_of_range")
    if not lon_in_range:
        reasons.append("longitude_out_of_range")
    return lat, lon, reasons


def _resolve_image_path(
    raw_path: str,
    *,
    image_root: Path,
) -> tuple[Path | None, str | None]:
    if not raw_path:
        return None, "path_missing"
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None, "path_absolute"
    if ".." in candidate.parts:
        return None, "path_escapes_image_root"
    try:
        target = (image_root / candidate).resolve()
        target.relative_to(image_root)
    except (OSError, ValueError):
        return None, "path_escapes_image_root"
    if not target.is_file():
        return None, "image_missing"
    if target.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        return None, "unsupported_image_suffix"
    return target, None


def _inspect_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            exif = image.getexif()
            gps_ifd = exif.get_ifd(34853) if exif else {}
            has_gps = bool(gps_ifd)
            result = {
                "status": "ok",
                "format": image.format,
                "width": int(image.width),
                "height": int(image.height),
                "exif_gps_present": has_gps,
                "size_bytes": path.stat().st_size,
            }
    except Exception as exc:
        return {
            "status": "unreadable",
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    if has_gps:
        result["status"] = "exif_gps_present"
    return result

def _load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ImportAuditError(f"override file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ImportAuditError(f"override file is invalid JSON: {exc}") from exc
    if isinstance(payload, dict):
        raw_overrides = payload.get("overrides")
    elif isinstance(payload, list):
        raw_overrides = payload
    else:
        raw_overrides = None
    if not isinstance(raw_overrides, list):
        raise ImportAuditError("override file must contain an 'overrides' list")

    overrides: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_overrides):
        if not isinstance(raw, dict):
            raise ImportAuditError(f"override {index} is not an object")
        raw_path = raw.get("path")
        raw_iso = raw.get("country_iso")
        reason = raw.get("reason")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ImportAuditError(f"override {index}: path is required")
        if not isinstance(raw_iso, str) or not _ISO_RE.fullmatch(raw_iso.strip().upper()):
            raise ImportAuditError(f"override {index}: country_iso must be a two-letter code")
        if not isinstance(reason, str) or not reason.strip():
            raise ImportAuditError(f"override {index}: reason is required")
        key = _normalize_path_key(raw_path)
        if key in overrides:
            raise ImportAuditError(f"duplicate override path: {raw_path}")
        override = dict(raw)
        override["path"] = raw_path
        override["country_iso"] = raw_iso.strip().upper()
        override["reason"] = reason.strip()
        overrides[key] = override
    return overrides


def _relative_image_ref(path: Path, manifest_parent: Path) -> str:
    try:
        relative = Path(os.path.relpath(path, manifest_parent))
    except ValueError as exc:
        raise ImportAuditError(f"cannot make image path relative: {path}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ImportAuditError(
            "manifest output must be at or above the image tree so image paths can "
            f"be relative to its parent: {path} is outside {manifest_parent}"
        )
    return relative.as_posix()


def _manifest_source(
    *,
    csv_name: str,
    csv_row: int,
    boundary_name: str,
    boundary_sha256: str,
    override: Mapping[str, Any] | None,
) -> str:
    source = (
        f"img2gps3k:{csv_name}#row={csv_row};"
        f"boundary:{boundary_name};boundary_sha256:{boundary_sha256}"
    )
    if override is not None:
        override_source = str(override.get("source") or "manual review")
        source += f";manual_override:{override_source}"
    return source


def _manifest_notes(
    *,
    csv_row: int,
    boundary_sha256: str,
    resolution: str,
    provenance: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> str:
    parts = [
        f"csv_row={csv_row}",
        "coordinates=img2gps3k_csv_lat_lon",
        f"country_provenance={resolution}",
        f"boundary_sha256={boundary_sha256}",
        "city_id=not_derived",
    ]
    if provenance:
        parts.append(
            "boundary_feature="
            f"{provenance.get('feature_index')}:{provenance.get('feature_name') or ''}"
        )
        parts.append(f"iso_field={provenance.get('iso_field')}")
    if override is not None:
        parts.append(f"override_reason={override['reason']}")
        if override.get("reviewed_by"):
            parts.append(f"override_reviewed_by={override['reviewed_by']}")
    return "; ".join(parts)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _discover_images(root: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        ),
        key=lambda item: item.as_posix(),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and convert img2gps3k CSV labels into an eval manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Override JSON schema: {\"overrides\": [{\"path\": \"photo.jpg\", "
            "\"country_iso\": \"US\", \"reason\": \"basis for review\", "
            "\"source\": \"review source\", \"reviewed_by\": \"name\"}]}"
        ),
    )
    parser.add_argument("--csv", required=True, help="img2gps3k CSV with path,lat,lon")
    parser.add_argument("--image-root", required=True, help="directory containing CSV images")
    parser.add_argument("--boundaries", required=True, help="local Natural Earth 10m country GeoJSON")
    parser.add_argument("--output", required=True, help="JSONL manifest to write")
    parser.add_argument("--audit-dir", required=True, help="directory for audit artifacts")
    parser.add_argument(
        "--catalog-db",
        default=str(REPO_ROOT / "data" / "geo_catalog.sqlite3"),
        help="catalog DB used only to verify country ISO membership",
    )
    parser.add_argument("--overrides", help="optional explicit manual label override JSON")
    parser.add_argument("--boundary-source", help="source description/URL for the boundary file")
    parser.add_argument(
        "--boundary-sha256",
        help="expected boundary SHA256; fail when it does not match the local file",
    )
    parser.add_argument("--csv-source", help="source description/URL for the CSV")
    parser.add_argument("--id-prefix", default=DEFAULT_ID_PREFIX, help="safe manifest id prefix")
    parser.add_argument("--force", action="store_true", help="overwrite target output files")
    parser.add_argument(
        "--retain-unlabeled-country",
        action="store_true",
        help=(
            "include samples with valid coordinates and a decodable, non-GPS image "
            "even when country ISO is unresolved; manifest country_iso is null"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any sample is unresolved or any image has an issue",
    )
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    boundary_path = Path(args.boundaries).expanduser().resolve()
    catalog_path = Path(args.catalog_db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    override_path = Path(args.overrides).expanduser().resolve() if args.overrides else None

    if not _ID_PREFIX_RE.fullmatch(args.id_prefix):
        print("error: --id-prefix must match [A-Za-z0-9._-]+", file=sys.stderr)
        return 2
    if not csv_path.is_file():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 2
    if not image_root.is_dir():
        print(f"error: image root not found: {image_root}", file=sys.stderr)
        return 2

    audit_files = {
        "audit": audit_dir / "audit.json",
        "unresolved": audit_dir / "unresolved.csv",
        "image_issues": audit_dir / "image-issues.csv",
        "extra_images": audit_dir / "extra-images.csv",
    }
    targets = [output_path, *audit_files.values()]
    if not args.force:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            print(
                "error: target output already exists (use --force): " + ", ".join(existing),
                file=sys.stderr,
            )
            return 2

    try:
        catalog_isos = _catalog_isos(catalog_path)
        boundary = _load_boundaries(boundary_path)
        overrides = _load_overrides(override_path)
        for key, override in overrides.items():
            if override["country_iso"] not in catalog_isos:
                raise ImportAuditError(
                    f"override {key} country_iso {override['country_iso']} is not in catalog"
                )
    except ImportAuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    boundary_sha256 = _sha256_file(boundary_path)
    if args.boundary_sha256 and args.boundary_sha256.strip().lower() != boundary_sha256:
        print(
            "error: boundary SHA256 mismatch: "
            f"expected {args.boundary_sha256.strip().lower()} got {boundary_sha256}",
            file=sys.stderr,
        )
        return 2
    csv_sha256 = _sha256_file(csv_path)
    override_sha256 = _sha256_file(override_path) if override_path else None
    manifest_parent = output_path.parent

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "lat", "lon"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            print("error: CSV must contain columns path,lat,lon", file=sys.stderr)
            return 2
        for raw in reader:
            rows.append({key: value or "" for key, value in raw.items() if key is not None})
    if not rows:
        print("error: CSV contains no samples", file=sys.stderr)
        return 2

    seen_paths: set[str] = set()
    seen_override_paths: set[str] = set()
    samples: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_ids: set[str] = set()

    for csv_row, row in enumerate(rows, start=2):
        raw_path = str(row.get("path", "")).strip()
        path_key = _normalize_path_key(raw_path)
        lat_raw = str(row.get("lat", ""))
        lon_raw = str(row.get("lon", ""))
        hard_reasons: list[str] = []
        lat, lon, coordinate_reasons = _parse_coordinates(lat_raw, lon_raw)
        hard_reasons.extend(coordinate_reasons)

        if path_key in seen_paths:
            hard_reasons.append("duplicate_csv_path")
        else:
            seen_paths.add(path_key)

        image_path, path_reason = _resolve_image_path(raw_path, image_root=image_root)
        if path_reason:
            hard_reasons.append(path_reason)
        if image_path is None:
            image_info: dict[str, Any] = {"status": "unresolved", "reason": path_reason}
        else:
            image_info = _inspect_image(image_path)
            if image_info["status"] == "unreadable":
                hard_reasons.append("image_unreadable")
            elif image_info["status"] == "exif_gps_present":
                hard_reasons.append("image_exif_gps")

        if lat is not None and lon is not None and not coordinate_reasons:
            matches = _boundary_matches(boundary, lat=lat, lon=lon)
        else:
            matches = []
        boundary_iso, boundary_resolution, boundary_reasons, candidates, provenance = (
            _resolve_without_override(matches, catalog_isos)
        )

        override = overrides.get(path_key)
        country_reasons: list[str] = []
        if override is not None:
            seen_override_paths.add(path_key)
            country_iso = override["country_iso"]
            resolution = "manual_override"
        else:
            country_iso = boundary_iso
            resolution = boundary_resolution
            country_reasons.extend(boundary_reasons)

        reason_codes = [*hard_reasons, *country_reasons]
        status = "resolved" if country_iso is not None and not hard_reasons else "unresolved"
        include_in_manifest = bool(
            not hard_reasons
            and (country_iso is not None or args.retain_unlabeled_country)
        )
        manifest_inclusion = (
            "country_labeled"
            if include_in_manifest and country_iso is not None
            else "coordinate_only"
            if include_in_manifest
            else "excluded_image_or_coordinates"
        )
        sample_id = _stable_sample_id(path_key, args.id_prefix)
        image_ref: str | None = None
        if image_path is not None:
            try:
                image_ref = _relative_image_ref(image_path, manifest_parent)
            except ImportAuditError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

        label_provenance = {
            "country_iso": {
                "kind": resolution,
                "boundary_sha256": boundary_sha256,
                "catalog_membership_checked": country_iso is not None,
                "feature": provenance,
            },
            "coordinates": {
                "kind": "img2gps3k_csv",
                "columns": ["lat", "lon"],
                "csv_row": csv_row,
                "source": args.csv_source,
            },
            "city_id": {"kind": "not_derived", "value": None},
        }
        sample = {
            "csv_row": csv_row,
            "path": raw_path,
            "sample_id": sample_id,
            "status": status,
            "country_label_status": "labeled" if country_iso is not None else "unresolved",
            "country_iso": country_iso,
            "resolution": resolution,
            "manifest_included": include_in_manifest,
            "manifest_inclusion": manifest_inclusion,
            "reasons": sorted(dict.fromkeys(reason_codes)),
            "excluded_reasons": sorted(dict.fromkeys(hard_reasons)),
            "coordinates": {"lat": lat, "lng": lon, "raw_lat": lat_raw, "raw_lon": lon_raw},
            "image": {
                **image_info,
                "image_root_path": str(image_path) if image_path else None,
                "manifest_image": image_ref,
            },
            "boundary": {
                "match_count": len(matches),
                "candidates": candidates,
                "selected": provenance,
                "unresolved_reasons": boundary_reasons,
            },
            "override": override,
            "label_provenance": label_provenance,
        }
        samples.append(sample)

        if not include_in_manifest:
            continue
        assert image_path is not None and image_ref is not None and lat is not None and lon is not None
        if sample_id in manifest_ids:
            print(f"error: stable id collision: {sample_id}", file=sys.stderr)
            return 2
        manifest_ids.add(sample_id)
        manifest_rows.append(
            {
                "id": sample_id,
                "image": image_ref,
                "country_iso": country_iso,
                "lat": lat,
                "lng": lon,
                "city_id": None,
                "city": None,
                "tags": ["img2gps3k"],
                "source": _manifest_source(
                    csv_name=csv_path.name,
                    csv_row=csv_row,
                    boundary_name=boundary_path.name,
                    boundary_sha256=boundary_sha256,
                    override=override,
                ),
                "notes": _manifest_notes(
                    csv_row=csv_row,
                    boundary_sha256=boundary_sha256,
                    resolution=resolution,
                    provenance=provenance,
                    override=override,
                ),
            }
        )

    unreferenced_overrides = sorted(set(overrides) - seen_override_paths)
    if unreferenced_overrides:
        print(
            "error: override paths not found in CSV: " + ", ".join(unreferenced_overrides),
            file=sys.stderr,
        )
        return 2

    discovered_images = _discover_images(image_root)
    csv_path_keys = {_normalize_path_key(sample["path"]) for sample in samples}
    extra_images = []
    for image_path in discovered_images:
        relative_to_root = image_path.relative_to(image_root).as_posix()
        if _normalize_path_key(relative_to_root) not in csv_path_keys:
            extra_images.append(
                {
                    "path": relative_to_root,
                    "size_bytes": image_path.stat().st_size,
                }
            )

    unresolved = [sample for sample in samples if sample["status"] == "unresolved"]
    image_issues = [
        sample
        for sample in samples
        if sample["image"].get("status") not in {"ok"}
    ]
    reason_counts = Counter(
        reason for sample in samples for reason in sample.get("reasons", [])
    )
    format_counts = Counter(
        str(sample["image"].get("format"))
        for sample in samples
        if sample["image"].get("status") in {"ok", "exif_gps_present"}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for manifest_row in manifest_rows:
            handle.write(json.dumps(manifest_row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest_sha256 = _sha256_file(output_path)

    image_status_counts = Counter(str(sample["image"].get("status")) for sample in samples)
    country_labeled_samples = sum(
        1 for manifest_row in manifest_rows if manifest_row["country_iso"] is not None
    )
    coordinate_only_samples = len(manifest_rows) - country_labeled_samples
    country_unresolved_samples = sum(
        1 for sample in samples if sample["country_iso"] is None
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "csv": {
                "path": str(csv_path),
                "source": args.csv_source,
                "sha256": csv_sha256,
                "samples": len(rows),
            },
            "boundaries": {
                "path": str(boundary_path),
                "source": args.boundary_source,
                "sha256": boundary_sha256,
                **boundary.stats,
            },
            "catalog": {
                "path": str(catalog_path),
                "sha256": _sha256_file(catalog_path),
                "country_count": len(catalog_isos),
                "usage": "country ISO membership only; no city lookup",
            },
            "overrides": {
                "path": str(override_path) if override_path else None,
                "sha256": override_sha256,
                "count": len(overrides),
            },
        },
        "manifest": {
            "path": str(output_path),
            "sha256": manifest_sha256,
            "samples": len(manifest_rows),
            "policy": (
                "resolved_only_eval_compatible"
                if not args.retain_unlabeled_country
                else "retain_unlabeled_country_coordinate_only"
            ),
            "retain_unlabeled_country": bool(args.retain_unlabeled_country),
            "order": "csv_order",
            "image_reference_base": str(manifest_parent),
        },
        "summary": {
            "csv_samples": len(rows),
            "manifest_samples": len(manifest_rows),
            "country_labeled_samples": country_labeled_samples,
            "coordinate_only_samples": coordinate_only_samples,
            "country_unresolved_samples": country_unresolved_samples,
            "unresolved_samples": len(unresolved),
            "excluded_from_manifest_samples": len(samples) - len(manifest_rows),
            "resolution_counts": dict(
                sorted(Counter(sample["resolution"] for sample in samples).items())
            ),
            "unresolved_reason_counts": dict(sorted(reason_counts.items())),
            "images": {
                "discovered": len(discovered_images),
                "csv_referenced": len(rows),
                "status_counts": dict(sorted(image_status_counts.items())),
                "format_counts": dict(sorted(format_counts.items())),
                "extra_count": len(extra_images),
                "extra_images": extra_images,
            },
        },
        "policy": {
            "country_label": (
                "single Natural Earth 10m polygon whose geometry covers the exact "
                "CSV point; use ISO_A2_EH when valid and present in the catalog, "
                "otherwise record an explicit fallback attempt to ISO_A2"
            ),
            "multi_match": "unresolved, never combined or ranked",
            "no_match": "unresolved, never nearest-country or nearest-city",
            "unknown_iso": "unresolved unless an explicit reviewed override exists",
            "override": "explicit JSON review with required reason; copied to audit",
            "city_id": "never derived; manifest city_id is always null",
            "exif_gps": "reported and excluded from manifest, never silently stripped/accepted",
            "manifest": (
                "country-labeled samples by default; --retain-unlabeled-country may "
                "also include coordinate-only samples with null country_iso while "
                "excluding image/coordinate failures; CSV order preserved"
            ),
            "ground_truth": (
                "lat/lng are labels from the CSV; the importer makes no model calls and "
                "never supplies ground-truth coordinates to an inference provider"
            ),
        },
        "samples": samples,
    }

    _write_json(audit_files["audit"], audit)
    _write_csv(
        audit_files["unresolved"],
        [
            "csv_row",
            "path",
            "lat",
            "lon",
            "reasons",
            "boundary_match_count",
            "boundary_candidates",
            "image_status",
            "manifest_included",
            "manifest_inclusion",
            "excluded_reasons",
            "override_reason",
        ],
        [
            {
                "csv_row": sample["csv_row"],
                "path": sample["path"],
                "lat": sample["coordinates"]["lat"],
                "lon": sample["coordinates"]["lng"],
                "reasons": ";".join(sample["reasons"]),
                "boundary_match_count": sample["boundary"]["match_count"],
                "boundary_candidates": json.dumps(
                    sample["boundary"]["candidates"], ensure_ascii=False, sort_keys=True
                ),
                "image_status": sample["image"].get("status"),
                "manifest_included": sample["manifest_included"],
                "manifest_inclusion": sample["manifest_inclusion"],
                "excluded_reasons": ";".join(sample["excluded_reasons"]),
                "override_reason": (sample.get("override") or {}).get("reason"),
            }
            for sample in unresolved
        ],
    )
    _write_csv(
        audit_files["image_issues"],
        ["csv_row", "path", "status", "reasons", "error"],
        [
            {
                "csv_row": sample["csv_row"],
                "path": sample["path"],
                "status": sample["image"].get("status"),
                "reasons": ";".join(sample["reasons"]),
                "error": sample["image"].get("error"),
            }
            for sample in image_issues
        ],
    )
    _write_csv(
        audit_files["extra_images"],
        ["path", "size_bytes"],
        extra_images,
    )

    print(
        f"audit: csv={len(rows)} manifest={len(manifest_rows)} "
        f"country_labeled={country_labeled_samples} "
        f"coordinate_only={coordinate_only_samples} "
        f"country_unresolved={country_unresolved_samples} "
        f"images={len(discovered_images)} extra={len(extra_images)}"
    )
    print(f"manifest: {output_path} sha256={manifest_sha256}")
    print(f"reports: {audit_dir}")
    if unresolved:
        print("warning: unresolved samples require review in unresolved.csv", file=sys.stderr)

    if not manifest_rows:
        print("error: no resolved samples; manifest is empty", file=sys.stderr)
        return 1
    if args.strict and (unresolved or image_issues):
        print("error: --strict found unresolved samples or image issues", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
