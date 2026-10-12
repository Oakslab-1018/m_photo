"""CLI builder for the labeled street-scene reference index.

Usage (from the repository root, with the inference interpreter)::

    python -m inference.streetclip.build_reference \
        --manifest path/to/reference-manifest.json \
        --output path/to/reference-index [--force] [--max-prototypes 3]

The manifest is an explicit user-provided list of licensed reference images
(schema in ``docs/reference-index.md``). This tool never downloads images and
never invents labels; it only encodes the manifest's local files with the
pinned StreetCLIP model and writes ``reference.json`` + ``reference.npz``.

The encoder is injected as an object exposing
``encode_image(PIL.Image) -> CPU-normalized ndarray`` (the shared
``StreetClipPredictor.encode_image`` hook). Tests inject a torch-free stub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps

from .catalog import _read_catalog, compute_catalog_revision
from .config import Settings, load_settings
from .predictor import StreetClipPredictor
from .reference import (
    DEFAULT_MAX_PROTOTYPES,
    ReferenceIndex,
    ReferenceIndexError,
    ReferenceRecord,
    build_index,
    load_manifest,
)


def _load_image(path: Path) -> Image.Image:
    """Decode one reference image as metadata-free RGB.

    EXIF orientation is applied; all other metadata (including any EXIF GPS)
    is dropped without ever being read. Labels and coordinates come only from
    the manifest, never from the image file.
    """
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        rgb.load()
        rgb.info.clear()
        return rgb


def _as_vector(value: Any, *, reference_id: str) -> Any:
    if hasattr(value, "detach"):  # torch tensor from a real predictor
        value = value.detach().to("cpu").numpy()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise ReferenceIndexError("numpy is required to build a reference index") from exc
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1:
        raise ReferenceIndexError(f"encoder returned a non-vector for {reference_id}")
    return vector


def build_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    predictor: Any,
    allowed_isos: set[str] | frozenset[str],
    catalog_revision: str,
    max_prototypes: int = DEFAULT_MAX_PROTOTYPES,
    force: bool = False,
) -> ReferenceIndex:
    """Validate a manifest, encode its images, cluster prototypes, and save."""
    records, paths = load_manifest(manifest_path, allowed_isos=allowed_isos)
    if not records:
        raise ReferenceIndexError("manifest contains no references; nothing to build")
    vectors = []
    for record, path in zip(records, paths):
        image = _load_image(path)
        vectors.append(_as_vector(predictor.encode_image(image), reference_id=record.reference_id))
    dimensions = {int(vector.shape[0]) for vector in vectors}
    if len(dimensions) != 1:
        raise ReferenceIndexError("encoder returned inconsistent vector dimensions")
    index = build_index(
        records,
        vectors,
        catalog_revision=catalog_revision,
        max_prototypes=max_prototypes,
    )
    index.save(output_dir, force=force)
    return index


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m inference.streetclip.build_reference",
        description="Build a labeled StreetCLIP reference index from a licensed manifest.",
    )
    parser.add_argument("--manifest", required=True, help="path to reference-manifest.json (required)")
    parser.add_argument("--output", required=True, help="output directory for reference.json + reference.npz")
    parser.add_argument("--force", action="store_true", help="overwrite existing reference artifacts in a validated directory")
    parser.add_argument(
        "--max-prototypes",
        type=int,
        default=DEFAULT_MAX_PROTOTYPES,
        help=f"prototypes per (iso, region, city, viewpoint) group (default {DEFAULT_MAX_PROTOTYPES})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    args = _parse_args(argv)
    try:
        resolved = settings or load_settings()
        countries = _read_catalog(resolved.catalog_db_path)
        catalog_revision = compute_catalog_revision(countries)
        predictor = StreetClipPredictor(
            resolved,
            countries=countries,
            catalog_revision=catalog_revision,
        )
        predictor.load()
        index = build_from_manifest(
            args.manifest,
            args.output,
            predictor=predictor,
            allowed_isos={country.iso for country in countries},
            catalog_revision=catalog_revision,
            max_prototypes=args.max_prototypes,
            force=args.force,
        )
    except ReferenceIndexError as exc:
        print(f"reference build failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # keep CLI failures short and path-free on stderr
        print(f"reference build failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        predictor = locals().get("predictor")
        if predictor is not None:
            try:
                predictor.close()
            except Exception:  # pragma: no cover - best effort release
                pass
    print(
        "reference index built: "
        f"references={index.reference_count} groups={index.group_count} "
        f"prototypes={index.prototype_count} coverage={','.join(index.coverage)} "
        f"revision={index.index_revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_from_manifest", "main"]
