#!/usr/bin/env python3
"""Emit a JSONL manifest skeleton from a directory of images.

The generated labels are deliberately empty: ``country_iso``, ``lat`` and
``lng`` must be filled in by hand before the manifest can be evaluated. This
script only removes the tedium of listing files and keeping ids unique.

Example::

    .venv\\Scripts\\python.exe scripts\\make_eval_manifest.py ^
        --image-root D:\\photos\\eval --output data\\eval\\manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

try:  # script execution vs. import scripts.make_eval_manifest in tests
    from .eval_manifest import ALLOWED_IMAGE_SUFFIXES
except ImportError:  # pragma: no cover - exercised by direct CLI runs
    from eval_manifest import ALLOWED_IMAGE_SUFFIXES

_ID_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _image_references(
    images: Sequence[Path], *, root: Path, output: Path
) -> tuple[list[str], Path | None]:
    """Return safe image refs and an image-root override when one is required.

    Prefer paths relative to the manifest directory so the generated file can be
    evaluated without extra flags. If that would need .. (for example when
    the manifest lives outside the image tree), keep the existing image-root
    contract and return that root for the caller to print as a required flag.
    """

    output_parent = output.resolve().parent
    manifest_refs: list[str] = []
    for image in images:
        try:
            relative = image.relative_to(output_parent)
        except ValueError:
            manifest_refs = []
            break
        if ".." in relative.parts:
            manifest_refs = []
            break
        manifest_refs.append(relative.as_posix())
    if len(manifest_refs) == len(images):
        return manifest_refs, None
    return [image.relative_to(root).as_posix() for image in images], root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a labeled-manifest skeleton from an image directory."
    )
    parser.add_argument("--image-root", required=True, help="directory containing images")
    parser.add_argument("--output", required=True, help="JSONL file to write")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="walk subdirectories instead of only the top level",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.image_root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: image root not found: {root}", file=sys.stderr)
        return 2

    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        print(f"error: output exists: {output} (use --force)", file=sys.stderr)
        return 2

    iterator = root.rglob("*") if args.recursive else root.glob("*")
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    )
    if not images:
        print(f"error: no images found under {root}", file=sys.stderr)
        return 2

    image_refs, required_image_root = _image_references(
        images, root=root, output=output
    )
    seen_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    for path, image_ref in zip(images, image_refs, strict=True):
        base = _ID_CLEAN_RE.sub("-", path.stem).strip("-") or "sample"
        sample_id = base
        suffix = 2
        while sample_id in seen_ids:
            sample_id = f"{base}-{suffix}"
            suffix += 1
        seen_ids.add(sample_id)
        rows.append(
            {
                "id": sample_id,
                "image": image_ref,
                "country_iso": "",
                "lat": None,
                "lng": None,
                "city_id": None,
                "city": None,
                "tags": [],
                "license": "",
                "source": "",
                "notes": "TODO: fill country_iso, lat, lng and (optionally) city_id",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} skeleton rows to {output}")
    print("Fill country_iso, lat and lng before running scripts/eval_geolocation.py.")
    command = (
        f'{sys.executable} scripts/eval_geolocation.py --manifest "{output.resolve()}"'
    )
    if required_image_root is not None:
        command += f' --image-root "{required_image_root}"'
        print(f"Image paths are relative to image root: {required_image_root}")
    print(f"Evaluate with: {command}")
    print("Remove EXIF GPS from the images: StreetCLIP rejects images that carry it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
