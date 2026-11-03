#!/usr/bin/env python3
"""Fetch BIRD and Spider into data/raw/ and check the SQLite files are there.

BIRD and Spider are both distributed as zip files behind their own landing
pages, and the links move. Rather than hard-code a URL that will rot, this
script downloads from a configurable source, and -- if the archives are
already on disk -- just unpacks and verifies them.

    python scripts/download_data.py --dataset bird
    python scripts/download_data.py --dataset bird --archive ~/Downloads/dev.zip
    python scripts/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from text2sql.data import load_dataset, missing_databases

# Official landing pages; the direct links below are the ones published there
# and are the most likely thing in this repo to go stale.
SOURCES: dict[str, dict[str, str]] = {
    "bird-dev": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "dest": "bird/dev",
        "home": "https://bird-bench.github.io/",
    },
    "bird-train": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
        "dest": "bird/train",
        "home": "https://bird-bench.github.io/",
    },
    "spider": {
        "url": "",  # Google Drive; download by hand and pass --archive
        "dest": "spider",
        "home": "https://yale-lily.github.io/spider",
    },
}


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[skip] {target} already downloaded")
        return target
    print(f"[get ] {url} -> {target}")
    with urllib.request.urlopen(url) as response, target.open("wb") as fh:  # noqa: S310
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    return target


def unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    # BIRD's archives nest the databases in a second zip.
    for inner in sorted(dest.rglob("*_databases.zip")):
        print(f"[unzip] {inner.name}")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)


def verify(data_root: Path) -> int:
    problems = 0
    for name, split in (("bird", "train"), ("bird", "dev"), ("spider", "train"), ("spider", "dev")):
        try:
            examples = load_dataset(name, data_root=data_root, split=split)
        except FileNotFoundError:
            print(f"[warn] {name}/{split}: questions file not found")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name}/{split}: {type(exc).__name__}: {exc}")
            problems += 1
            continue
        missing = missing_databases(examples)
        status = "ok" if not missing else f"{len(missing)} missing databases"
        print(f"[ok  ] {name}/{split}: {len(examples)} examples, {status}")
        if missing:
            problems += 1
            for path in missing[:5]:
                print(f"        missing: {path}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(SOURCES), action="append", default=None)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--archive", type=Path, help="a zip you downloaded by hand")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return 1 if verify(args.data_root) else 0

    targets = args.dataset or ["bird-dev", "bird-train"]
    for key in targets:
        source = SOURCES[key]
        dest = args.data_root / source["dest"]
        if args.archive:
            unpack(args.archive, dest)
            continue
        if not source["url"]:
            print(f"[skip] {key}: no direct link, download from {source['home']} "
                  f"and re-run with --dataset {key} --archive <file.zip>")
            continue
        archive = args.data_root / "_archives" / f"{key}.zip"
        try:
            download(source["url"], archive)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {key}: {type(exc).__name__}: {exc}")
            print(f"       download it by hand from {source['home']}, then re-run "
                  f"with --dataset {key} --archive <file.zip>")
            continue
        unpack(archive, dest)

    print()
    return 1 if verify(args.data_root) else 0


if __name__ == "__main__":
    sys.exit(main())
