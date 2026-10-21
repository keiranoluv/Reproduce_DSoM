#!/usr/bin/env python3
"""Resumable MMKB-DB15K raw-image downloader with source provenance.

Downloads up to one valid image per DBpedia entity, using only the published
MMKB URL archive. Does not crawl search engines, encode ViT, or change splits.

Designed for the DSoM reproduction workspace. Python 3 + Pillow only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
import json
import os
import random
import re
import socket
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path("/workspace/workspace/DSoM")
MMKB = Path("/workspace/workspace/mmkb")
RAW = Path("/workspace/workspace/dataset_audit/MMRNS/MMKB-DB15K")
PROVIDERS = ("google", "bing", "yahoo")
PROVIDER_RANK = {p: i for i, p in enumerate(PROVIDERS)}
URL_LIST_RE = re.compile(r"(?:^|/)URLS_(google|bing|yahoo)\.txt$")
MID_RE = re.compile(r"^m\.[A-Za-z0-9_]+$")
Image.MAX_IMAGE_PIXELS = 40_000_000


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def public_url(url: str) -> bool:
    """Best-effort protection from non-public and unexpected archive URLs."""
    try:
        if len(url) > 8192:
            return False
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            return False
        if parsed.username or parsed.password or parsed.port not in (None, 80, 443):
            return False
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname == "localhost" or hostname.endswith(
            (".local", ".localhost", ".internal")
        ):
            return False
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
        return bool(addresses) and all(
            ipaddress.ip_address(item[4][0]).is_global for item in addresses
        )
    except (OSError, ValueError, OverflowError):
        return False


class PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if not public_url(newurl):
            raise ValueError("Unsafe or non-public redirect target")
        return super().redirect_request(request, fp, code, message, headers, newurl)


OPENER = build_opener(PublicRedirectHandler())


def load_entities(ent_file: Path) -> dict[str, int]:
    entities: dict[str, int] = {}
    for line_number, line in enumerate(ent_file.open(encoding="utf-8"), 1):
        uri, idx_string = line.rstrip("\n").split("\t")
        idx = int(idx_string)
        if uri in entities:
            raise ValueError(f"Duplicate ent_id URI on line {line_number}")
        entities[uri] = idx
    if len(entities) != 12842 or set(entities.values()) != set(range(12842)):
        raise ValueError("Unexpected MMKB-DB15K entity mapping")
    return entities


def load_sameas(path: Path) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        fields = line.split()
        if len(fields) != 4 or fields[1] != "<SameAs>" or fields[3] != ".":
            raise ValueError(f"Invalid SameAsLink row {line_number}")
        freebase, _, wrapped_uri, _ = fields
        if (
            not freebase.startswith("/m/")
            or not wrapped_uri.startswith("<http")
            or not wrapped_uri.endswith(">")
        ):
            raise ValueError(f"Unexpected SameAsLink identifiers on line {line_number}")
        key = "m." + freebase[3:]
        uri = wrapped_uri[1:-1]
        if key in mapped and mapped[key] != uri:
            raise ValueError(f"Conflicting SameAsLink entry for {key}")
        mapped[key] = uri
    return mapped


def index_urls(
    archive: Path,
    fb_to_uri: dict[str, str],
    entities: dict[str, int],
    per_provider: int,
):
    """Stream the URL archive once, retain a bounded number of URLs per entity/provider."""
    candidates: dict[str, dict[str, list[tuple[int, str, str]]]] = {}
    rows = 0
    relevant = 0
    malformed = 0
    providers_found = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            match = URL_LIST_RE.search(member.name)
            if not member.isfile() or not match:
                continue
            provider = match.group(1)
            if provider in providers_found:
                raise ValueError(f"Duplicate URL list for {provider}")
            providers_found.add(provider)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read {member.name}")
            for raw in stream:
                rows += 1
                try:
                    url, ref = (
                        raw.decode("utf-8", errors="replace")
                        .rstrip("\r\n")
                        .rsplit("\t", 1)
                    )
                    fb_id, index_string = ref.rsplit("/", 1)
                    if not MID_RE.fullmatch(fb_id) or not index_string.isdigit():
                        raise ValueError("Invalid image ID")
                    uri = fb_to_uri.get(fb_id)
                    if uri not in entities:
                        continue
                    relevant += 1
                    chosen = candidates.setdefault(uri, {}).setdefault(provider, [])
                    if len(chosen) < per_provider and urlsplit(url).scheme.lower() in (
                        "http",
                        "https",
                    ):
                        chosen.append((int(index_string), url, fb_id))
                except (ValueError, IndexError):
                    malformed += 1
    if providers_found != set(PROVIDERS):
        raise ValueError(
            f"Expected providers {PROVIDERS}, saw {sorted(providers_found)}"
        )
    return candidates, {
        "url_rows_scanned": rows,
        "rows_matching_db15k": relevant,
        "malformed_rows": malformed,
        "candidate_entities": len(candidates),
        "providers": sorted(providers_found),
    }


def ordered_candidates(sources: dict[str, list[tuple[int, str, str]]], max_tries: int):
    options: list[tuple[str, int, str, str]] = []
    seen: set[str] = set()
    for provider in PROVIDERS:
        for idx, url, fb_id in sources.get(provider, []):
            options.append((provider, idx, url, fb_id))
    options.sort(key=lambda x: (x[1], PROVIDER_RANK[x[0]], x[3], x[2]))
    result = []
    for option in options:
        if option[2] not in seen:
            result.append(option)
            seen.add(option[2])
        if len(result) >= max_tries:
            break
    return result


def verified_image(path: Path, expected_sha: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if min(img.size) < 32:
                return False
        if (
            expected_sha
            and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha
        ):
            return False
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def fetch_and_normalize(
    url: str, timeout: float, max_bytes: int
) -> tuple[bytes, tuple[int, int]]:
    if not public_url(url):
        raise ValueError("Not a safe public HTTP(S) URL")
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; MMKBReproduction/1.0)",
            "Accept": "image/*,*/*;q=0.5",
        },
    )
    with OPENER.open(request, timeout=timeout) as response:
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("Response exceeds maximum allowed image bytes")
    with Image.open(io.BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        if min(image.size) < 32:
            raise ValueError("Image smaller than 32px")
        original_size = image.size
        # MMKB original downloader also limits larger side to 500px.
        image.thumbnail((500, 500), resample=Image.Resampling.LANCZOS)
        target_buffer = io.BytesIO()
        image.save(target_buffer, format="JPEG", quality=90)
        return target_buffer.getvalue(), original_size


def process_entity(
    item, images_dir: Path, max_tries: int, timeout: float, max_bytes: int
):
    uri, sources = item
    attempts = []
    for provider, image_idx, url, freebase_id in ordered_candidates(sources, max_tries):
        target = images_dir / freebase_id / f"{provider}_{image_idx}.jpg"
        if verified_image(target):
            content = target.read_bytes()
            return {
                "entity": uri,
                "status": "cached",
                "image_path": str(target),
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_url": url,
                "freebase_id": freebase_id,
                "provider": provider,
                "image_index": image_idx,
                "attempts": attempts,
                "timestamp_utc": utcnow(),
            }
        try:
            encoded, original_size = fetch_and_normalize(url, timeout, max_bytes)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            try:
                with tmp.open("wb") as f:
                    f.write(encoded)
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            return {
                "entity": uri,
                "status": "downloaded",
                "image_path": str(target),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "source_url": url,
                "freebase_id": freebase_id,
                "provider": provider,
                "image_index": image_idx,
                "original_size": list(original_size),
                "attempts": attempts,
                "timestamp_utc": utcnow(),
            }
        except (
            HTTPError,
            URLError,
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as exc:
            attempts.append(
                {
                    "url": url,
                    "provider": provider,
                    "image_index": image_idx,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
            )
        except Exception as exc:
            # Keep one malformed/obsolete URL from aborting a batch.
            attempts.append(
                {
                    "url": url,
                    "provider": provider,
                    "image_index": image_idx,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
            )
    return {
        "entity": uri,
        "status": "failed",
        "attempts": attempts,
        "timestamp_utc": utcnow(),
    }


def journal_latest(journal: Path) -> dict[str, dict]:
    latest = {}
    if not journal.exists():
        return latest
    with journal.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid journal JSONL row {n} in {journal}") from e
            if "entity" not in row or "status" not in row:
                raise ValueError(f"Malformed journal row {n}")
            latest[row["entity"]] = row
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ent-file", type=Path, default=ROOT / "data/MMKB-DB15K/ent_id"
    )
    parser.add_argument("--sameas", type=Path, default=RAW / "DB15K_SameAsLink.txt")
    parser.add_argument(
        "--url-archive", type=Path, default=MMKB / "image-graph_urls.tar.gz"
    )
    parser.add_argument("--images-dir", type=Path, default=MMKB / "image-graph_images")
    parser.add_argument(
        "--journal",
        type=Path,
        default=ROOT / "data/MMKB-DB15K/mmkb_raw_image_downloads.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data/MMKB-DB15K/mmkb_raw_image_download_summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum new entities to attempt this invocation (0=all remaining)",
    )
    parser.add_argument(
        "--max-tries", type=int, default=9, help="Maximum URL attempts per entity"
    )
    parser.add_argument(
        "--per-provider",
        type=int,
        default=10,
        help="URLs per provider retained in archive index",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--max-mb", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry previously failed entities"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan/plan only, no network or writes"
    )
    args = parser.parse_args()
    if not (
        0 <= args.limit <= 12842
        and 1 <= args.max_tries <= 30
        and 1 <= args.per_provider <= 30
        and 1 <= args.workers <= 24
        and 1 <= args.timeout <= 30
        and 1 <= args.max_mb <= 20
    ):
        parser.error(
            "Invalid limits; workers<=24, timeout<=30s, image<=20MB, limit 0..12842"
        )
    if args.journal.resolve() == args.summary.resolve():
        parser.error("--journal and --summary must differ")
    for path in (args.ent_file, args.sameas, args.url_archive):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    entities = load_entities(args.ent_file)
    fb_to_uri = load_sameas(args.sameas)
    print("Scanning MMKB URL archive once...", flush=True)
    candidates, audit = index_urls(
        args.url_archive, fb_to_uri, entities, args.per_provider
    )
    print("URL audit:", json.dumps(audit, sort_keys=True), flush=True)
    if len(candidates) < 12000:
        raise RuntimeError(
            "Unexpectedly low DB15K entity URL coverage; inspect mapping/archive"
        )
    records = journal_latest(args.journal)
    shuffled = sorted(candidates)
    random.Random(args.seed).shuffle(shuffled)
    pending = []
    completed = 0
    prev_failed = 0
    for uri in shuffled:
        old = records.get(uri)
        if (
            old
            and old["status"] in ("downloaded", "cached")
            and verified_image(Path(old["image_path"]), old.get("sha256"))
        ):
            completed += 1
            continue
        if old and old["status"] == "failed" and not args.retry_failed:
            prev_failed += 1
            continue
        pending.append((uri, candidates[uri]))
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        "Known downloaded/cached:", completed, "Previous failures skipped:", prev_failed
    )
    print("Selected for this run:", len(pending), "Dry run:", args.dry_run, flush=True)
    if args.dry_run:
        for uri, src in pending[:5]:
            print("Example:", uri, "URLs available:", sum(map(len, src.values())))
        print("DB15K MMKB DOWNLOAD DRY RUN: PASS")
        return
    if not pending:
        print("No pending entities; retry failures with --retry-failed if needed")
        return
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    # One journal row per finished entity, flushed synchronously in main process.
    totals = {"downloaded": 0, "cached": 0, "failed": 0}
    with args.journal.open("a", encoding="utf-8") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_entity,
                    item,
                    args.images_dir,
                    args.max_tries,
                    args.timeout,
                    args.max_mb * 1024 * 1024,
                ): item[0]
                for item in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                journal.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
                journal.flush()
                totals[result["status"]] += 1
                if i <= 5 or i % 25 == 0 or i == len(pending):
                    print(
                        f"[{i}/{len(pending)}] {result['status']}: {result['entity']}"
                        f" | run totals={totals}",
                        flush=True,
                    )
    full = journal_latest(args.journal)
    overall = {
        "total_kg_entities": len(entities),
        "entities_with_urls": len(candidates),
        "entities_without_urls": len(entities) - len(candidates),
        "recorded_successes": sum(
            x["status"] in ("downloaded", "cached") for x in full.values()
        ),
        "recorded_failures": sum(x["status"] == "failed" for x in full.values()),
        "run_totals": totals,
        "url_audit": audit,
        "source_archive": str(args.url_archive),
        "images_dir": str(args.images_dir),
        "journal": str(args.journal),
        "timestamp_utc": utcnow(),
        "note": "Historical MMKB URLs; downloaded images are not verified as DSoM original image set",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temp = args.summary.with_name(args.summary.name + ".tmp")
    temp.write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, args.summary)
    print("Summary:", args.summary, flush=True)
    print("Journal:", args.journal, flush=True)
    print("DB15K MMKB IMAGE DOWNLOAD BATCH: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
