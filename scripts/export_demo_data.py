#!/usr/bin/env python3
"""Export a small, deterministic MKG-Y teaching demo without model inference.

The public bundle contains 150 original entity IDs, training edges, ten held-out
questions and selected image thumbnails. Raw benchmark files are never changed.
Run with the reproduction environment (Pillow is required for thumbnails).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT.parent / "dataset_audit/MMRNS/MKG-W-Y-raw_data"
RELATIONS = {
    "actedIn": "diễn xuất trong",
    "created": "tạo nên",
    "dealsWith": "đề cập đến",
    "diedIn": "qua đời tại",
    "directed": "đạo diễn",
    "edited": "dựng phim",
    "graduatedFrom": "tốt nghiệp tại",
    "happenedIn": "diễn ra tại",
    "hasAcademicAdvisor": "có người hướng dẫn",
    "hasCapital": "có thủ đô",
    "hasChild": "có con là",
    "hasNeighbor": "tiếp giáp",
    "hasOfficialLanguage": "có ngôn ngữ chính thức",
    "hasWonPrize": "đạt giải",
    "influences": "ảnh hưởng đến",
    "isAffiliatedTo": "có liên hệ với",
    "isCitizenOf": "là công dân của",
    "isLeaderOf": "lãnh đạo",
    "isLocatedIn": "nằm tại",
    "isMarriedTo": "kết hôn với",
    "isPoliticianOf": "là chính trị gia của",
    "livesIn": "sống tại",
    "owns": "sở hữu",
    "participatedIn": "tham gia",
    "playsFor": "thi đấu cho",
    "wasBornIn": "sinh tại",
    "worksAt": "làm việc tại",
    "wroteMusicFor": "soạn nhạc cho",
}

# All ten triples were checked against the original, unchanged benchmark splits.
QUESTION_SPECS = [
    ("valid", "Bon_Jovi", "created", "This_Is_Our_House"),
    ("valid", "George_Cukor", "directed", "David_Copperfield_(1935_film)"),
    ("test", "Franklin_J._Schaffner", "directed", "Planet_of_the_Apes_(1968_film)"),
    ("test", "Elijah_Wood", "actedIn", "Sin_City_(film)"),
    ("test", "Jerry_Bruckheimer", "created", "Black_Hawk_Down_(film)"),
    ("test", "Dario_Argento", "directed", "Dracula_3D"),
    ("test", "Tom_Hanks", "actedIn", "Turner_&_Hooch"),
    ("test", "Alfred_Hitchcock", "directed", "Shadow_of_a_Doubt"),
    ("test", "Julie_Bowen", "wasBornIn", "Baltimore"),
    ("test", "Xuân_Trường_town", "isLocatedIn", "Nam_Định_Province"),
]

CASES = [
    {
        "id": "c01",
        "title": "Bắt đầu từ các liên kết",
        "subtitle": "Cấu trúc · những gì đồ thị đã biết",
        "category": "structure",
        "entityIds": [
            "Franklin_J._Schaffner",
            "The_Boys_from_Brazil_(film)",
            "Islands_in_the_Stream_(film)",
            "The_Stripper_(film)",
        ],
        "questionId": "q03",
        "observation": "Trong train, Franklin J. Schaffner đã nối với ba tác phẩm qua quan hệ directed. Liên kết tới Planet of the Apes (1968) nằm ở test và được giữ kín trong ngữ cảnh câu hỏi.",
        "lesson": "Link prediction đặt câu hỏi về một liên kết chưa quan sát. Các cạnh lân cận cung cấp ngữ cảnh, nhưng không tự chứng minh đáp án.",
    },
    {
        "id": "c02",
        "title": "Khi văn bản bổ sung ngữ cảnh",
        "subtitle": "Văn bản · đọc thêm về một thực thể",
        "category": "text",
        "entityIds": ["Alfred_Hitchcock", "Shadow_of_a_Doubt"],
        "questionId": "q08",
        "observation": "MKG-Y có mô tả tiếng Anh cho Alfred Hitchcock và Shadow of a Doubt. Bạn có thể đọc trích đoạn gốc bên cạnh các quan hệ của từng thực thể.",
        "lesson": "Mô tả có thể cung cấp nghề nghiệp, thời gian hoặc tên riêng mà vùng đồ thị nhỏ chưa thể hiện. Đây là gợi ý cho người xem, không phải bằng chứng model đã dùng những từ đó.",
    },
    {
        "id": "c03",
        "title": "Nhận diện qua hình ảnh",
        "subtitle": "Hình ảnh · một góc nhìn khác",
        "category": "image",
        "entityIds": ["Planet_of_the_Apes_(1968_film)", "Franklin_J._Schaffner"],
        "questionId": "q03",
        "observation": "Hai ảnh thu nhỏ này được lấy đúng từ các file mà manifest MKG-Y đã chọn cho bộ phim Planet of the Apes (1968) và đạo diễn Franklin J. Schaffner.",
        "lesson": "Ảnh có thể giúp nhận diện một thực thể. Muốn xác định quan hệ như directed, ta vẫn cần ngữ cảnh; việc nhìn thấy hai ảnh không xác nhận quan hệ giữa chúng.",
    },
    {
        "id": "c04",
        "title": "Một bộ phim, nhiều vai trò",
        "subtitle": "Mơ hồ · nhìn quen chưa đủ để biết quan hệ",
        "category": "ambiguity",
        "entityIds": [
            "Sin_City_(film)",
            "Robert_Rodriguez",
            "Quentin_Tarantino",
            "Bruce_Willis",
            "Elijah_Wood",
        ],
        "questionId": "q04",
        "observation": "Trong train, Sin City nối với Robert Rodriguez qua các quan hệ directed và created; Quentin Tarantino qua directed; Bruce Willis qua actedIn. Cạnh actedIn của Elijah Wood nằm ở test.",
        "lesson": "Cùng một tác phẩm có nhiều người và nhiều vai trò. Một gợi ý hình ảnh hoặc tên phim có thể gây nhầm giữa diễn viên và đạo diễn. Đây là minh họa khái niệm, không phải kết luận về lỗi của model.",
    },
    {
        "id": "c05",
        "title": "Khi một modality vắng mặt",
        "subtitle": "Thiếu dữ liệu · khoảng trống có thật trong nguồn",
        "category": "missing",
        "entityIds": [
            "Dario_Argento",
            "Jerry_Bruckheimer",
            "Julie_Bowen",
            "This_Is_Our_House",
        ],
        "questionId": "q06",
        "observation": "Dario Argento, Jerry Bruckheimer và Julie Bowen có ảnh trong archive đã dùng, nhưng không có mô tả trong file MKG-Y_ent2descreption.txt. Ngược lại, manifest ảnh không ghi nhận ảnh nguồn cho This Is Our House.",
        "lesson": "Ảnh, văn bản và cấu trúc không phải lúc nào cũng đầy đủ. DSoM truyền tri thức từ teacher đa phương thức sang student dùng cấu trúc; bản reproduction được liên kết ở đây là Source-Frozen RC, không chứng minh hiệu quả của learned RL.",
    },
]


def label(entity: str) -> str:
    return entity.replace("_", " ")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptions(path: Path) -> dict[str, str]:
    result = {}
    pattern = re.compile(r'^"(.*)"(?:@[a-zA-Z-]+)?\s*\.\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        uri, literal = line.split(maxsplit=1)
        match = pattern.match(literal)
        if not match:
            raise ValueError(f"Invalid description literal for {uri}")
        text = match.group(1).replace('\\"', '"').replace("\\n", " ")
        result[uri.removeprefix("http://dbpedia.org/resource/")] = text
    return result


def excerpt(text: str | None) -> str | None:
    if not text:
        return None
    # Preserve original wording; use a clear ellipsis when the excerpt is cut.
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    shortened = " ".join(sentences[:2])
    text = shortened + ("…" if len(sentences) > 2 else "")
    if len(text) <= 430:
        return text
    return text[:427].rsplit(" ", 1)[0] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src_data/MKG-Y")
    parser.add_argument("--mapping", type=Path, default=ROOT / "data/MKG-Y/ent_id")
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=ROOT / "data/MKG-Y/img_features_manifest.json",
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_RAW / "MKG-Y_img.zip")
    parser.add_argument(
        "--descriptions", type=Path, default=DEFAULT_RAW / "MKG-Y_ent2descreption.txt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "demo")
    args = parser.parse_args()
    from PIL import Image, ImageOps

    rows = {
        split: [
            tuple(line.split("\t"))
            for line in (args.source_dir / split)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for split in ("train", "valid", "test")
    }
    row_sets = {split: set(triples) for split, triples in rows.items()}
    union = set().union(*row_sets.values())
    mapping = {
        entity: int(idx)
        for entity, idx in (
            line.split("\t") for line in args.mapping.read_text().splitlines()
        )
    }
    manifest = json.loads(args.image_manifest.read_text())
    image_sources = {
        entity: manifest["sources_by_internal_id"][idx]
        for entity, idx in mapping.items()
    }
    texts = descriptions(args.descriptions)
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for head, relation, tail in union:
        positives[head, relation].add(tail)

    # Display categories are derived from relation roles and are not benchmark labels.
    kinds: dict[str, str] = {}
    for head, relation, tail in sorted(union):
        if relation in ("actedIn", "directed", "edited", "wroteMusicFor"):
            kinds[head] = "person"
            kinds[tail] = "work"
        elif relation == "created":
            kinds.setdefault(head, "person")
            kinds.setdefault(tail, "work")
        elif relation == "wasBornIn":
            kinds[head] = "person"
            kinds[tail] = "place"
        elif relation == "isLocatedIn":
            kinds.setdefault(head, "place")
            kinds.setdefault(tail, "place")
        elif relation in ("playsFor", "isAffiliatedTo"):
            kinds.setdefault(head, "person")
            kinds.setdefault(tail, "organization")
    kinds.update(
        {
            "Bon_Jovi": "organization",
            "ABBA": "organization",
            "Xuân_Trường_town": "place",
            "Nam_Định_Province": "place",
        }
    )

    question_entities: set[str] = set()
    questions = []
    movie_pool = [
        spec[3]
        for spec in QUESTION_SPECS
        if spec[2] in ("directed", "actedIn", "created") and spec[1] != "Bon_Jovi"
    ]
    for index, (split, head, relation, answer) in enumerate(QUESTION_SPECS, 1):
        triple = (head, relation, answer)
        assert triple in row_sets[split]
        assert triple not in row_sets["train"]
        if split == "test":
            assert triple not in row_sets["valid"]
        if head == "Bon_Jovi":
            pool = ["Waterloo_(ABBA_song)", "People_Need_Love", "Ring_Ring_(song)"]
        elif relation == "wasBornIn":
            pool = ["Leytonstone", "Bologna", "Lubbock,_Texas"]
        elif relation == "isLocatedIn":
            pool = ["Maryland", "Virginia", "Hokkaido"]
        else:
            pool = movie_pool
        distractors = [
            entity for entity in pool if entity not in positives[head, relation]
        ][:3]
        assert len(distractors) == 3
        options = [answer, *distractors]
        random.Random(2022 + index).shuffle(options)
        known = sorted(positives[head, relation])
        question_entities.update((head, *options, *known))
        prompts = {
            "created": f"Trong các tác phẩm sau, {label(head)} được nối với tác phẩm nào qua quan hệ created?",
            "directed": f"Trong các phim sau, {label(head)} đạo diễn phim nào?",
            "actedIn": f"Trong các phim sau, {label(head)} diễn xuất trong phim nào?",
            "wasBornIn": f"Theo MKG-Y, {label(head)} sinh tại đâu?",
            "isLocatedIn": f"Theo MKG-Y, {label(head)} nằm tại đâu?",
        }
        questions.append(
            {
                "id": f"q{index:02d}",
                "head": head,
                "relation": relation,
                "answer": answer,
                "options": options,
                "split": split,
                "prompt": prompts[relation],
                "acceptedIds": known,
                "explanation": f"Triple gốc trong {split}: {label(head)} → {RELATIONS[relation]} → {label(answer)}. "
                "Đây là đáp án được ghi nhận trong benchmark; các lựa chọn còn lại không phải liên kết dương đã biết cho truy vấn này trong toàn bộ train/valid/test. "
                "Một liên kết không xuất hiện trong benchmark không đồng nghĩa chắc chắn sai ngoài đời.",
            }
        )

    selected = question_entities | {
        entity for case in CASES for entity in case["entityIds"]
    }
    selected.update(
        {
            "ABBA",
            "Steven_Spielberg",
            "Alma_Reville",
            "Robert_Rodriguez",
            "Quentin_Tarantino",
            "Bruce_Willis",
            "Dimitri_Tiomkin",
            "Claudio_Simonetti",
            "Nam_Định_Province",
            "Giao_Thủy_District",
            "Red_River_Delta",
        }
    )
    assert selected <= mapping.keys()
    degree = Counter(
        entity for head, _, tail in rows["train"] for entity in (head, tail)
    )
    roots = set(selected)
    # Expand along real training facts. Prioritize edges to existing selected nodes,
    # then useful multimedia context; never add fabricated graph links.
    while len(selected) < 150:
        neighbors = Counter()
        for head, relation, tail in rows["train"]:
            if head in selected and tail not in selected:
                neighbors[tail] += 3 if head in roots else 1
            if tail in selected and head not in selected:
                neighbors[head] += 3 if tail in roots else 1
        if not neighbors:
            raise ValueError("Not enough training neighbors to select 150 entities")
        candidate = min(
            neighbors,
            key=lambda entity: (
                -neighbors[entity],
                -bool(texts.get(entity)),
                -bool(image_sources.get(entity)),
                -min(degree[entity], 10),
                entity,
            ),
        )
        selected.add(candidate)
    assert len(selected) == 150

    train_edges = sorted(
        (head, relation, tail)
        for head, relation, tail in row_sets["train"]
        if head in selected and tail in selected
    )
    edges = [
        {
            "id": f"e{index:03d}",
            "source": head,
            "target": tail,
            "relation": relation,
            "split": "train",
        }
        for index, (head, relation, tail) in enumerate(train_edges, 1)
    ]

    # Useful, bounded visual assets: question heads/answers/options, cases, then hubs.
    thumbnail_priority = list(
        dict.fromkeys(
            [q["head"] for q in questions]
            + [q["answer"] for q in questions]
            + [entity for case in CASES for entity in case["entityIds"]]
            + [option for q in questions for option in q["options"]]
            + sorted(selected, key=lambda entity: (-degree[entity], entity))
        )
    )
    thumbnail_entities = [
        entity for entity in thumbnail_priority if image_sources[entity]
    ][:48]
    image_dir = args.output / "assets/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_records = {}
    archive_metadata = []
    with zipfile.ZipFile(args.images) as archive:
        for entity in thumbnail_entities:
            source = image_sources[entity]
            info = archive.getinfo(source)
            with Image.open(io.BytesIO(archive.read(source))) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((440, 440), Image.Resampling.LANCZOS)
                filename = hashlib.sha256(entity.encode()).hexdigest()[:16] + ".webp"
                destination = image_dir / filename
                image.save(destination, "WEBP", quality=78, method=6)
            image_records[entity] = f"assets/images/{filename}"
            archive_metadata.append(
                {
                    "entity": entity,
                    "archiveMember": source,
                    "archiveMemberBytes": info.file_size,
                    "archiveMemberCRC32": f"{info.CRC:08x}",
                    "thumbnail": image_records[entity],
                    "thumbnailSha256": fingerprint(destination),
                    "thumbnailBytes": destination.stat().st_size,
                }
            )

    entities = [
        {
            "id": entity,
            "label": label(entity),
            "kind": kinds.get(entity, "entity"),
            "description": excerpt(texts.get(entity)),
            "textAvailable": bool(texts.get(entity)),
            "image": image_records.get(entity),
            "imageAvailable": bool(image_sources[entity]),
            "imageSource": image_sources[entity] or None,
            "sourceUrl": "https://dbpedia.org/resource/"
            + quote(entity, safe="()_,!'-"),
        }
        for entity in sorted(selected)
    ]

    dataset = {
        "meta": {
            "dataset": "MKG-Y",
            "version": "1.0.0",
            "seed": 2022,
            "entityCount": len(entities),
            "edgeCount": len(edges),
            "questionCount": len(questions),
            "caseCount": len(CASES),
            "splitCounts": {split: len(triples) for split, triples in rows.items()},
            "selectedTextCount": sum(e["textAvailable"] for e in entities),
            "selectedRawImageCount": sum(e["imageAvailable"] for e in entities),
            "thumbnailCount": len(image_records),
            "selectionNote": "150 thực thể được tuyển chọn quanh 10 truy vấn và 5 case study; các thực thể bổ sung được nối theo cạnh train. Đây là lát cắt minh họa có chủ đích, không đại diện ngẫu nhiên cho toàn benchmark.",
            "graphNote": "Đồ thị chỉ chứa các cạnh train. Hai câu hướng dẫn từ validation; tám câu thử thách từ test. Mỗi triple câu hỏi vắng mặt ở train; các câu test cũng vắng mặt ở validation.",
            "optionsNote": "Mỗi câu có một đáp án trong bốn lựa chọn. Distractors đã loại toàn bộ đuôi dương đã biết cho cùng head/relation trong train/valid/test; quan hệ vẫn có thể có nhiều đuôi đúng ngoài các lựa chọn.",
            "kindNote": "Nhóm hiển thị người/tác phẩm/địa điểm/tổ chức được suy ra từ vai trò quan hệ, không phải nhãn phân loại gốc của benchmark.",
            "protocol": "Source-Frozen RC reproduction. Demo không chạy inference và không đưa ra dự đoán model. Các gợi ý và case study giúp con người khám phá dữ liệu.",
            "imageCredit": "48 ảnh thu nhỏ từ archive MKG-Y_img.zip của MMRNS, theo đúng sources_by_internal_id trong manifest reproduction. Quyền ảnh thuộc các chủ sở hữu gốc; việc xuất demo không cấp thêm giấy phép cho ảnh. Đường dẫn file nguồn được lưu trong provenance.json.",
            "textCredit": "Trích đoạn mô tả tiếng Anh nguyên văn từ MKG-Y_ent2descreption.txt; liên kết DBpedia trên từng thực thể là liên kết tham khảo định danh, không phải xác nhận URL ảnh gốc.",
            "sourceLinks": [
                {
                    "title": "MMRNS · nguồn MKG-Y",
                    "url": "https://github.com/quqxui/MMRNS#datasets",
                },
                {
                    "title": "DSoM · mã tác giả",
                    "url": "https://github.com/OreOZhao/DSoM",
                },
                {"title": "DSoM · bài báo", "url": "https://arxiv.org/abs/2507.20738"},
                {
                    "title": "Reproduction · nguồn và protocol",
                    "url": "https://github.com/keiranoluv/Reproduce_DSoM/blob/main/docs/research/methodology/PROTOCOL.md",
                },
            ],
        },
        "relations": [
            {"id": relation, "label": RELATIONS[relation]}
            for relation in sorted(
                {edge["relation"] for edge in edges}
                | {q["relation"] for q in questions}
            )
        ],
        "entities": entities,
        "edges": edges,
        "questions": questions,
        "cases": CASES,
    }

    # Check exported referential integrity, benchmark fidelity and question isolation.
    exported_ids = {entity["id"] for entity in entities}
    assert len(exported_ids) == 150 and len(questions) == 10 and len(CASES) == 5
    assert Counter(q["split"] for q in questions) == {"valid": 2, "test": 8}
    for edge in edges:
        assert edge["source"] in exported_ids and edge["target"] in exported_ids
        assert (edge["source"], edge["relation"], edge["target"]) in row_sets["train"]
    for question in questions:
        assert question["head"] in exported_ids
        assert set(question["acceptedIds"]) <= exported_ids
        assert len(set(question["options"])) == 4
        assert set(question["options"]) <= exported_ids
        assert set(question["options"]) & set(question["acceptedIds"]) == {
            question["answer"]
        }
    for case in CASES:
        assert set(case["entityIds"]) <= exported_ids
    assert all(
        entity["description"] is None
        for entity in entities
        if entity["id"] in {"Dario_Argento", "Jerry_Bruckheimer", "Julie_Bowen"}
    )

    provenance = {
        "exporter": "scripts/export_demo_data.py",
        "schemaVersion": 1,
        "inputs": {
            **{
                f"split/{split}": {
                    "file": f"src_data/MKG-Y/{split}",
                    "sha256": fingerprint(args.source_dir / split),
                    "rows": len(triples),
                }
                for split, triples in rows.items()
            },
            "entityMapping": {
                "file": "data/MKG-Y/ent_id",
                "sha256": fingerprint(args.mapping),
            },
            "descriptions": {
                "file": args.descriptions.name,
                "sha256": fingerprint(args.descriptions),
            },
            "imageManifest": {
                "file": args.image_manifest.name,
                "sha256": fingerprint(args.image_manifest),
            },
            "imageArchive": {
                "file": args.images.name,
                "bytes": args.images.stat().st_size,
                "note": "Archive not rehashed; selected members recorded with size and CRC32 below.",
            },
        },
        "selection": "Fixed ten source triples and five cases; deterministic train-neighbor expansion to 150 IDs; seed-2022 option shuffling.",
        "benchmarkMutated": False,
        "inferencePerformed": False,
        "images": archive_metadata,
        "credits": {
            "images": dataset["meta"]["imageCredit"],
            "text": dataset["meta"]["textCredit"],
        },
    }
    output_dir = args.output / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "demo.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "entities": len(entities),
                "train_edges": len(edges),
                "questions": len(questions),
                "cases": len(CASES),
                "thumbnails": len(image_records),
                "thumbnail_bytes": sum(
                    item["thumbnailBytes"] for item in archive_metadata
                ),
                "selected_texts": dataset["meta"]["selectedTextCount"],
                "selected_raw_images": dataset["meta"]["selectedRawImageCount"],
            }
        )
    )


if __name__ == "__main__":
    main()
