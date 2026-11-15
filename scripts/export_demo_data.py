#!/usr/bin/env python3
"""Export the WN9 Animals & Everyday Objects demo without model inference.

The benchmark files remain unchanged. Pillow is needed only to regenerate
thumbnails. Source names, English definitions, relations and evidence are kept
verbatim from the pinned MoSE WN9 snapshot used by the reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "192d9ff2882b5be6ef8239a37cbd90c5dd102b5a"
SOURCE = f"https://github.com/OreOZhao/MoSE4MKGC/blob/{REVISION}/src_data/WN9"
INVERSES = {
    "_hypernym": "_hyponym",
    "_hyponym": "_hypernym",
    "_has_part": "_part_of",
    "_part_of": "_has_part",
    "_member_holonym": "_member_meronym",
    "_member_meronym": "_member_holonym",
    "_member_of_domain_topic": "_synset_domain_topic_of",
    "_synset_domain_topic_of": "_member_of_domain_topic",
    "_derivationally_related_form": "_derivationally_related_form",
}
RELATION_MEANINGS = {
    "_hypernym": "là một loại của: từ nghĩa cụ thể đến lớp rộng hơn",
    "_hyponym": "có một loại cụ thể là: chiều ngược của _hypernym",
    "_part_of": "là một bộ phận của một tổng thể",
    "_has_part": "có một bộ phận: chiều ngược của _part_of",
    "_member_holonym": "là thành viên của một tập hợp",
    "_member_meronym": "có thành viên là",
    "_member_of_domain_topic": "có thành viên thuộc lĩnh vực chủ đề",
    "_synset_domain_topic_of": "thuộc lĩnh vực chủ đề",
    "_derivationally_related_form": "có dạng từ liên quan về cấu tạo từ",
}
QUESTION_SPECS = [
    {
        "id": "q01",
        "split": "test",
        "head": "n02112826",
        "relation": "_hypernym",
        "answer": "n02084071",
        "distractors": ["n02121620", "n01503061", "n02512053"],
        "prompt": "corgi thuộc nhóm động vật nào?",
        "context": [
            ("n02084071", "_hypernym", "n02083346"),
            ("n02103406", "_hypernym", "n02084071"),
            ("n02084071", "_hyponym", "n02111626"),
            ("n02110341", "_hypernym", "n02084071"),
            ("n02084071", "_hypernym", "n01317541"),
        ],
        "explanation": "Đáp án là dog. _hypernym đi từ một nghĩa cụ thể (corgi) tới lớp rộng hơn (dog). Cạnh thuận nằm ở test dòng 392; train dòng 6465 đã chứa dog → _hyponym → corgi. Vì vậy, ở ví dụ này có thể xác định đáp án từ chính cạnh nghịch trong train; không nên coi đây là bằng chứng về suy luận mới của model.",
    },
    {
        "id": "q02",
        "split": "valid",
        "head": "n01816887",
        "relation": "_hypernym",
        "answer": "n01503061",
        "distractors": ["n02084071", "n02512053", "n02159955"],
        "prompt": "parrot thuộc nhóm động vật nào?",
        "context": [
            ("n01816887", "_hyponym", "n01821203"),
            ("n01816887", "_hyponym", "n01819115"),
            ("n01503061", "_hypernym", "n01471682"),
            ("n01503061", "_hyponym", "n01604330"),
            ("n01503061", "_hyponym", "n01517966"),
        ],
        "explanation": "Đáp án là bird, theo validation dòng 1062. Train có parrot → _hyponym → parakeet và cockatoo, đồng thời có bird → _hypernym → vertebrate. Các cạnh này là ngữ cảnh, chưa tạo thành chứng minh cho liên kết cần điền. Cạnh nghịch bird → _hyponym → parrot nằm ở test dòng 830, nên không được đưa vào đồ thị train của câu hỏi.",
    },
    {
        "id": "q03",
        "split": "test",
        "head": "n03903424",
        "relation": "_part_of",
        "answer": "n02834778",
        "distractors": ["n03001627", "n03046257", "n02876657"],
        "prompt": "pedal là bộ phận của vật nào trong các lựa chọn?",
        "context": [
            ("n03903424", "_hypernym", "n03659292"),
            ("n03903424", "_hyponym", "n02890804"),
            ("n02834778", "_has_part", "n03487090"),
            ("n02834778", "_has_part", "n02836035"),
            ("n03903424", "_part_of", "n03854065"),
        ],
        "explanation": "Trong các lựa chọn, đáp án là bicycle: pedal → _part_of → bicycle ở test dòng 219. Train dòng 9165 đã có bicycle → _has_part → pedal, là quan hệ nghịch trực tiếp. Quan hệ có thể có nhiều đuôi đúng: WN9 còn ghi pedal là bộ phận của motor vehicle và organ (pipe organ); các thực thể đó được loại khỏi distractors.",
    },
    {
        "id": "q04",
        "split": "test",
        "head": "n02958343",
        "relation": "_has_part",
        "answer": "n02761834",
        "distractors": ["n03467517", "n04284002", "n11669921"],
        "prompt": "car có bộ phận nào trong các lựa chọn dưới đây?",
        "context": [
            ("n02958343", "_has_part", "n02963821"),
            ("n02958343", "_has_part", "n02761557"),
            ("n02761834", "_hypernym", "n02694426"),
            ("n02761834", "_has_part", "n03537412"),
            ("n03791235", "_hyponym", "n02958343"),
        ],
        "explanation": "Đáp án là automobile horn, nghĩa cụ thể là thiết bị trên ô tô tạo tiếng cảnh báo. Cạnh car → _has_part → automobile horn ở test dòng 756; cạnh nghịch automobile horn → _part_of → car có ở train dòng 3629. Những bộ phận khác của car vẫn có thể đúng ngoài bốn lựa chọn này.",
    },
    {
        "id": "q05",
        "split": "test",
        "head": "n02913152",
        "relation": "_has_part",
        "answer": "n04546855",
        "distractors": ["n03487090", "n04284002", "n11669921"],
        "prompt": "building có bộ phận nào trong các lựa chọn dưới đây?",
        "context": [
            ("n02913152", "_has_part", "n04105893"),
            ("n02913152", "_has_part", "n04105068"),
            ("n04546855", "_hypernym", "n03894379"),
            ("n04546855", "_part_of", "n04105893"),
            ("n04587648", "_part_of", "n02913152"),
        ],
        "explanation": "Đáp án là wall. Building → _has_part → wall nằm ở test dòng 300, còn wall → _part_of → building đã có ở train dòng 6821. Các cạnh về room, roof và window cho thêm ngữ cảnh kiến trúc. Wall ở đây là vách kiến trúc n04546855, được xác định bằng synset và định nghĩa tiếng Anh gốc.",
    },
]
ANIMAL_NAMES = [
    "animal",
    "domestic animal",
    "mammal",
    "vertebrate",
    "canine",
    "dog",
    "cat",
    "rabbit",
    "horse",
    "cow",
    "sheep",
    "goat",
    "deer",
    "elephant",
    "tiger",
    "lion",
    "leopard",
    "bear",
    "fox",
    "wolf",
    "monkey",
    "gorilla",
    "zebra",
    "giraffe",
    "camel",
    "kangaroo",
    "bat",
    "mouse",
    "rat",
    "squirrel",
    "hamster",
    "hedgehog",
    "otter",
    "beaver",
    "seal",
    "whale",
    "dolphin",
    "bird",
    "parrot",
    "parakeet",
    "cockatoo",
    "eagle",
    "owl",
    "duck",
    "goose",
    "swan",
    "chicken",
    "turkey",
    "pigeon",
    "sparrow",
    "penguin",
    "ostrich",
    "crane",
    "pelican",
    "crow",
    "raven",
    "robin",
    "gull",
    "woodpecker",
    "fish",
    "shark",
    "salmon",
    "tuna",
    "trout",
    "goldfish",
    "carp",
    "ray",
    "butterfly",
    "moth",
    "bee",
    "ant",
    "fly",
    "beetle",
    "grasshopper",
    "dragonfly",
    "spider",
    "crab",
    "lobster",
    "shrimp",
    "snail",
    "frog",
    "turtle",
    "snake",
    "lizard",
    "crocodile",
    "alligator",
    "insect",
    "reptile",
    "amphibian",
]
OBJECT_NAMES = [
    "artifact",
    "vehicle",
    "wheeled vehicle",
    "motor vehicle",
    "furniture",
    "chair",
    "table",
    "bed",
    "clock",
    "bottle",
    "cup",
    "spoon",
    "bowl",
    "door",
    "window",
    "wheel",
    "handlebar",
    "bicycle wheel",
    "bicycle seat",
    "kickstand",
    "hammer",
    "piano",
    "violin",
    "guitar",
    "musical instrument",
    "organ",
    "bicycle",
    "car",
    "bus",
    "train",
    "boat",
    "ship",
    "airplane",
    "book",
    "pencil",
    "pen",
    "paper",
    "computer",
    "keyboard",
    "telephone",
    "camera",
    "shoe",
    "hat",
    "coat",
    "shirt",
    "trousers",
    "umbrella",
    "bag",
    "basket",
    "lamp",
    "mirror",
    "plate",
    "kettle",
    "teapot",
    "sofa",
    "house",
    "building",
    "room",
    "wall",
    "roof",
    "stairway",
    "tool",
    "lever",
    "pedal",
    "button",
]
PLANT_NAMES = ["plant", "tree", "flower", "rose", "oak", "pine", "sunflower", "grass"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pairs(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    values, rows = {}, {}
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        key, value = line.split("\t", 1)
        values[key], rows[key] = value, row
    return values, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src_data/WN9")
    parser.add_argument("--mapping", type=Path, default=ROOT / "data/WN9/ent_id")
    parser.add_argument(
        "--parent-mapping", type=Path, default=ROOT / "data/WN18/ent_id"
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=ROOT / "data/WN18/img_features_manifest.json",
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT.parent / "fb15k_wn18_images.zip"
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
    assert {split: len(triples) for split, triples in rows.items()} == {
        "train": 11741,
        "valid": 1337,
        "test": 1319,
    }
    union = set().union(*row_sets.values())
    source_rows = {
        split: {triple: row for row, triple in enumerate(triples, 1)}
        for split, triples in rows.items()
    }
    names, name_rows = read_pairs(args.source_dir / "ent_name.txt")
    definitions, definition_rows = read_pairs(args.source_dir / "ent_dscp.txt")
    mapping, _ = read_pairs(args.mapping)
    parent_mapping, _ = read_pairs(args.parent_mapping)
    manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    ids = set(mapping)
    assert len(ids) == 6555
    # WN9 nIDs and WN18 IDs differ. Do not use WN9 row indices on WN18 arrays.
    image_sources = {
        entity: manifest["sources_by_internal_id"][int(parent_mapping[entity[1:]])]
        for entity in ids
    }
    assert sum(bool(source) for source in image_sources.values()) == 6547
    assert all(entity in names and entity[1:] in definitions for entity in ids)
    positives = defaultdict(set)
    for head, relation, tail in union:
        positives[head, relation].add(tail)

    def primary(entity):
        return names[entity].split(", ")[0]

    def named(word, category):
        candidates = [entity for entity in sorted(ids) if primary(entity) == word]
        if category == "animal":
            candidates = [
                entity
                for entity in candidates
                if entity == "n00015388" or 1300000 <= int(entity[1:]) < 2700000
            ]
        elif category == "object":
            candidates = [
                entity
                for entity in candidates
                if entity == "n00021939" or 2600000 <= int(entity[1:]) < 4700000
            ]
        elif category == "plant":
            candidates = [
                entity
                for entity in candidates
                if entity == "n00017222" or 11000000 <= int(entity[1:]) < 14000000
            ]
        return candidates[0] if candidates else None

    preferred = {
        category: [entity for word in words if (entity := named(word, category))]
        for category, words in (
            ("animal", ANIMAL_NAMES),
            ("object", OBJECT_NAMES),
            ("plant", PLANT_NAMES),
        )
    }
    selected = {"n03501614", "n05538625", "n03481172", "n00015388"}
    for spec in QUESTION_SPECS:
        selected.update((spec["head"], spec["answer"], *spec["distractors"]))
        selected.update(positives[spec["head"], spec["relation"]])
        for head, _, tail in spec["context"]:
            selected.update((head, tail))
    selected.update(preferred["animal"][:49])
    selected.update(preferred["plant"])
    assert selected <= ids and len(selected) <= 150
    for entity in preferred["object"] + preferred["animal"][49:]:
        if len(selected) == 150:
            break
        selected.add(entity)
    assert len(selected) == 150

    edges = [
        {
            "id": f"e{row:05d}",
            "source": head,
            "target": tail,
            "relation": relation,
            "split": "train",
            "sourceRow": row,
            "sourceUrl": f"{SOURCE}/train#L{row}",
        }
        for row, (head, relation, tail) in enumerate(rows["train"], 1)
        if head in selected and tail in selected
    ]
    edges_by_triple = {(e["source"], e["relation"], e["target"]): e for e in edges}
    questions = []
    for index, spec in enumerate(QUESTION_SPECS, 1):
        head, relation, answer = spec["head"], spec["relation"], spec["answer"]
        triple, split = (head, relation, answer), spec["split"]
        assert triple in row_sets[split] and triple not in row_sets["train"]
        if split == "test":
            assert triple not in row_sets["valid"]
        inverse = (answer, INVERSES[relation], head)
        inverse_edges = (
            [edges_by_triple[inverse]["id"]] if inverse in row_sets["train"] else []
        )
        context = [edges_by_triple[t] for t in spec["context"]]
        evidence_entities = sorted(
            {head, answer}
            | {e[field] for e in context for field in ("source", "target")}
        )
        assert 5 <= len(evidence_entities) <= 10
        answer_row = source_rows[split][triple]
        options = [answer, *spec["distractors"]]
        random.Random(2022 + index).shuffle(options)
        accepted = sorted(positives[head, relation])
        assert len(set(options)) == 4 and set(options) & set(accepted) == {answer}
        assert set(accepted) <= selected
        questions.append(
            {
                "id": spec["id"],
                "head": head,
                "relation": relation,
                "answer": answer,
                "options": options,
                "split": split,
                "prompt": spec["prompt"],
                "explanation": f"Đáp án là {primary(answer)}. Triple gốc: {primary(head)} → {relation} → {primary(answer)}. Mở bằng chứng để xem các cạnh train và dòng nguồn đáp án.",
                "acceptedIds": accepted,
                "evidence": {
                    "answerRecord": {
                        "split": split,
                        "row": answer_row,
                        "triple": list(triple),
                        "sourceUrl": f"{SOURCE}/{split}#L{answer_row}",
                    },
                    "inverseEdgeIds": inverse_edges,
                    "contextEdgeIds": [e["id"] for e in context],
                    "entityIds": evidence_entities,
                    "kind": "inverse" if inverse_edges else "context",
                    "explanation": spec["explanation"],
                },
            }
        )

    cases = [
        {
            "id": "c01",
            "title": "Một con vật, nhiều cấp phân loại",
            "subtitle": "Animals · hiểu chiều của quan hệ",
            "category": "structure",
            "entityIds": [
                "n02112826",
                "n02084071",
                "n02083346",
                "n01816887",
                "n01503061",
            ],
            "questionId": "q01",
            "observation": "WN9 ghi corgi → _hypernym → dog ở test, dog → _hypernym → canine ở train. Với parrot, cạnh _hypernym tới bird nằm ở validation. Tên và định nghĩa tiếng Anh giữ nguyên từ nguồn.",
            "lesson": "_hypernym đi tới lớp rộng hơn, còn _hyponym đi tới loại cụ thể hơn. Hai mũi tên khác chiều có thể biểu diễn cùng một quan hệ phân loại.",
        },
        {
            "id": "c02",
            "title": "Bộ phận không phải một loại",
            "subtitle": "Everyday objects · phân biệt part và type",
            "category": "structure",
            "entityIds": [
                "n03903424",
                "n02834778",
                "n03659292",
                "n02836035",
                "n03487090",
            ],
            "questionId": "q03",
            "observation": "Pedal nối tới lever bằng _hypernym và tới bicycle bằng _part_of. Train còn có bicycle → _has_part → bicycle wheel và handlebar.",
            "lesson": "Pedal là một loại lever và là một bộ phận của bicycle. Đọc đúng nhãn quan hệ giúp phân biệt quan hệ phân loại với quan hệ bộ phận–tổng thể.",
        },
        {
            "id": "c03",
            "title": "Khi đáp án đã có ở chiều nghịch",
            "subtitle": "Dataset evidence · công khai thông tin giữa các split",
            "category": "ambiguity",
            "entityIds": [
                "n02112826",
                "n02084071",
                "n03903424",
                "n02834778",
                "n02958343",
                "n02761834",
            ],
            "questionId": "q04",
            "observation": "Bốn trong năm câu được chọn có cạnh nghịch trực tiếp trong train: dog → _hyponym → corgi; bicycle → _has_part → pedal; automobile horn → _part_of → car; wall → _part_of → building. Bảng bằng chứng của từng câu dẫn tới đúng dòng nguồn.",
            "lesson": "Việc che cạnh thuận chưa loại bỏ thông tin từ cạnh nghịch. Đây là đặc điểm của lát cắt WN9 đang dùng, không phải bằng chứng model đã suy luận được một sự thật hoàn toàn mới.",
        },
        {
            "id": "c04",
            "title": "Cùng là head, nhưng khác nghĩa",
            "subtitle": "Synset ambiguity · đọc ID và định nghĩa",
            "category": "text",
            "entityIds": ["n03501614", "n05538625", "n03481172", "n00015388"],
            "observation": "Head n03501614 là phần đập của dụng cụ; train có head → _part_of → hammer. Head n05538625 là phần đầu cơ thể; train có head → _part_of → animal. Hai thực thể có cùng primary lemma nhưng khác synset và định nghĩa.",
            "lesson": "Không gộp hai node chỉ vì nhãn chữ giống nhau. Định nghĩa tiếng Anh và ID là phần quan trọng để xác định đúng nghĩa; ảnh chỉ cung cấp thêm một góc nhìn.",
        },
        {
            "id": "c05",
            "title": "Ảnh gợi ý, định nghĩa xác định nghĩa",
            "subtitle": "Modalities · xem nguồn trước khi diễn giải",
            "category": "image",
            "entityIds": [
                "n01816887",
                "n01819115",
                "n01503061",
                "n02761834",
                "n02958343",
            ],
            "questionId": "q02",
            "observation": "Ảnh được lấy đúng theo manifest WN18 và căn chỉnh sang WN9 bằng synset ID, không ghép theo vị trí hàng. Parrot và cockatoo có thể được xem cùng tên và định nghĩa gốc; automobile horn có nghĩa cụ thể là thiết bị tạo tiếng cảnh báo trên ô tô.",
            "lesson": "Một ảnh cụ thể không biểu diễn hết nghĩa của synset hay mọi quan hệ. Xếp hạng trong web là nội dung minh họa được biên soạn; không được suy ra rằng ảnh hoặc một cạnh nào đã gây ra điểm số của checkpoint.",
        },
    ]
    for case in cases:
        assert set(case["entityIds"]) <= selected

    image_priority = list(
        dict.fromkeys(
            [q["head"] for q in questions]
            + [q["answer"] for q in questions]
            + [e for case in cases for e in case["entityIds"]]
            + preferred["animal"][5:19]
            + preferred["object"][:18]
            + preferred["plant"][:5]
            + sorted(selected)
        )
    )
    image_entities = [e for e in image_priority if e in selected and image_sources[e]][
        :48
    ]
    assert len(image_entities) == 48
    image_dir = args.output / "assets/images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images, image_paths = [], {}
    with zipfile.ZipFile(args.images) as archive:
        for entity in image_entities:
            member = image_sources[entity]
            assert member.startswith(f"wn18-images/{entity}/")
            info = archive.getinfo(member)
            with Image.open(io.BytesIO(archive.read(member))) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((360, 360), Image.Resampling.LANCZOS)
                destination = image_dir / (entity + ".webp")
                image.save(destination, "WEBP", quality=73, method=6)
            image_paths[entity] = f"assets/images/{destination.name}"
            images.append(
                {
                    "entity": entity,
                    "parentEntity": entity[1:],
                    "parentInternalId": int(parent_mapping[entity[1:]]),
                    "archiveMember": member,
                    "archiveMemberBytes": info.file_size,
                    "archiveMemberCRC32": f"{info.CRC:08x}",
                    "thumbnail": image_paths[entity],
                    "thumbnailSha256": sha256(destination),
                    "thumbnailBytes": destination.stat().st_size,
                }
            )
    keep = {Path(path).name for path in image_paths.values()}
    for old_thumbnail in image_dir.glob("*.webp"):
        if old_thumbnail.name not in keep:
            old_thumbnail.unlink()
    assert sum(record["thumbnailBytes"] for record in images) < 1_000_000

    known_kinds = {
        entity: kind for kind, entities in preferred.items() for entity in entities
    }
    known_kinds.update(
        {
            "n02112826": "animal",
            "n02103406": "animal",
            "n02110341": "animal",
            "n02111626": "animal",
            "n01604330": "animal",
            "n01517966": "animal",
            "n03501614": "object",
            "n05538625": "concept",
        }
    )

    def kind(entity):
        if entity in known_kinds:
            return known_kinds[entity]
        offset = int(entity[1:])
        if 1300000 <= offset < 2700000:
            return "animal"
        if 2600000 <= offset < 4700000:
            return "object"
        if 11000000 <= offset < 14000000:
            return "plant"
        return "concept"

    entities = [
        {
            "id": entity,
            "label": primary(entity),
            "aliases": names[entity].split(", ")[1:],
            "fullName": names[entity],
            "kind": kind(entity),
            "description": definitions[entity[1:]],
            "textAvailable": True,
            "image": image_paths.get(entity),
            "imageAvailable": bool(image_sources[entity]),
            "imageSource": image_sources[entity] or None,
            "sourceUrl": f"{SOURCE}/ent_name.txt#L{name_rows[entity]}",
            "descriptionSourceUrl": f"{SOURCE}/ent_dscp.txt#L{definition_rows[entity[1:]]}",
        }
        for entity in sorted(selected)
    ]
    image_credit = "48 ảnh thu nhỏ từ fb15k_wn18_images.zip, giữ đúng file ảnh mà WN18 training manifest chọn; căn chỉnh WN9 n######## → WN18 ######## → parentInternalId. Ảnh thuộc các chủ sở hữu gốc; không suy ra giấy phép mới từ việc xuất demo. Tên archive/member và checksum thumbnail được ghi trong provenance.json."
    data = {
        "meta": {
            "dataset": "WN9",
            "title": "Animals & Everyday Objects",
            "version": "2.0.0",
            "seed": 2022,
            "sourceRevision": REVISION,
            "entityCount": len(entities),
            "edgeCount": len(edges),
            "questionCount": len(questions),
            "caseCount": len(cases),
            "splitCounts": {split: len(triples) for split, triples in rows.items()},
            "benchmarkEntityCount": 6555,
            "selectedTextCount": len(entities),
            "selectedRawImageCount": sum(e["imageAvailable"] for e in entities),
            "thumbnailCount": len(images),
            "inferencePerformed": False,
            "sourceImageCoverage": {"real": 6547, "imputed": 8, "total": 6555},
            "selectionNote": "150 synset được tuyển chọn quanh động vật, đồ vật quen thuộc, năm truy vấn và năm case study. Đây là lát cắt minh họa có chủ đích, không phải mẫu ngẫu nhiên đại diện toàn benchmark.",
            "graphNote": "Đồ thị chỉ chứa các cạnh train nguyên bản. Một câu hỏi từ validation, bốn câu từ test. Các cạnh thuận cần điền không có trong train; bốn câu có cạnh nghịch trực tiếp trong train và được công khai trong bằng chứng.",
            "inversionNote": "WN9 có nhiều inverse facts giữa các split. Trong năm câu tuyển chọn, q01/q03/q04/q05 có cạnh nghịch trong train; q02 chỉ có ngữ cảnh train, cạnh nghịch của nó nằm ở test.",
            "optionsNote": "Mỗi câu có một đáp án trong bốn lựa chọn. Distractors đã loại toàn bộ đuôi dương đã biết cho cùng head/relation trong train/valid/test; việc một liên kết vắng mặt không chứng minh nó sai ngoài benchmark.",
            "kindNote": "Animal/object/plant/concept là nhóm trình bày được biên soạn cho demo, không phải nhãn phân loại được dự đoán bởi model. Giữ nguyên English primary lemma, aliases, raw relation và source definition.",
            "protocol": "Reproduction lịch sử được liên kết là Source-Frozen RC. Web hiện tại sử dụng xếp hạng minh họa do biên soạn, không được sinh từ checkpoint. Dữ liệu, đáp án và bằng chứng đồ thị lấy từ nguồn WN9 đã pin.",
            "evidenceNote": "Đáp án dẫn tới đúng split và dòng nguồn. Cạnh nghịch là bằng chứng dữ liệu trực tiếp; các cạnh context chỉ cung cấp ngữ cảnh, không chứng minh cơ chế hoặc nguyên nhân điểm số của model.",
            "imageCredit": image_credit,
            "textCredit": "Tên, aliases và toàn bộ định nghĩa tiếng Anh giữ nguyên từ ent_name.txt/ent_dscp.txt của MoSE WN9 tại revision đã pin. Mỗi node có liên kết tới đúng dòng nguồn.",
            "sourceLinks": [
                {"title": "MoSE WN9 · pinned source", "url": SOURCE},
                {
                    "title": "WordNet · attribution and license",
                    "url": "https://wordnet.princeton.edu/license-and-commercial-use",
                },
                {
                    "title": "WN9 · source audit",
                    "url": "https://github.com/keiranoluv/Reproduce_DSoM/blob/main/docs/research/provenance/wn9-sources.md",
                },
                {"title": "DSoM · paper", "url": "https://arxiv.org/abs/2507.20738"},
                {
                    "title": "DSoM · author code",
                    "url": "https://github.com/OreOZhao/DSoM",
                },
                {
                    "title": "Reproduction · protocol",
                    "url": "https://github.com/keiranoluv/Reproduce_DSoM/blob/main/docs/research/methodology/protocol.md",
                },
            ],
        },
        "relations": [
            {"id": relation, "label": relation, "meaning": RELATION_MEANINGS[relation]}
            for relation in sorted(
                {e["relation"] for e in edges} | {q["relation"] for q in questions}
            )
        ],
        "entities": entities,
        "edges": edges,
        "questions": questions,
        "cases": cases,
    }
    provenance = {
        "exporter": "scripts/export_demo_data.py",
        "schemaVersion": 2,
        "source": {
            "repository": "https://github.com/OreOZhao/MoSE4MKGC",
            "revision": REVISION,
            "folder": "src_data/WN9",
        },
        "inputs": {
            **{
                f"split/{split}": {
                    "file": f"src_data/WN9/{split}",
                    "sha256": sha256(args.source_dir / split),
                    "rows": len(triples),
                }
                for split, triples in rows.items()
            },
            "entityNames": {
                "file": "src_data/WN9/ent_name.txt",
                "sha256": sha256(args.source_dir / "ent_name.txt"),
            },
            "descriptions": {
                "file": "src_data/WN9/ent_dscp.txt",
                "sha256": sha256(args.source_dir / "ent_dscp.txt"),
            },
            "entityMapping": {
                "file": "data/WN9/ent_id",
                "sha256": sha256(args.mapping),
            },
            "parentMapping": {
                "file": "data/WN18/ent_id",
                "sha256": sha256(args.parent_mapping),
            },
            "imageManifest": {
                "file": "data/WN18/img_features_manifest.json",
                "sha256": sha256(args.image_manifest),
            },
            "imageArchive": {
                "file": args.images.name,
                "bytes": args.images.stat().st_size,
                "note": "Archive not rehashed; selected members recorded with CRC32 and size.",
            },
        },
        "imageAlignment": {
            "mapping": "WN9 canonical n######## -> strip exactly one n -> WN18 ent_id -> sources_by_internal_id",
            "wn9RealImages": 6547,
            "wn9ImputedImages": 8,
            "parentManifestRealImages": manifest["real_image_count"],
            "parentManifestEntities": len(manifest["sources_by_internal_id"]),
            "note": "Parent aggregate coverage is for WN18, not WN9. WN9 coverage is recomputed from aligned IDs.",
        },
        "selection": "Fixed five original triples, all known positive tails, explicit train evidence, familiar animal/object/plant primary lemmas; deterministic fill to 150; seed-2022 option shuffling.",
        "benchmarkMutated": False,
        "inferencePerformed": False,
        "rankingSource": "Authored illustrative ranking in UI; not generated by a checkpoint.",
        "images": images,
        "credits": {"images": image_credit, "text": data["meta"]["textCredit"]},
    }
    output = args.output / "data"
    output.mkdir(parents=True, exist_ok=True)
    (output / "demo.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "entities": len(entities),
                "train_edges": len(edges),
                "questions": len(questions),
                "cases": len(cases),
                "question_splits": dict(Counter(q["split"] for q in questions)),
                "thumbnails": len(images),
                "thumbnail_bytes": sum(i["thumbnailBytes"] for i in images),
                "kinds": dict(Counter(e["kind"] for e in entities)),
                "source_images": data["meta"]["selectedRawImageCount"],
            }
        )
    )


if __name__ == "__main__":
    main()
