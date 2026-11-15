"""Validate the WN9 public teaching slice and its source-backed evidence.

CI checks the committed bundle without raw downloads or ML dependencies. Local
checks additionally compare every source record, source gloss and image mapping
against the unmodified reproduction inputs when those files are available.
"""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
REVISION = "192d9ff2882b5be6ef8239a37cbd90c5dd102b5a"
INVERSES = {"_hypernym": "_hyponym", "_has_part": "_part_of", "_part_of": "_has_part"}
EXPECTED_QUESTIONS = {
    "q01": ("test", 392, "n02112826", "_hypernym", "n02084071", 6465),
    "q02": ("valid", 1062, "n01816887", "_hypernym", "n01503061", None),
    "q03": ("test", 219, "n03903424", "_part_of", "n02834778", 9165),
    "q04": ("test", 756, "n02958343", "_has_part", "n02761834", 3629),
    "q05": ("test", 300, "n02913152", "_has_part", "n04546855", 6821),
}


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((DEMO / "data/demo.json").read_text(encoding="utf-8"))
        cls.provenance = json.loads(
            (DEMO / "data/provenance.json").read_text(encoding="utf-8")
        )
        cls.entities = {e["id"]: e for e in cls.data["entities"]}
        cls.edges = {e["id"]: e for e in cls.data["edges"]}

    def test_release_scope_and_provenance(self):
        self.assertEqual(self.data["meta"]["dataset"], "WN9")
        self.assertEqual(self.data["meta"]["sourceRevision"], REVISION)
        self.assertEqual(self.provenance["source"]["revision"], REVISION)
        self.assertFalse(self.provenance["inferencePerformed"])
        self.assertFalse(self.data["meta"]["inferencePerformed"])
        self.assertFalse(self.provenance["benchmarkMutated"])
        self.assertEqual(len(self.entities), 150)
        self.assertEqual(len(self.data["entities"]), 150)
        self.assertEqual(len(self.data["questions"]), 5)
        self.assertEqual(len(self.data["cases"]), 5)
        self.assertEqual(
            Counter(q["split"] for q in self.data["questions"]), {"test": 4, "valid": 1}
        )
        self.assertEqual(
            self.data["meta"]["splitCounts"],
            {"train": 11741, "valid": 1337, "test": 1319},
        )
        self.assertEqual(
            self.data["meta"]["sourceImageCoverage"],
            {"real": 6547, "imputed": 8, "total": 6555},
        )
        for field, key in (
            ("entities", "entityCount"),
            ("edges", "edgeCount"),
            ("questions", "questionCount"),
            ("cases", "caseCount"),
        ):
            self.assertEqual(len(self.data[field]), self.data["meta"][key])
            ids = [item["id"] for item in self.data[field]]
            self.assertEqual(len(ids), len(set(ids)))
        for entity in self.entities.values():
            self.assertRegex(entity["id"], r"^n\d{8}$")
            self.assertIn(entity["kind"], {"animal", "object", "plant", "concept"})
            self.assertTrue(entity["label"] and entity["description"])
            self.assertIn(REVISION, entity["sourceUrl"])
            self.assertRegex(entity["sourceUrl"], r"/ent_name\.txt#L\d+$")
        for relation in self.data["relations"]:
            self.assertEqual(relation["id"], relation["label"])
            self.assertTrue(relation["meaning"])

    def test_graph_only_contains_traceable_training_edges(self):
        graph = set()
        relations = {r["id"] for r in self.data["relations"]}
        for edge in self.edges.values():
            self.assertEqual(edge["split"], "train")
            self.assertIn(edge["source"], self.entities)
            self.assertIn(edge["target"], self.entities)
            self.assertIn(edge["relation"], relations)
            self.assertIsInstance(edge["sourceRow"], int)
            self.assertGreaterEqual(edge["sourceRow"], 1)
            self.assertLessEqual(edge["sourceRow"], 11741)
            self.assertTrue(edge["sourceUrl"].endswith(f"/train#L{edge['sourceRow']}"))
            self.assertIn(REVISION, edge["sourceUrl"])
            graph.add((edge["source"], edge["relation"], edge["target"]))
        self.assertEqual(len(graph), len(self.edges))
        for question in self.data["questions"]:
            self.assertNotIn(
                (question["head"], question["relation"], question["answer"]), graph
            )

    def test_five_original_queries_have_one_known_positive_choice(self):
        for question in self.data["questions"]:
            with self.subTest(question=question["id"]):
                split, row, head, relation, answer, _ = EXPECTED_QUESTIONS[
                    question["id"]
                ]
                self.assertEqual(
                    (
                        question["split"],
                        question["head"],
                        question["relation"],
                        question["answer"],
                    ),
                    (split, head, relation, answer),
                )
                self.assertEqual(question["evidence"]["answerRecord"]["row"], row)
                options, accepted = (
                    set(question["options"]),
                    set(question["acceptedIds"]),
                )
                self.assertEqual(len(question["options"]), 4)
                self.assertEqual(len(options), 4)
                self.assertLessEqual(
                    options | accepted | {question["head"]}, self.entities.keys()
                )
                self.assertEqual(options & accepted, {question["answer"]})
                self.assertTrue(question["prompt"] and question["explanation"])

    def test_evidence_separates_named_inverse_from_context(self):
        for question in self.data["questions"]:
            evidence = question["evidence"]
            answer_record = evidence["answerRecord"]
            triple = (question["head"], question["relation"], question["answer"])
            self.assertEqual(answer_record["triple"], list(triple))
            self.assertEqual(answer_record["split"], question["split"])
            self.assertTrue(
                answer_record["sourceUrl"].endswith(
                    f"/{question['split']}#L{answer_record['row']}"
                )
            )
            self.assertIn(REVISION, answer_record["sourceUrl"])
            evidence_entities = set(evidence["entityIds"])
            self.assertGreaterEqual(len(evidence_entities), 5)
            self.assertLessEqual(len(evidence_entities), 10)
            self.assertLessEqual(evidence_entities, self.entities.keys())
            self.assertLessEqual(
                {question["head"], question["answer"]}, evidence_entities
            )
            self.assertGreaterEqual(len(evidence["contextEdgeIds"]), 2)
            self.assertLessEqual(len(evidence["contextEdgeIds"]), 6)
            self.assertFalse(
                set(evidence["inverseEdgeIds"]) & set(evidence["contextEdgeIds"])
            )
            for edge_id in evidence["contextEdgeIds"] + evidence["inverseEdgeIds"]:
                self.assertIn(edge_id, self.edges)
                edge = self.edges[edge_id]
                self.assertLessEqual(
                    {edge["source"], edge["target"]}, evidence_entities
                )
                self.assertEqual(edge["split"], "train")
            expected_inverse_row = EXPECTED_QUESTIONS[question["id"]][-1]
            if expected_inverse_row is None:
                self.assertEqual(evidence["inverseEdgeIds"], [])
                self.assertEqual(evidence["kind"], "context")
            else:
                self.assertEqual(len(evidence["inverseEdgeIds"]), 1)
                edge = self.edges[evidence["inverseEdgeIds"][0]]
                self.assertEqual(
                    (edge["source"], edge["relation"], edge["target"]),
                    (
                        question["answer"],
                        INVERSES[question["relation"]],
                        question["head"],
                    ),
                )
                self.assertEqual(edge["sourceRow"], expected_inverse_row)
                self.assertEqual(evidence["kind"], "inverse")
            self.assertTrue(evidence["explanation"])
            self.assertIn(str(answer_record["row"]), evidence["explanation"])
            if expected_inverse_row is not None:
                self.assertIn(str(expected_inverse_row), evidence["explanation"])

    def test_cases_preserve_synset_ambiguity_and_valid_links(self):
        questions = {q["id"] for q in self.data["questions"]}
        for case in self.data["cases"]:
            self.assertLessEqual(set(case["entityIds"]), self.entities.keys())
            if "questionId" in case:
                self.assertIn(case["questionId"], questions)
            self.assertTrue(case["observation"] and case["lesson"])
        mechanical, anatomical = self.entities["n03501614"], self.entities["n05538625"]
        self.assertEqual(mechanical["label"], "head")
        self.assertEqual(anatomical["label"], "head")
        self.assertNotEqual(mechanical["description"], anatomical["description"])
        graph = {(e["source"], e["relation"], e["target"]) for e in self.edges.values()}
        self.assertIn(("n03501614", "_part_of", "n03481172"), graph)
        self.assertIn(("n05538625", "_part_of", "n00015388"), graph)

    def test_thumbnails_have_per_synset_alignment_integrity_and_small_budget(self):
        images = self.provenance["images"]
        self.assertEqual(len(images), 48)
        self.assertEqual(len(images), self.data["meta"]["thumbnailCount"])
        for image in images:
            path = (DEMO / image["thumbnail"]).resolve()
            self.assertTrue(path.is_relative_to(DEMO.resolve()))
            content = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(), image["thumbnailSha256"]
            )
            self.assertEqual(len(content), image["thumbnailBytes"])
            self.assertEqual(content[:4], b"RIFF")
            self.assertEqual(content[8:12], b"WEBP")
            entity = self.entities[image["entity"]]
            self.assertEqual(entity["image"], image["thumbnail"])
            self.assertEqual(entity["imageSource"], image["archiveMember"])
            self.assertEqual(image["parentEntity"], entity["id"][1:])
            self.assertTrue(
                image["archiveMember"].startswith(f"wn18-images/{entity['id']}/")
            )
        bundled = {
            entity["image"] for entity in self.entities.values() if entity["image"]
        }
        self.assertEqual(bundled, {image["thumbnail"] for image in images})
        self.assertEqual(
            {str(p.relative_to(DEMO)) for p in (DEMO / "assets/images").glob("*.webp")},
            bundled,
        )
        self.assertLess(sum(image["thumbnailBytes"] for image in images), 1_000_000)
        size = sum(p.stat().st_size for p in DEMO.rglob("*") if p.is_file())
        self.assertLess(size, 2 * 1024 * 1024)

    def test_original_records_glosses_and_named_inverses_when_available(self):
        paths = {
            split: ROOT / "src_data/WN9" / split for split in ("train", "valid", "test")
        }
        if not all(path.is_file() for path in paths.values()):
            self.skipTest("Raw WN9 is intentionally not committed")
        rows = {}
        for split, path in paths.items():
            content = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                self.provenance["inputs"][f"split/{split}"]["sha256"],
            )
            rows[split] = [
                tuple(line.split("\t")) for line in content.decode().splitlines()
            ]
        row_sets = {split: set(triples) for split, triples in rows.items()}
        union = set().union(*row_sets.values())
        for edge in self.edges.values():
            self.assertEqual(
                (edge["source"], edge["relation"], edge["target"]),
                rows["train"][edge["sourceRow"] - 1],
            )
        for question in self.data["questions"]:
            triple = (question["head"], question["relation"], question["answer"])
            record = question["evidence"]["answerRecord"]
            self.assertEqual(triple, rows[record["split"]][record["row"] - 1])
            self.assertNotIn(triple, row_sets["train"])
            if question["split"] == "test":
                self.assertNotIn(triple, row_sets["valid"])
            actual = {
                tail
                for head, relation, tail in union
                if (head, relation) == (question["head"], question["relation"])
            }
            self.assertEqual(actual, set(question["acceptedIds"]))
            inverse = (
                question["answer"],
                INVERSES[question["relation"]],
                question["head"],
            )
            self.assertEqual(
                bool(question["evidence"]["inverseEdgeIds"]),
                inverse in row_sets["train"],
            )
        for key, filename in (
            ("entityNames", "ent_name.txt"),
            ("descriptions", "ent_dscp.txt"),
        ):
            path = ROOT / "src_data/WN9" / filename
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                self.provenance["inputs"][key]["sha256"],
            )
        names = dict(
            line.split("\t", 1)
            for line in (ROOT / "src_data/WN9/ent_name.txt").read_text().splitlines()
        )
        definitions = dict(
            line.split("\t", 1)
            for line in (ROOT / "src_data/WN9/ent_dscp.txt").read_text().splitlines()
        )
        for entity in self.entities.values():
            self.assertEqual(entity["label"], names[entity["id"]].split(", ")[0])
            self.assertEqual(entity["aliases"], names[entity["id"]].split(", ")[1:])
            self.assertEqual(entity["description"], definitions[entity["id"][1:]])

    def test_parent_image_mapping_when_available(self):
        mapping_file = ROOT / "data/WN18/ent_id"
        manifest_file = ROOT / "data/WN18/img_features_manifest.json"
        if not mapping_file.is_file() or not manifest_file.is_file():
            self.skipTest("Raw parent image manifest is intentionally not committed")
        mapping = {
            entity: int(index)
            for entity, index in (
                line.split("\t") for line in mapping_file.read_text().splitlines()
            )
        }
        manifest = json.loads(manifest_file.read_text())
        self.assertEqual(
            hashlib.sha256(mapping_file.read_bytes()).hexdigest(),
            self.provenance["inputs"]["parentMapping"]["sha256"],
        )
        self.assertEqual(
            hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
            self.provenance["inputs"]["imageManifest"]["sha256"],
        )
        for entity in self.entities.values():
            source = manifest["sources_by_internal_id"][mapping[entity["id"][1:]]]
            self.assertEqual(entity["imageSource"], source or None)
            self.assertEqual(entity["imageAvailable"], bool(source))
        for image in self.provenance["images"]:
            self.assertEqual(image["parentInternalId"], mapping[image["entity"][1:]])


if __name__ == "__main__":
    unittest.main()
