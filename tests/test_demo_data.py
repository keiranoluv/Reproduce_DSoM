"""Checks for the public demo bundle; no training dependencies or raw downloads."""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((DEMO / "data/demo.json").read_text(encoding="utf-8"))
        cls.provenance = json.loads(
            (DEMO / "data/provenance.json").read_text(encoding="utf-8")
        )
        cls.entities = {entity["id"]: entity for entity in cls.data["entities"]}

    def test_release_scope_and_references(self):
        self.assertEqual(len(self.entities), 150)
        self.assertEqual(len(self.data["entities"]), 150)
        self.assertEqual(len(self.data["questions"]), 10)
        self.assertEqual(len(self.data["cases"]), 5)
        self.assertEqual(
            Counter(q["split"] for q in self.data["questions"]), {"valid": 2, "test": 8}
        )
        for field, key in (
            ("entities", "entityCount"),
            ("edges", "edgeCount"),
            ("questions", "questionCount"),
            ("cases", "caseCount"),
        ):
            self.assertEqual(len(self.data[field]), self.data["meta"][key])
        for collection in ("edges", "questions", "cases"):
            ids = [item["id"] for item in self.data[collection]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_graph_has_only_training_context(self):
        graph = set()
        relations = {relation["id"] for relation in self.data["relations"]}
        for edge in self.data["edges"]:
            self.assertEqual(edge["split"], "train")
            self.assertIn(edge["source"], self.entities)
            self.assertIn(edge["target"], self.entities)
            self.assertIn(edge["relation"], relations)
            graph.add((edge["source"], edge["relation"], edge["target"]))
        for question in self.data["questions"]:
            self.assertNotIn(
                (question["head"], question["relation"], question["answer"]), graph
            )

    def test_questions_have_one_known_positive_choice(self):
        for question in self.data["questions"]:
            with self.subTest(question=question["id"]):
                self.assertIn(question["head"], self.entities)
                options = set(question["options"])
                accepted = set(question["acceptedIds"])
                self.assertEqual(len(question["options"]), 4)
                self.assertEqual(len(options), 4)
                self.assertLessEqual(options | accepted, self.entities.keys())
                self.assertEqual(options & accepted, {question["answer"]})
                self.assertTrue(question["prompt"] and question["explanation"])

    def test_case_links_and_real_missing_modalities(self):
        questions = {q["id"] for q in self.data["questions"]}
        for case in self.data["cases"]:
            self.assertLessEqual(set(case["entityIds"]), self.entities.keys())
            if "questionId" in case:
                self.assertIn(case["questionId"], questions)
            self.assertTrue(case["observation"] and case["lesson"])
        for entity in ("Dario_Argento", "Jerry_Bruckheimer", "Julie_Bowen"):
            self.assertFalse(self.entities[entity]["textAvailable"])
            self.assertIsNone(self.entities[entity]["description"])
        self.assertFalse(self.entities["This_Is_Our_House"]["imageAvailable"])

    def test_thumbnail_integrity_and_bundle_budget(self):
        images = self.provenance["images"]
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
        bundled = {
            entity["image"] for entity in self.entities.values() if entity["image"]
        }
        self.assertEqual(bundled, {image["thumbnail"] for image in images})
        size = sum(p.stat().st_size for p in DEMO.rglob("*") if p.is_file())
        self.assertLess(
            size, 2 * 1024 * 1024, "Keep this static teaching demo below 2 MiB"
        )

    def test_original_splits_when_available(self):
        """Verify source fidelity locally; CI uses the committed fingerprint manifest."""
        paths = {
            split: ROOT / "src_data/MKG-Y" / split
            for split in ("train", "valid", "test")
        }
        if not all(path.is_file() for path in paths.values()):
            self.skipTest("Raw MKG-Y is intentionally not committed")
        rows = {}
        for split, path in paths.items():
            content = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                self.provenance["inputs"][f"split/{split}"]["sha256"],
            )
            rows[split] = {
                tuple(line.split("\t")) for line in content.decode().splitlines()
            }
        union = rows["train"] | rows["valid"] | rows["test"]
        for edge in self.data["edges"]:
            self.assertIn(
                (edge["source"], edge["relation"], edge["target"]), rows["train"]
            )
        for question in self.data["questions"]:
            triple = (question["head"], question["relation"], question["answer"])
            self.assertIn(triple, rows[question["split"]])
            self.assertNotIn(triple, rows["train"])
            if question["split"] == "test":
                self.assertNotIn(triple, rows["valid"])
            actual = {
                tail
                for head, relation, tail in union
                if (head, relation) == (question["head"], question["relation"])
            }
            self.assertEqual(actual, set(question["acceptedIds"]))


if __name__ == "__main__":
    unittest.main()
