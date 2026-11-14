import tempfile
import unittest
from pathlib import Path

from rag import DocumentStore, extract_sections, split_into_chunks


class ExtractionTests(unittest.TestCase):
    def test_extracts_text_and_csv(self):
        self.assertEqual(
            extract_sections("manual.txt", b"Pump P-101 inspection interval"),
            [("Document", "Pump P-101 inspection interval")],
        )
        self.assertEqual(
            extract_sections("register.csv", b"Tag,Description\nP-101,Transfer pump"),
            [("CSV", "Tag | Description\nP-101 | Transfer pump")],
        )

    def test_rejects_unsupported_files(self):
        with self.assertRaises(ValueError):
            extract_sections("drawing.dwg", b"")

    def test_chunk_overlap_and_validation(self):
        chunks = split_into_chunks("one two three four five", size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five"])
        with self.assertRaises(ValueError):
            split_into_chunks("text", size=2, overlap=2)


class DocumentStoreTests(unittest.TestCase):
    def test_indexes_deduplicates_and_retrieves_with_location(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.db")
            content = b"Pump P-101 inspection interval is every six months."
            self.assertTrue(store.add_document("inspection.txt", content))
            self.assertFalse(store.add_document("renamed-copy.txt", content))

            results = store.search("P-101 inspection interval")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].document_name, "inspection.txt")
            self.assertEqual(results[0].location, "Document")
            self.assertIn("six months", results[0].text)
            self.assertEqual(store.list_documents(), [("inspection.txt", 1)])

    def test_unrelated_search_returns_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.db")
            store.add_document("inspection.txt", b"Pump inspection every six months.")
            self.assertEqual(store.search("bridge clearance"), [])

    def test_document_filename_is_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.db")
            store.add_document("P-101-maintenance.txt", b"Scheduled work is documented here.")
            results = store.search("P-101 maintenance")
            self.assertEqual(results[0].document_name, "P-101-maintenance.txt")


if __name__ == "__main__":
    unittest.main()
