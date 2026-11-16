import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rag import (
    DocumentStore,
    _agent_chat_completion,
    agentic_answer,
    extract_sections,
    split_into_chunks,
)


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

    def test_docling_converts_structured_files_and_releases_input(self):
        docling = types.ModuleType("docling")
        docling.__path__ = []
        datamodel = types.ModuleType("docling.datamodel")
        datamodel.__path__ = []
        base_models = types.ModuleType("docling.datamodel.base_models")
        base_models.DocumentStream = Mock(side_effect=lambda **kwargs: kwargs)
        document_converter = types.ModuleType("docling.document_converter")
        document_converter.DocumentConverter = Mock()
        converter = Mock()
        conversion = Mock()
        conversion.document.export_to_markdown.return_value = "# Pump P-101"
        converter.convert.return_value = conversion
        document_converter.DocumentConverter.return_value = converter

        modules = {
            "docling": docling,
            "docling.datamodel": datamodel,
            "docling.datamodel.base_models": base_models,
            "docling.document_converter": document_converter,
        }
        with patch.dict(sys.modules, modules):
            self.assertEqual(
                extract_sections("manual.pdf", b"document bytes"),
                [("Document", "# Pump P-101")],
            )

        base_models.DocumentStream.assert_called_once()
        stream = base_models.DocumentStream.call_args.kwargs["stream"]
        self.assertEqual(stream.read(), b"document bytes")
        conversion.input.release.assert_called_once()

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

    def test_agent_searches_local_documents_and_returns_cited_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.db")
            store.add_document(
                "inspection.txt",
                b"Pump P-101 inspection interval is every six months.",
            )
            tool_call = {
                "id": "call-search",
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": json.dumps({"query": "P-101 inspection"}),
                },
            }
            refined_tool_call = {
                "id": "call-refined-search",
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": json.dumps({"query": "P-101 inspection interval"}),
                },
            }
            responses = [
                {"choices": [{"message": {"tool_calls": [tool_call]}}]},
                {"choices": [{"message": {"tool_calls": [refined_tool_call]}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Pump P-101 is inspected every six months [S1]."
                            }
                        }
                    ]
                },
            ]

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch("rag._agent_chat_completion", side_effect=responses):
                    response = agentic_answer("When is P-101 inspected?", store)

            self.assertIn("[S1]", response.answer)
            self.assertEqual(
                response.searches,
                ["P-101 inspection", "P-101 inspection interval"],
            )
            self.assertEqual(response.sources[0].document_name, "inspection.txt")

    def test_agent_search_limit_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.db")
            with self.assertRaises(ValueError):
                agentic_answer("Question", store, max_searches=0)

    def test_chat_completion_uses_bearer_authentication(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"choices": []}'
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("rag.urllib.request.urlopen", return_value=response) as urlopen:
                _agent_chat_completion({"messages": []})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
