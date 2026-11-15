import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from pypdf import PdfWriter
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

    def test_ocr_extracts_text_from_pdf_pages_without_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf = io.BytesIO()
        writer.write(pdf)

        pymupdf = MagicMock()
        pdf_document = MagicMock()
        pymupdf.open.return_value.__enter__.return_value = pdf_document
        pixmap = Mock()
        pixmap.tobytes.return_value = b"rendered page"
        pdf_document.load_page.return_value.get_pixmap.return_value = pixmap

        tesseract_not_found = type("TesseractNotFoundError", (RuntimeError,), {})
        pytesseract = Mock()
        pytesseract.TesseractNotFoundError = tesseract_not_found
        pytesseract.pytesseract = types.SimpleNamespace(tesseract_cmd="tesseract")
        pytesseract.get_tesseract_version.return_value = "5.0"
        pytesseract.image_to_string.return_value = "Pump P-101 inspection interval"

        with patch.dict(
            "sys.modules",
            {"pymupdf": pymupdf, "pytesseract": pytesseract},
        ):
            sections = extract_sections("scanned.pdf", pdf.getvalue())

            self.assertEqual(
                sections,
                [("Page 1", "Pump P-101 inspection interval")],
            )
            pytesseract.image_to_string.assert_called_once()
            self.assertEqual(
                pytesseract.image_to_string.call_args.args[0].read(),
                b"rendered page",
            )
            pymupdf.open.assert_called_once_with(
                stream=pdf.getvalue(), filetype="pdf"
            )

            pytesseract.get_tesseract_version.side_effect = tesseract_not_found()
            with self.assertRaisesRegex(RuntimeError, "Tesseract OCR was not found"):
                extract_sections("scanned.pdf", pdf.getvalue())

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
