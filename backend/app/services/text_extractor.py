"""
Text extraction service for multiple file formats.
Supports TXT, PDF, DOCX, and image files (OCR).
"""
import os
import threading
from typing import Optional, Dict, Any
from pathlib import Path


class TextExtractor:
    """Service for extracting text from various file formats."""

    def __init__(self):
        """Initialize the text extractor."""
        self.extractors = {
            'txt': self._extract_txt,
            'pdf': self._extract_pdf,
            'docx': self._extract_docx,
            'png': self._extract_image,
            'jpg': self._extract_image,
            'jpeg': self._extract_image,
        }
        # Cached PaddleOCR instance (created once - instantiation is slow)
        # and a lock so concurrent requests do not share the predictor
        # unsafely.
        self._ocr = None
        self._ocr_lock = threading.Lock()

    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with extracted text and metadata
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                'success': False,
                'text': '',
                'error': f'File not found: {file_path}',
                'format': None
            }

        # Get file extension
        file_ext = Path(file_path).suffix.lower().lstrip('.')

        # Get appropriate extractor
        extractor = self.extractors.get(file_ext)

        if extractor is None:
            return {
                'success': False,
                'text': '',
                'error': f'Unsupported file format: {file_ext}',
                'format': file_ext
            }

        # Extract text
        try:
            text = extractor(file_path)
            if text and text.strip():
                return {
                    'success': True,
                    'text': text.strip(),
                    'error': None,
                    'format': file_ext,
                    'length': len(text.strip())
                }
            else:
                return {
                    'success': False,
                    'text': '',
                    'error': 'No text extracted (empty or failed)',
                    'format': file_ext
                }
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'error': f'Extraction failed: {str(e)}',
                'format': file_ext
            }

    def _extract_txt(self, file_path: str) -> str:
        """
        Extract text from TXT file.

        Args:
            file_path: Path to the TXT file

        Returns:
            Extracted text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text
        """
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF extraction. Install with: pip install PyPDF2")

    def _extract_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file.

        Args:
            file_path: Path to the DOCX file

        Returns:
            Extracted text
        """
        try:
            import docx
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            return '\n'.join(text)
        except ImportError:
            raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")

    def _extract_image(self, file_path: str) -> str:
        """
        Extract text from image using PaddleOCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text
        """
        try:
            # Skip PaddleOCR's model-host connectivity check (speeds up
            # repeated inference calls in the API service).
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr is required for image OCR. Install with: pip install paddleocr")

        with self._ocr_lock:
            if self._ocr is None:
                # PaddleOCR 3.x defaults to the oneDNN (MKL-DNN) CPU backend,
                # which can crash on some CPUs - disable it for stability.
                # Older versions do not accept the kwarg, so fall back.
                try:
                    self._ocr = PaddleOCR(
                        use_angle_cls=True, lang='en', enable_mkldnn=False
                    )
                except (TypeError, ValueError):
                    self._ocr = PaddleOCR(use_angle_cls=True, lang='en')

            try:
                result = self._ocr.ocr(file_path, cls=True)  # PaddleOCR 2.x API
            except TypeError:
                result = self._ocr.ocr(file_path)  # PaddleOCR 3.x (no 'cls' kwarg)

            # Extract text from OCR results.
            # PaddleOCR 3.x returns dict-like OCRResult objects with a
            # 'rec_texts' list; 2.x returns [box, (text, score)] pairs.
            text_lines = []
            if result and result[0]:
                page = result[0]
                if isinstance(page, dict):
                    text_lines = [
                        t.strip()
                        for t in (page.get('rec_texts') or [])
                        if t and t.strip()
                    ]
                else:
                    for line in page:
                        if line and len(line) > 0:
                            text_lines.append(line[1][0])

            return '\n'.join(text_lines)
