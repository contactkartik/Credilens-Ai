"""
Script to create test sample files for text extraction testing.
"""
import os

# Create test_samples directory
os.makedirs('test_samples', exist_ok=True)

# TXT file (already created)
print("TXT sample already exists")

# Create a simple DOCX file if python-docx is available
try:
    from docx import Document
    doc = Document()
    doc.add_heading('Sample Document', 0)
    doc.add_paragraph('This is a sample DOCX file for testing text extraction.')
    doc.add_paragraph('The CrediLens AI project uses advanced NLP techniques.')
    doc.add_paragraph('This document contains multiple paragraphs.')
    doc.save('test_samples/sample.docx')
    print("Created sample.docx")
except ImportError:
    print("python-docx not available, skipping DOCX sample creation")

# Create a simple PDF file if reportlab is available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas("test_samples/sample.pdf", pagesize=letter)
    c.drawString(100, 750, "Sample PDF Document")
    c.drawString(100, 730, "This is a sample PDF for testing text extraction.")
    c.drawString(100, 710, "The CrediLens AI project detects information credibility.")
    c.drawString(100, 690, "This PDF contains multiple lines of text.")
    c.save()
    print("Created sample.pdf")
except ImportError:
    print("reportlab not available, skipping PDF sample creation")

# Create a simple image with text if PIL is available
try:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (400, 100), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Sample Image for OCR Testing", fill='black')
    d.text((10, 40), "CrediLens AI Text Extraction", fill='black')
    d.text((10, 70), "This is an image sample", fill='black')
    img.save('test_samples/sample.png')
    print("Created sample.png")
except ImportError:
    print("PIL not available, skipping image sample creation")

print("\nTest samples created in tests/test_samples/")
