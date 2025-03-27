"""
Tests for the PDF Reader module.
"""

import os
import pytest
from pathlib import Path

from src.pdf_extractor.pdf_reader import PDFReader, extract_text_from_pdf


# Get the sample PDF path from the root directory
@pytest.fixture
def sample_pdf_path():
    # This assumes the test is run from the project root
    return Path('TOPIK 기출문제.pdf')


def test_pdf_reader_initialization(sample_pdf_path):
    """Test that the PDFReader can be initialized with a valid PDF path."""
    reader = PDFReader(sample_pdf_path)
    assert reader.pdf_path == sample_pdf_path
    

def test_pdf_reader_file_not_found():
    """Test that PDFReader raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        PDFReader('non_existent_file.pdf')


def test_extract_text_with_pdfplumber(sample_pdf_path):
    """Test extracting text using pdfplumber."""
    reader = PDFReader(sample_pdf_path)
    pages = reader.extract_text_with_pdfplumber()
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    # Check that at least one page has content
    assert any(len(page) > 0 for page in pages)


def test_extract_text_with_pypdf2(sample_pdf_path):
    """Test extracting text using PyPDF2."""
    reader = PDFReader(sample_pdf_path)
    pages = reader.extract_text_with_pypdf2()
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    # Check that at least one page has content
    assert any(len(page) > 0 for page in pages)


def test_extract_text_function(sample_pdf_path):
    """Test the convenience function."""
    pages = extract_text_from_pdf(sample_pdf_path)
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    # Check that at least one page has content
    assert any(len(page) > 0 for page in pages) 