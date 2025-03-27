"""
PDF Reader Module

This module handles extraction of text from PDF files with specific
support for Korean language content.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union

import PyPDF2
import pdfplumber


logger = logging.getLogger(__name__)


class PDFReader:
    """Class to extract text from PDF files."""
    
    def __init__(self, pdf_path: Union[str, Path]):
        """
        Initialize the PDF reader.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        logger.info(f"Initialized PDF reader for: {self.pdf_path}")
    
    def extract_text_with_pdfplumber(self) -> List[str]:
        """
        Extract text from PDF using pdfplumber.
        Better for maintaining layout and handling non-Latin scripts.
        
        Returns:
            List of text content for each page
        """
        pages_text = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages_text.append(text)
                    logger.debug(f"Extracted page {i+1}/{total_pages} with {len(text)} characters")
                
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            raise
        
        return pages_text
    
    def extract_text_with_pypdf2(self) -> List[str]:
        """
        Extract text from PDF using PyPDF2.
        Backup method if pdfplumber fails.
        
        Returns:
            List of text content for each page
        """
        pages_text = []
        
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                for i in range(total_pages):
                    text = reader.pages[i].extract_text() or ""
                    pages_text.append(text)
                    logger.debug(f"Extracted page {i+1}/{total_pages} with {len(text)} characters")
                
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            raise
        
        return pages_text
    
    def extract_text(self, prefer_method: str = "pdfplumber") -> List[str]:
        """
        Extract text from PDF using the preferred method,
        falling back to the alternative if needed.
        
        Args:
            prefer_method: Preferred method to use ("pdfplumber" or "pypdf2")
            
        Returns:
            List of text content for each page
        """
        try:
            if prefer_method.lower() == "pdfplumber":
                return self.extract_text_with_pdfplumber()
            else:
                return self.extract_text_with_pypdf2()
        except Exception as e:
            logger.warning(f"Failed to extract text with {prefer_method}: {e}")
            
            # Try the alternative method
            alternative = "pypdf2" if prefer_method.lower() == "pdfplumber" else "pdfplumber"
            logger.info(f"Trying alternative method: {alternative}")
            
            if alternative == "pdfplumber":
                return self.extract_text_with_pdfplumber()
            else:
                return self.extract_text_with_pypdf2()


def extract_text_from_pdf(pdf_path: Union[str, Path], prefer_method: str = "pdfplumber") -> List[str]:
    """
    Convenience function to extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        prefer_method: Preferred extraction method
        
    Returns:
        List of text content for each page
    """
    reader = PDFReader(pdf_path)
    return reader.extract_text(prefer_method) 