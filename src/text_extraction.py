"""
Text Extraction Module

This module handles extraction of text from various file formats:
- PDF files using pdfplumber
- DOCX files using python-docx
- TXT files (plain text)
"""

import pdfplumber
from docx import Document
from typing import Optional
import io


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_content: Binary content of the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If PDF extraction fails
    """
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_content: Binary content of the DOCX file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If DOCX extraction fails
    """
    try:
        doc = Document(io.BytesIO(file_content))
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        return "\n".join(text).strip()
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")


def extract_text_from_txt(file_content: bytes) -> str:
    """
    Extract text from a TXT file.
    
    Args:
        file_content: Binary content of the TXT file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If TXT extraction fails
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                return text.strip()
            except UnicodeDecodeError:
                continue
        raise Exception("Could not decode text file with common encodings")
    except Exception as e:
        raise Exception(f"Error extracting text from TXT: {str(e)}")


def extract_text(file_content: bytes, file_name: str) -> Optional[str]:
    """
    Main function to extract text from any supported file format.
    
    Args:
        file_content: Binary content of the file
        file_name: Name of the file (used to determine file type)
        
    Returns:
        Extracted text as a string, or None if file type is not supported
    """
    file_name_lower = file_name.lower()
    
    if file_name_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif file_name_lower.endswith('.docx') or file_name_lower.endswith('.doc'):
        return extract_text_from_docx(file_content)
    elif file_name_lower.endswith('.txt'):
        return extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type: {file_name}. Supported formats: PDF, DOCX, TXT")





