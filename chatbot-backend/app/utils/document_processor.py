"""
Document processing utilities for extracting text from various file formats
"""
import os
from typing import Optional
import PyPDF2
from docx import Document


class DocumentProcessor:
    """Handles text extraction from different document formats."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If PDF appears to be image-based (scanned) with no extractable text
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # If no text extracted, try alternative method (pypdf as fallback)
                if not text.strip() and total_pages > 0:
                    # Try using pypdf if available
                    try:
                        import pypdf
                        with open(file_path, 'rb') as file:
                            pdf_reader_pypdf = pypdf.PdfReader(file)
                            for page in pdf_reader_pypdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                    except ImportError:
                        pass
                    except Exception:
                        pass
                    
                    # If still no text, it's likely an image-based PDF
                    if not text.strip():
                        raise ValueError(
                            "This PDF appears to be image-based (scanned or photographed) with no extractable text. "
                            "PyPDF2 cannot extract text from images. Please:\n"
                            "1. Use OCR software to convert images to text first, OR\n"
                            "2. Upload a text-based PDF (created from Word/Google Docs), OR\n"
                            "3. Copy the transcript text and paste it into a .txt file, OR\n"
                            "4. Convert the PDF to a Word document (.docx) with text selectable."
                        )
                        
        except ValueError:
            # Re-raise our custom error
            raise
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a document based on its extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return DocumentProcessor.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return DocumentProcessor.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """
        Check if the file type is supported.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if file type is supported, False otherwise
        """
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in allowed_extensions
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        return {
            "name": file_name,
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "extension": file_ext,
            "path": file_path
        }

