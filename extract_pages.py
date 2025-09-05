#!/usr/bin/env python3
"""
Extract first 5 pages from a PDF for testing purposes.
"""

import fitz  # PyMuPDF
import sys

def extract_pages(input_pdf, output_pdf, num_pages=5):
    """Extract first num_pages from input_pdf and save to output_pdf."""
    doc = fitz.open(input_pdf)
    
    # Create new document with first num_pages
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=0, to_page=min(num_pages-1, len(doc)-1))
    
    # Save the new document
    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()
    
    print(f"Extracted first {num_pages} pages to {output_pdf}")

if __name__ == "__main__":
    input_file = "P50MD017338_Center.SciBiosFacil.06.20.25.pdf"
    output_file = "test_sample.pdf"
    
    extract_pages(input_file, output_file, 5)
