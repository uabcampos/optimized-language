#!/usr/bin/env python3
"""
Extract first 10 pages from a PDF for testing purposes.
"""

import fitz  # PyMuPDF
import sys
import os

def extract_first_10_pages(input_pdf: str, output_pdf: str):
    """Extract first 10 pages from input PDF and save to output PDF."""
    try:
        # Open the input PDF
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        
        print(f"Input PDF has {total_pages} pages")
        
        # Create new document with first 10 pages
        new_doc = fitz.open()
        
        # Extract first 10 pages (or all pages if less than 10)
        pages_to_extract = min(10, total_pages)
        
        for page_num in range(pages_to_extract):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            print(f"Extracted page {page_num + 1}")
        
        # Save the new PDF
        new_doc.save(output_pdf)
        new_doc.close()
        doc.close()
        
        print(f"✅ Successfully created {output_pdf} with {pages_to_extract} pages")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting pages: {e}")
        return False

if __name__ == "__main__":
    input_file = "P50MD017338_Center.SciBiosFacil.09.04.25v3.pdf"
    output_file = "test_first_10_pages.pdf"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    success = extract_first_10_pages(input_file, output_file)
    if success:
        print(f"🎉 Test file ready: {output_file}")
    else:
        print("❌ Failed to extract pages")
        sys.exit(1)
