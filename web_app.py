#!/usr/bin/env python3
"""
Web interface for the Smart PDF Language Flagger
"""

import streamlit as st
import json
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple
import pandas as pd
import zipfile
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Smart PDF Language Flagger",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_json_file(file_path: str) -> dict:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return {}

def save_json_file(file_path: str, data: dict) -> bool:
    """Save JSON file safely."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving {file_path}: {e}")
        return False

def run_processing(input_file: str, flagged_terms: List[str], replacements: Dict[str, str], 
                  outdir: str, style: str, model: str, temperature: float) -> Tuple[bool, str, List[dict]]:
    """Run the processing script and return results."""
    try:
        # Create temporary JSON files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(flagged_terms, f, indent=2)
            flagged_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(replacements, f, indent=2)
            replacements_file = f.name
        
        # Run the processing script
        cmd = [
            sys.executable, "smart_flag_pdf.py",
            input_file,
            "--flagged", flagged_file,
            "--map", replacements_file,
            "--outdir", outdir,
            "--style", style,
            "--model", model,
            "--temperature", str(temperature)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Clean up temporary files
        os.unlink(flagged_file)
        os.unlink(replacements_file)
        
        if result.returncode == 0:
            # Load results
            csv_path = os.path.join(outdir, "flag_report.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                hits = df.to_dict('records')
            else:
                hits = []
            
            return True, result.stdout, hits
        else:
            return False, result.stderr, []
            
    except Exception as e:
        return False, str(e), []

def main():
    st.title("📝 Smart PDF Language Flagger")
    st.markdown("Upload documents and configure language flagging settings")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("LLM Settings")
        model = st.selectbox(
            "Model",
            ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # Processing settings
        st.subheader("Processing Settings")
        style = st.selectbox("Annotation Style", ["highlight", "underline"], index=0)
        
        # File upload
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX file to process"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Flagged Terms Configuration")
        
        # Load existing flagged terms
        if os.path.exists("flagged_terms_grant_enhanced.json"):
            with open("flagged_terms_grant_enhanced.json", 'r') as f:
                default_flagged_terms = json.load(f)
        else:
            default_flagged_terms = []
        
        # Flagged terms editor
        flagged_terms_text = st.text_area(
            "Flagged Terms (JSON array)",
            value=json.dumps(default_flagged_terms, indent=2),
            height=300,
            help="Enter terms to flag as a JSON array of strings"
        )
        
        try:
            flagged_terms = json.loads(flagged_terms_text)
            if not isinstance(flagged_terms, list):
                st.error("Flagged terms must be a JSON array")
                flagged_terms = []
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            flagged_terms = []
    
    with col2:
        st.header("🔄 Replacements Configuration")
        
        # Load existing replacements
        if os.path.exists("replacements_grant_enhanced.json"):
            with open("replacements_grant_enhanced.json", 'r') as f:
                default_replacements = json.load(f)
        else:
            default_replacements = {}
        
        # Replacements editor
        replacements_text = st.text_area(
            "Replacements (JSON object)",
            value=json.dumps(default_replacements, indent=2),
            height=300,
            help="Enter term replacements as a JSON object mapping 'term' -> 'suggestion'"
        )
        
        try:
            replacements = json.loads(replacements_text)
            if not isinstance(replacements, dict):
                st.error("Replacements must be a JSON object")
                replacements = {}
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            replacements = {}
    
    # Process button
    if uploaded_file is not None and flagged_terms and replacements:
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Create output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = f"web_output_{timestamp}"
                os.makedirs(outdir, exist_ok=True)
                
                # Process the document
                success, output, hits = run_processing(
                    tmp_file_path, flagged_terms, replacements, 
                    outdir, style, model, temperature
                )
                
                if success:
                    st.success("✅ Document processed successfully!")
                    
                    # Display results
                    st.header("📊 Results")
                    
                    if hits:
                        st.subheader(f"Found {len(hits)} flagged terms")
                        
                        # Display hits in a table
                        df = pd.DataFrame(hits)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download options
                        st.subheader("📥 Download Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # CSV download
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download CSV Report",
                                data=csv_data,
                                file_name=f"flag_report_{timestamp}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON download
                            json_data = df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="📋 Download JSON Report",
                                data=json_data,
                                file_name=f"flag_report_{timestamp}.json",
                                mime="application/json"
                            )
                        
                        with col3:
                            # Annotated PDF download
                            annotated_pdf_path = os.path.join(outdir, "flagged_output.pdf")
                            if os.path.exists(annotated_pdf_path):
                                with open(annotated_pdf_path, 'rb') as f:
                                    pdf_data = f.read()
                                st.download_button(
                                    label="📄 Download Annotated PDF",
                                    data=pdf_data,
                                    file_name=f"flagged_output_{timestamp}.pdf",
                                    mime="application/pdf"
                                )
                        
                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Flags", len(hits))
                        
                        with col2:
                            if hits:
                                unique_terms = len(set(hit.get('original_key', '') for hit in hits))
                                st.metric("Unique Terms", unique_terms)
                            else:
                                st.metric("Unique Terms", 0)
                        
                        with col3:
                            if hits:
                                pages = len(set(hit.get('page_num', 0) for hit in hits))
                                st.metric("Pages with Flags", pages)
                            else:
                                st.metric("Pages with Flags", 0)
                    
                    else:
                        st.info("No flagged terms found in the document.")
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    
                else:
                    st.error(f"❌ Processing failed: {output}")
                    os.unlink(tmp_file_path)
    
    elif uploaded_file is None:
        st.info("👆 Please upload a PDF or DOCX file to get started")
    
    elif not flagged_terms or not replacements:
        st.warning("⚠️ Please configure both flagged terms and replacements before processing")
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart PDF Language Flagger** - Upload documents and configure language flagging settings")

if __name__ == "__main__":
    main()
