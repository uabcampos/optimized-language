#!/usr/bin/env python3
"""
Hybrid Language Flagging Web App - Beta Version
Combines traditional pattern matching with LangExtract semantic analysis
"""

import streamlit as st
import os
import json
import tempfile
import subprocess
import sys
import re
import time
from typing import Dict, List, Tuple
import pandas as pd
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Optional PDF report generation
try:
    from pdf_report_generator import generate_pdf_report
    PDF_REPORT_AVAILABLE = True
except ImportError:
    PDF_REPORT_AVAILABLE = False
    print("PDF report generation not available")

# Set page config
st.set_page_config(
    page_title="Smart PDF Language Flagger - Hybrid Beta",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_config_files(preset: str) -> Tuple[str, str]:
    """Get flagged terms and replacements file names based on preset selection."""
    preset_mapping = {
        "Standard (flagged_terms.json)": ("flagged_terms.json", "replacements.json"),
        "Overkill (flagged_terms_overkill.json)": ("flagged_terms_overkill.json", "replacements_overkill.json"),
        "Overkill Lite (flagged_terms_overkill_lite.json)": ("flagged_terms_overkill_lite.json", "replacements_overkill_lite.json"),
        "Grant (flagged_terms_grant.json)": ("flagged_terms_grant.json", "replacements_grant.json"),
        "Grant Enhanced (flagged_terms_grant_enhanced.json)": ("flagged_terms_grant_enhanced.json", "replacements_grant_enhanced.json")
    }
    return preset_mapping.get(preset, ("flagged_terms.json", "replacements.json"))

def main():
    """Main application function."""
    st.title("🧠 Smart PDF Language Flagger - Hybrid Beta")
    st.markdown("**Enhanced with LangExtract semantic analysis**")
    
    # Configuration preset selection (outside sidebar for scope)
    config_preset = st.selectbox(
        "📋 Choose flagged terms configuration:",
        [
            "Standard (flagged_terms.json)",
            "Overkill (flagged_terms_overkill.json)", 
            "Overkill Lite (flagged_terms_overkill_lite.json)",
            "Grant (flagged_terms_grant.json)",
            "Grant Enhanced (flagged_terms_grant_enhanced.json)"
        ],
        help="Select which set of flagged terms and replacements to use"
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis mode selection
        st.subheader("🔧 Analysis Mode")
        analysis_mode = st.radio(
            "Choose analysis method:",
            ["Standard Pattern Matching", "Hybrid (Pattern + LangExtract)", "LangExtract Only"],
            help="Hybrid mode combines traditional pattern matching with LangExtract semantic analysis"
        )
        
        # Convert to flags
        use_hybrid = analysis_mode in ["Hybrid (Pattern + LangExtract)", "LangExtract Only"]
        use_langextract_only = analysis_mode == "LangExtract Only"
        
        # Show selected configuration
        flagged_file, replacements_file = get_config_files(config_preset)
        st.info(f"📋 Using: {flagged_file}")
        
        if use_hybrid:
            st.success("✅ Hybrid mode enabled")
            if use_langextract_only:
                st.info("🧠 LangExtract-only mode selected")
            else:
                st.info("🔧 Hybrid mode: Pattern matching + LangExtract")
        else:
            st.info("🔍 Standard pattern matching mode")
        
        # API configuration
        st.subheader("🔑 API Configuration")
        api_provider = st.selectbox(
            "API Provider:",
            ["OpenAI", "Gemini", "Auto"],
            help="Choose which API to use for language processing"
        )
        
        # Model selection based on API
        if api_provider == "OpenAI":
            model = st.selectbox("OpenAI Model:", ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"])
        elif api_provider == "Gemini":
            model = st.selectbox("Gemini Model:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
        else:  # Auto
            model = st.selectbox("Model (Auto-detect):", ["gpt-4.1-mini", "gemini-2.5-flash"])
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.2, 0.1)
            chunk_size = st.slider("Chunk Size:", 10, 50, 20, 5)
            num_processes = st.slider("Number of Processes:", 1, 8, 4, 1)
        
        # Skip terms
        st.subheader("🚫 Skip Terms")
        skip_terms_input = st.text_area(
            "Terms to skip (one per line):",
            value="determinant\ncohort\nmorbidity\nmortality",
            help="Enter terms that should be skipped during analysis"
        )
        skip_terms = [term.strip() for term in skip_terms_input.split('\n') if term.strip()]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx'],
            help="Upload a document to analyze for language improvements"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_file = tmp_file.name
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                process_document(input_file, analysis_mode, api_provider, model, temperature, skip_terms, config_preset)
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Display session state stats
        if 'processing_success' in st.session_state and st.session_state.processing_success:
            st.metric("Total Flags", len(st.session_state.processing_hits))
            st.metric("Pages Processed", len(set(hit.get('page_num', 0) for hit in st.session_state.processing_hits)))
            st.metric("Processing Time", f"{st.session_state.processing_duration:.1f}s" if st.session_state.processing_duration else "N/A")
            
            if st.session_state.processing_hits:
                # Analysis method breakdown
                methods = {}
                for hit in st.session_state.processing_hits:
                    method = hit.get('method', 'unknown')
                    methods[method] = methods.get(method, 0) + 1
                
                if methods:
                    st.subheader("🔍 Analysis Methods")
                    for method, count in methods.items():
                        st.metric(method.replace('_', ' ').title(), count)
        else:
            st.info("Upload and process a document to see statistics")

def process_document(input_file: str, analysis_mode: str, api_provider: str, model: str, 
                    temperature: float, skip_terms: List[str], config_preset: str):
    """Process the uploaded document."""
    
    # Initialize session state
    if 'processing_success' not in st.session_state:
        st.session_state.processing_success = False
    if 'processing_hits' not in st.session_state:
        st.session_state.processing_hits = []
    if 'processing_output' not in st.session_state:
        st.session_state.processing_output = ""
    if 'processing_outdir' not in st.session_state:
        st.session_state.processing_outdir = None
    if 'processing_timestamp' not in st.session_state:
        st.session_state.processing_timestamp = None
    if 'processing_duration' not in st.session_state:
        st.session_state.processing_duration = None
    
    # Get configuration files based on preset
    flagged_file, replacements_file = get_config_files(config_preset)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"hybrid_output_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    
    # Prepare command
    cmd = [
        sys.executable, "smart_flag_pdf.py",
        input_file,  # Positional argument, not --input
        "--flagged", flagged_file,
        "--map", replacements_file, 
        "--outdir", outdir,
        "--model", model,
        "--temperature", str(temperature),
        "--api", api_provider.lower(),
        "--skip-terms"] + skip_terms
    
    # Add hybrid flag if needed
    if analysis_mode in ["Hybrid (Pattern + LangExtract)", "LangExtract Only"]:
        cmd.append("--hybrid")
    
    # Show command for debugging
    with st.expander("🔧 Debug: Command", expanded=False):
        st.code(" ".join(cmd))
        st.info(f"Configuration: {flagged_file} + {replacements_file}")
    
    # Process with progress bar and detailed progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_container = st.empty()
    
    # Progress tracking variables
    total_pages = 0
    current_page = 0
    processing_rate = 0
    start_time = time.time()
    last_update_time = start_time
    
    try:
        status_text.text("🚀 Starting document analysis...")
        progress_bar.progress(10)
        
        # Run the command with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        total_hits = 0
        methods = {}
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.strip())
                
                # Parse progress information
                if "Processing" in line and "pages with" in line:
                    # Extract total pages: "Processing 245 pages with Overkill preset..."
                    match = re.search(r'Processing (\d+) pages', line)
                    if match:
                        total_pages = int(match.group(1))
                
                elif "Processing page" in line and "/" in line:
                    # Extract current page and rate: "Processing page 45/245 (18.4%) - Rate: 2.3 pages/sec - ETA: 12m 45s"
                    page_match = re.search(r'Processing page (\d+)/(\d+)', line)
                    rate_match = re.search(r'Rate: ([\d.]+) pages/sec', line)
                    
                    if page_match:
                        current_page = int(page_match.group(1))
                        if not total_pages:
                            total_pages = int(page_match.group(2))
                    
                    if rate_match:
                        processing_rate = float(rate_match.group(1))
                    
                    # Update UI every 10 seconds
                    current_time = time.time()
                    if (current_time - last_update_time) >= 10:
                        elapsed = current_time - start_time
                        progress_pct = (current_page / total_pages * 100) if total_pages > 0 else 0
                        eta_seconds = (total_pages - current_page) / processing_rate if processing_rate > 0 else 0
                        eta_minutes = eta_seconds / 60
                        
                        with progress_container.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Pages Processed", f"{current_page}/{total_pages}")
                            with col2:
                                st.metric("Progress", f"{progress_pct:.1f}%")
                            with col3:
                                st.metric("Rate", f"{processing_rate:.1f} pages/sec")
                            with col4:
                                st.metric("ETA", f"{eta_minutes:.1f} min")
                        
                        progress_bar.progress(progress_pct / 100)
                        last_update_time = current_time
                
                # Parse final statistics
                elif "Total flags:" in line:
                    total_hits = int(line.split(":")[1].strip())
                elif "Pattern matching only:" in line:
                    methods["pattern_matching"] = int(line.split(":")[1].strip())
                elif "LangExtract only:" in line:
                    methods["langextract"] = int(line.split(":")[1].strip())
                elif "Found by both:" in line:
                    methods["both_methods"] = int(line.split(":")[1].strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            progress_bar.progress(80)
            status_text.text("📊 Processing results...")
            
            # Load results
            csv_file = os.path.join(outdir, "flag_report.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                hits = df.to_dict('records')
                
                # Update session state
                st.session_state.processing_success = True
                st.session_state.processing_hits = hits
                st.session_state.processing_output = '\n'.join(output_lines)
                st.session_state.processing_outdir = outdir
                st.session_state.processing_timestamp = timestamp
                st.session_state.processing_duration = time.time() - start_time
                st.session_state.total_hits = total_hits
                st.session_state.methods = methods
                
                progress_bar.progress(100)
                status_text.text(f"✅ Found {len(hits)} language issues!")
                
                # Force page refresh
                st.rerun()
            else:
                st.error("❌ Results file not found")
        else:
            stderr_output = process.stderr.read()
            st.error(f"❌ Processing failed: {stderr_output}")
            status_text.text("❌ Analysis failed")
            
    except subprocess.TimeoutExpired:
        st.error("⏰ Processing timed out (5 minutes)")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()
    
    # Clean up input file
    if os.path.exists(input_file):
        os.unlink(input_file)

def display_results():
    """Display processing results."""
    if not st.session_state.get('processing_success', False):
        return
    
    st.header("📊 Analysis Results")
    
    hits = st.session_state.processing_hits
    if not hits:
        st.warning("No language issues found!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flags", len(hits))
    
    with col2:
        pages = len(set(hit.get('page_num', 0) for hit in hits))
        st.metric("Pages Affected", pages)
    
    with col3:
        unique_terms = len(set(hit.get('original_key', '') for hit in hits))
        st.metric("Unique Terms", unique_terms)
    
    with col4:
        duration = st.session_state.get('processing_duration', 0)
        st.metric("Processing Time", f"{duration:.1f}s")
    
    # Analysis method breakdown
    if hits:
        methods = {}
        for hit in hits:
            method = hit.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        if methods:
            st.subheader("🔍 Analysis Method Breakdown")
            method_df = pd.DataFrame(list(methods.items()), columns=['Method', 'Count'])
            fig = px.pie(method_df, values='Count', names='Method', title="Flags by Analysis Method")
            st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    if hits:
        df = pd.DataFrame(hits)
        
        # Add search functionality
        search_term = st.text_input(
            "🔍 Search results:", 
            placeholder="Search by term, suggestion, or page...",
            key="search_results"
        )
        
        # Filter results
        if search_term and search_term.strip():
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            df = df[mask]
            st.info(f"Showing {len(df)} results matching '{search_term}'")
        
        # Display table
        st.dataframe(
            df,
            column_config={
                "page_num": st.column_config.NumberColumn("Page", help="Page number"),
                "original_key": st.column_config.TextColumn("Original Term", help="Flagged term"),
                "matched_text": st.column_config.TextColumn("Matched Text", help="Actual text found"),
                "suggestion": st.column_config.TextColumn("Suggestion", help="Recommended replacement"),
                "reason": st.column_config.TextColumn("Reason", help="Why this was flagged"),
                "method": st.column_config.TextColumn("Method", help="How it was found")
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Download options
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"hybrid_analysis_{st.session_state.processing_timestamp}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Download Annotated PDF"):
            if st.session_state.processing_outdir:
                pdf_path = os.path.join(st.session_state.processing_outdir, "flagged_output.pdf")
                if os.path.exists(pdf_path):
                    with open(pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"hybrid_flagged_{st.session_state.processing_timestamp}.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
    
    # Display results if available
    if st.session_state.get('processing_success', False):
        display_results()
    
    # Debug information
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("**Session State:**")
        st.write(f"- Processing Success: {st.session_state.get('processing_success', False)}")
        st.write(f"- Hits Count: {len(st.session_state.get('processing_hits', []))}")
        st.write(f"- Output Directory: {st.session_state.get('processing_outdir', 'None')}")
        
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()
