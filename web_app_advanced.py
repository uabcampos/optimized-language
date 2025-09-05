#!/usr/bin/env python3
"""
Advanced Web interface for the Smart PDF Language Flagger
"""

import streamlit as st
import json
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import time
import re
from typing import Dict, List, Tuple
import pandas as pd
import zipfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Smart PDF Language Flagger - Advanced",
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
                  outdir: str, style: str, model: str, temperature: float, api_type: str = "auto", progress_container=None) -> Tuple[bool, str, List[dict]]:
    """Run the processing script and return results with real-time progress updates."""
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
            "--temperature", str(temperature),
            "--api", api_type
        ]
        
        # Use Popen for real-time output capture
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            cwd=os.getcwd(),
            bufsize=1,
            universal_newlines=True
        )
        
        # Parse output in real-time and update UI
        output_lines = []
        current_page = 0
        total_pages = 0
        processing_rate = 0
        start_time = time.time()
        last_update_time = start_time
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                
                # Parse progress information
                if "Processing" in output and "pages with" in output:
                    # Extract total pages: "Processing 245 pages with Overkill preset..."
                    match = re.search(r'Processing (\d+) pages', output)
                    if match:
                        total_pages = int(match.group(1))
                
                elif "Processing page" in output and "/" in output:
                    # Extract current page and rate: "Processing page 45/245 (18.4%) - Rate: 2.3 pages/sec - ETA: 12m 45s"
                    page_match = re.search(r'Processing page (\d+)/(\d+)', output)
                    rate_match = re.search(r'Rate: ([\d.]+) pages/sec', output)
                    
                    if page_match:
                        current_page = int(page_match.group(1))
                        if not total_pages:
                            total_pages = int(page_match.group(2))
                    
                    if rate_match:
                        processing_rate = float(rate_match.group(1))
                    
                    # Update UI every 10 seconds
                    current_time = time.time()
                    if progress_container and (current_time - last_update_time) >= 10:
                        elapsed = current_time - start_time
                        progress_pct = (current_page / total_pages * 100) if total_pages > 0 else 0
                        eta_seconds = (total_pages - current_page) / processing_rate if processing_rate > 0 else 0
                        eta_minutes = eta_seconds / 60
                        elapsed_minutes = elapsed / 60
                        
                        # Determine confidence level
                        confidence = "Low"
                        if current_page > 10:
                            confidence = "Medium"
                        if current_page > 50:
                            confidence = "High"
                        
                        with progress_container.container():
                            st.progress(progress_pct / 100)
                            st.text(f"📄 Page {current_page}/{total_pages} ({progress_pct:.1f}%)")
                            st.text(f"📊 Rate: {processing_rate:.1f} pages/sec")
                            st.text(f"⏱️ Elapsed: {elapsed_minutes:.1f}m | ETA: {eta_minutes:.1f}m")
                            st.text(f"🎯 Confidence: {confidence} (based on {current_page} pages)")
                            st.text(f"📝 Output: {len(output_lines)} lines processed")
                        
                        last_update_time = current_time
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Clean up temporary files
        os.unlink(flagged_file)
        os.unlink(replacements_file)
        
        full_output = "\n".join(output_lines)
        
        if return_code == 0:
            # Load results
            csv_path = os.path.join(outdir, "flag_report.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                hits = df.to_dict('records')
            else:
                hits = []
            
            return True, full_output, hits
        else:
            return False, full_output, []
            
    except Exception as e:
        return False, str(e), []

def create_visualizations(hits: List[dict]) -> None:
    """Create visualizations for the results."""
    if not hits:
        return
    
    df = pd.DataFrame(hits)
    
    # Page distribution
    st.subheader("📊 Page Distribution")
    page_counts = df['page_num'].value_counts().sort_index()
    fig_pages = px.bar(
        x=page_counts.index, 
        y=page_counts.values,
        title="Flags per Page",
        labels={'x': 'Page Number', 'y': 'Number of Flags'}
    )
    st.plotly_chart(fig_pages, use_container_width=True)
    
    # Top flagged terms
    st.subheader("🏷️ Most Flagged Terms")
    term_counts = df['original_key'].value_counts().head(10)
    fig_terms = px.bar(
        x=term_counts.values,
        y=term_counts.index,
        orientation='h',
        title="Top 10 Flagged Terms",
        labels={'x': 'Number of Occurrences', 'y': 'Term'}
    )
    fig_terms.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_terms, use_container_width=True)
    
    # Suggestion categories
    st.subheader("💡 Suggestion Analysis")
    if 'suggestion' in df.columns:
        # Word cloud of suggestions (simplified)
        suggestions = df['suggestion'].str.lower().str.split().explode()
        suggestion_counts = suggestions.value_counts().head(20)
        fig_suggestions = px.bar(
            x=suggestion_counts.values,
            y=suggestion_counts.index,
            orientation='h',
            title="Most Common Words in Suggestions",
            labels={'x': 'Frequency', 'y': 'Word'}
        )
        fig_suggestions.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_suggestions, use_container_width=True)

def main():
    st.title("📝 Smart PDF Language Flagger - Advanced")
    st.markdown("Upload documents and configure language flagging settings with advanced analytics")
    
    # Sidebar for configuration only
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Setup
        st.subheader("🔑 API Configuration")
        
        # Set up API keys from Streamlit secrets
        try:
            if "GEMINI_API_KEY" in st.secrets:
                os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
                st.success("✅ Gemini API key loaded from secrets")
            else:
                st.warning("⚠️ Gemini API key not found in secrets")
                
            if "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                st.success("✅ OpenAI API key loaded from secrets")
            else:
                st.warning("⚠️ OpenAI API key not found in secrets")
        except Exception as e:
            st.error(f"Error loading API keys: {e}")
        
        # API Selection
        api_type = st.selectbox(
            "API Provider",
            ["auto", "gemini", "openai"],
            index=0,
            help="Auto: Prefers Gemini if available, otherwise OpenAI"
        )
        
        # Model settings
        st.subheader("LLM Settings")
        model = st.selectbox(
            "Model",
            ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="OpenAI model (ignored if using Gemini)"
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # Processing settings
        st.subheader("Processing Settings")
        style = st.selectbox("Annotation Style", ["highlight", "underline"], index=0)
        
        # Configuration presets
        st.subheader("🎛️ Configuration Presets")
        preset = st.selectbox(
            "Load Preset",
            ["Custom", "Grant (Enhanced)", "RPPR", "General", "Overkill"],
            index=1
        )
        
        if preset != "Custom":
            if preset == "Grant (Enhanced)":
                flagged_file = "flagged_terms_grant_enhanced.json"
                replacements_file = "replacements_grant_enhanced.json"
            elif preset == "RPPR":
                flagged_file = "flagged_terms_grant.json"
                replacements_file = "replacements_grant.json"
            elif preset == "General":
                flagged_file = "flagged_terms_general.json"
                replacements_file = "replacements_general.json"
            elif preset == "Overkill":
                flagged_file = "flagged_terms_overkill.json"
                replacements_file = "replacements_overkill.json"
            else:  # Custom
                flagged_file = "flagged_terms.json"
                replacements_file = "replacements.json"
            
            if os.path.exists(flagged_file) and os.path.exists(replacements_file):
                st.success(f"✅ Loaded {preset} configuration")
            else:
                st.warning(f"⚠️ {preset} configuration files not found")
    
    # Initialize variables
    flagged_terms = []
    replacements = {}
    
    # Main content area
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file to process",
        type=['pdf', 'docx'],
        help="Upload a PDF or DOCX file to process",
        accept_multiple_files=False
    )
    
    # Only show configuration and processing if file is uploaded
    if uploaded_file is not None:
        try:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Debug information
            st.write(f"File type: {uploaded_file.type}")
            st.write(f"File size: {uploaded_file.size} bytes")
            
            # Validate file type
            if uploaded_file.type not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                st.error("❌ Please upload a valid PDF or DOCX file")
                st.stop()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.header("📋 Flagged Terms Configuration")
                
                # Load flagged terms based on preset
                if preset != "Custom" and 'flagged_file' in locals() and os.path.exists(flagged_file):
                    with open(flagged_file, 'r') as f:
                        default_flagged_terms = json.load(f)
                else:
                    default_flagged_terms = []
                
                # Flagged terms as simple text area
                st.subheader("Terms to Flag")
                st.markdown("Enter terms to flag (one per line):")
                
                # Simple text area for flagged terms
                flagged_terms_text = st.text_area(
                    "Flagged Terms",
                    value="\n".join(default_flagged_terms),
                    height=200,
                    help="Enter terms to flag, one per line"
                )
                
                # Parse the terms
                flagged_terms = [term.strip() for term in flagged_terms_text.split('\n') if term.strip()]
                
                # Show count
                st.info(f"📊 Total flagged terms: {len(flagged_terms)}")
                
                # Show first few terms
                if flagged_terms:
                    st.write("First 10 terms:")
                    for i, term in enumerate(flagged_terms[:10]):
                        st.write(f"{i+1}. {term}")
                    if len(flagged_terms) > 10:
                        st.write(f"... and {len(flagged_terms) - 10} more")
            
            with col2:
                st.header("🔄 Replacements Configuration")
                
                # Load replacements based on preset
                if preset != "Custom" and 'replacements_file' in locals() and os.path.exists(replacements_file):
                    with open(replacements_file, 'r') as f:
                        default_replacements = json.load(f)
                else:
                    default_replacements = {}
                
                # Replacements as simple text area
                st.subheader("Term Replacements")
                st.markdown("Enter replacements as JSON (term: replacement):")
                
                # Simple text area for replacements
                replacements_text = st.text_area(
                    "Replacements JSON",
                    value=json.dumps(default_replacements, indent=2),
                    height=200,
                    help="Enter replacements as JSON object"
                )
                
                # Parse the replacements
                try:
                    replacements = json.loads(replacements_text)
                    if not isinstance(replacements, dict):
                        st.error("❌ Replacements must be a JSON object")
                        replacements = default_replacements
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {e}")
                    replacements = default_replacements
                
                # Show count
                st.info(f"📊 Total replacements: {len(replacements)}")
                
                # Show first few replacements
                if replacements:
                    st.write("First 5 replacements:")
                    for i, (term, replacement) in enumerate(list(replacements.items())[:5]):
                        st.write(f"{i+1}. {term} → {replacement}")
                    if len(replacements) > 5:
                        st.write(f"... and {len(replacements) - 5} more")
            
            # Process button
            if flagged_terms and replacements:
                if st.button("🚀 Process Document", type="primary", use_container_width=True):
                    # Create progress containers
                    progress_container = st.container()
                    status_container = st.container()
                    
                    with progress_container:
                        st.markdown("### 📊 Processing Progress")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        details_text = st.empty()
                    
                    try:
                        # Step 1: Save uploaded file
                        status_text.text("📁 Saving uploaded file...")
                        details_text.text(f"Preparing {uploaded_file.name} for processing")
                        progress_bar.progress(10)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Step 2: Create output directory
                        status_text.text("📂 Creating output directory...")
                        details_text.text("Setting up workspace for results")
                        progress_bar.progress(20)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir = f"web_output_{timestamp}"
                        os.makedirs(outdir, exist_ok=True)
                        
                        # Step 3: Initialize processing
                        status_text.text("🔧 Initializing language processing...")
                        details_text.text(f"Loading {len(flagged_terms)} flagged terms and {len(replacements)} replacements")
                        progress_bar.progress(30)
                        
                        # Step 4: Run processing with detailed progress
                        status_text.text("🤖 Running AI language analysis...")
                        details_text.text("Analyzing document for flagged terms and generating suggestions")
                        progress_bar.progress(50)
                        
                        # Create progress container for real-time updates
                        progress_container = st.empty()
                        
                        # Show initial processing message
                        with progress_container.container():
                            st.info("🔄 Starting parallel processing... Calculating time estimates...")
                            st.progress(0)
                            st.text("📄 Initializing...")
                            st.text("📊 Rate: Calculating...")
                            st.text("⏱️ Elapsed: 0m | ETA: Calculating...")
                            st.text("🎯 Confidence: Low (initializing)")
                        
                        success, output, hits = run_processing(
                            tmp_file_path, flagged_terms, replacements, 
                            outdir, style, model, temperature, api_type, progress_container
                        )
                        
                        # Step 5: Processing complete
                        status_text.text("✅ Processing complete!")
                        details_text.text(f"Found {len(hits) if hits else 0} flagged terms")
                        progress_bar.progress(100)
                        
                        # Display processing output
                        if output:
                            st.subheader("📋 Processing Log")
                            st.text_area("Processing Output", value=output, height=300, help="Detailed processing log from the language flagging script")
                        
                        if success:
                            # Clear progress indicators
                            progress_container.empty()
                            status_container.empty()
                            
                            st.success("✅ Document processed successfully!")
                            
                            # Show processing summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Flagged Terms Found", len(hits) if hits else 0)
                            with col2:
                                st.metric("Processing Time", f"{datetime.now().strftime('%H:%M:%S')}")
                            with col3:
                                st.metric("Output Files", "1 PDF + Reports")
                            
                            # Display results with progress
                            st.header("📊 Results")
                            
                            if hits:
                                # Results processing progress
                                with st.spinner("Generating detailed analysis..."):
                                    st.subheader(f"Found {len(hits)} flagged terms")
                                    
                                    # Create progress for results processing
                                    results_progress = st.progress(0)
                                    results_status = st.empty()
                                    
                                    # Step 1: Process hits data
                                    results_status.text("📋 Processing flagged terms data...")
                                    results_progress.progress(25)
                                    
                                    # Step 2: Generate visualizations
                                    results_status.text("📈 Creating visualizations...")
                                    results_progress.progress(50)
                                    
                                    # Step 3: Prepare download files
                                    results_status.text("📥 Preparing download files...")
                                    results_progress.progress(75)
                                    
                                    # Step 4: Complete
                                    results_status.text("✅ Analysis complete!")
                                    results_progress.progress(100)
                                    
                                    # Clear progress indicators
                                    results_progress.empty()
                                    results_status.empty()
                                
                                # Create visualizations
                                create_visualizations(hits)
                                
                                # Display hits in a table
                                st.subheader("📋 Detailed Results")
                                df = pd.DataFrame(hits)
                                st.dataframe(df, use_container_width=True)
                                
                                # Download options
                                st.subheader("📥 Download Results")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
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
                                
                                with col4:
                                    # ZIP download with all files
                                    zip_path = f"results_{timestamp}.zip"
                                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                                        for root, dirs, files in os.walk(outdir):
                                            for file in files:
                                                file_path = os.path.join(root, file)
                                                zipf.write(file_path, os.path.relpath(file_path, outdir))
                                    
                                    with open(zip_path, 'rb') as f:
                                        zip_data = f.read()
                                    st.download_button(
                                        label="📦 Download All Files (ZIP)",
                                        data=zip_data,
                                        file_name=zip_path,
                                        mime="application/zip"
                                    )
                                
                                # Summary statistics
                                st.subheader("📈 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                
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
                                
                                with col4:
                                    if hits:
                                        avg_flags_per_page = len(hits) / len(set(hit.get('page_num', 0) for hit in hits))
                                        st.metric("Avg Flags per Page", f"{avg_flags_per_page:.1f}")
                                    else:
                                        st.metric("Avg Flags per Page", 0)
                            
                            else:
                                st.info("No flagged terms found in the document.")
                            
                            # Clean up
                            os.unlink(tmp_file_path)
                        
                        else:
                            st.error(f"❌ Processing failed: {output}")
                            os.unlink(tmp_file_path)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
                        st.error("Please try uploading a different file or check the file format.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Please upload a PDF or DOCX file to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart PDF Language Flagger - Advanced** - Upload documents and configure language flagging settings with advanced analytics")

if __name__ == "__main__":
    main()