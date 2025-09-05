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
from pdf_report_generator import generate_pdf_report

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
                  outdir: str, style: str, model: str, temperature: float, api_type: str = "auto", 
                  skip_terms: List[str] = None, progress_container=None) -> Tuple[bool, str, List[dict]]:
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
        
        # Add skip terms if provided
        if skip_terms:
            cmd.extend(["--skip-terms"] + skip_terms)
            print(f"DEBUG: Adding skip terms to command: {skip_terms}")
        else:
            print("DEBUG: No skip terms provided")
        
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
                print(f"DEBUG: Loaded {len(hits)} hits from CSV")
                if hits:
                    print(f"DEBUG: Sample hit keys: {list(hits[0].keys())}")
            else:
                hits = []
                print("DEBUG: CSV file not found")
            
            
            return True, full_output, hits
        else:
            return False, full_output, []
            
    except Exception as e:
        return False, str(e), []

def load_document_analysis(outdir: str) -> dict:
    """Load document analysis results from JSON file."""
    try:
        analysis_file = os.path.join(outdir, "document_analysis.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading document analysis: {e}")
    return None

def display_document_analysis(analysis_data: dict) -> None:
    """Display document analysis results in the UI."""
    if not analysis_data:
        return
    
    st.subheader("📊 Document Analysis")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document Type", analysis_data.get('document_type', 'Unknown'))
    with col2:
        st.metric("NIH Alignment", f"{analysis_data.get('nih_alignment_score', 0):.1f}%")
    with col3:
        st.metric("Project 2025", f"{analysis_data.get('project_2025_score', 0):.1f}%")
    with col4:
        st.metric("Alabama SB 129", f"{analysis_data.get('alabama_sb129_score', 0):.1f}%")
    
    # Main themes and focus
    col1, col2 = st.columns(2)
    with col1:
        st.write("**🎯 Key Themes:**")
        themes = analysis_data.get('main_themes', [])
        for theme in themes[:5]:  # Show top 5 themes
            st.write(f"• {theme}")
    
    with col2:
        st.write("**🔬 Scientific Focus:**")
        st.write(analysis_data.get('scientific_focus', 'Not specified'))
    
    # NIH Priorities
    nih_priorities = analysis_data.get('nih_priority_alignment', [])
    if nih_priorities:
        st.write("**🏥 NIH Priority Alignment:**")
        for priority in nih_priorities[:3]:  # Show top 3 priorities
            st.write(f"• {priority}")
    
    # Strategic recommendations
    recommendations = analysis_data.get('strategic_recommendations', [])
    if recommendations:
        st.write("**💡 Language & Communication Recommendations:**")
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3 recommendations
            st.write(f"{i}. {rec}")
    
    # Policy relevance
    policy_relevance = analysis_data.get('policy_relevance', '')
    if policy_relevance:
        st.write("**🏛️ Policy Relevance:**")
        st.write(policy_relevance)

def create_visualizations(hits: List[dict]) -> None:
    """Create comprehensive visualizations for the results."""
    if not hits:
        st.info("ℹ️ No data available for visualization.")
        return
    
    df = pd.DataFrame(hits)
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏷️ Terms Analysis", "📄 Page Analysis", "💡 Suggestions", "📈 Trends"])
    
    with tab1:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Flags", len(df))
        with col2:
            st.metric("Unique Terms", df['original_key'].nunique())
        with col3:
            st.metric("Pages with Flags", df['page_num'].nunique())
        with col4:
            st.metric("Avg Flags per Page", f"{len(df) / df['page_num'].nunique():.1f}")
        
        # Page distribution
        st.subheader("📊 Flags per Page")
        page_counts = df['page_num'].value_counts().sort_index()
        fig_pages = px.bar(
            x=page_counts.index, 
            y=page_counts.values,
            title="Distribution of Flags Across Pages",
            labels={'x': 'Page Number', 'y': 'Number of Flags'},
            color=page_counts.values,
            color_continuous_scale='Blues'
        )
        fig_pages.update_layout(showlegend=False)
        st.plotly_chart(fig_pages, width='stretch')
    
    with tab2:
        # Top flagged terms
        st.subheader("🏷️ Most Flagged Terms")
        term_counts = df['original_key'].value_counts().head(15)
        fig_terms = px.bar(
            x=term_counts.values,
            y=term_counts.index,
            orientation='h',
            title="Top 15 Flagged Terms",
            labels={'x': 'Number of Occurrences', 'y': 'Term'},
            color=term_counts.values,
            color_continuous_scale='Reds'
        )
        fig_terms.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_terms, width='stretch')
        
        # Term frequency pie chart
        if len(term_counts) > 1:
            st.subheader("🥧 Term Distribution")
            fig_pie = px.pie(
                values=term_counts.values,
                names=term_counts.index,
                title="Distribution of Flagged Terms"
            )
            st.plotly_chart(fig_pie, width='stretch')
    
    with tab3:
        # Page analysis
        st.subheader("📄 Page-by-Page Analysis")
        
        # Create a detailed page analysis
        page_analysis = df.groupby('page_num').agg({
            'original_key': 'count',
            'matched_text': lambda x: ', '.join(x.unique()[:5])  # First 5 unique matches
        }).rename(columns={'original_key': 'flag_count', 'matched_text': 'sample_terms'})
        
        st.dataframe(page_analysis, width='stretch')
        
        # Page heatmap
        if len(page_counts) > 1:
            st.subheader("🔥 Page Activity Heatmap")
            # Create a simple heatmap representation
            max_page = df['page_num'].max()
            heatmap_data = []
            for page in range(1, max_page + 1):
                count = page_counts.get(page, 0)
                heatmap_data.append({'Page': page, 'Flags': count})
            
            heatmap_df = pd.DataFrame(heatmap_data)
            fig_heatmap = px.bar(
                heatmap_df, 
                x='Page', 
                y='Flags',
                title="Page Activity Heatmap",
                color='Flags',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_heatmap, width='stretch')
    
    with tab4:
        # Suggestion analysis
        st.subheader("💡 Suggestion Analysis")
        if 'suggestion' in df.columns and not df['suggestion'].isna().all():
            # Most common words in suggestions
            suggestions = df['suggestion'].str.lower().str.split().explode()
            suggestion_counts = suggestions.value_counts().head(20)
            
            if len(suggestion_counts) > 0:
                fig_suggestions = px.bar(
                    x=suggestion_counts.values,
                    y=suggestion_counts.index,
                    orientation='h',
                    title="Most Common Words in Suggestions",
                    labels={'x': 'Frequency', 'y': 'Word'},
                    color=suggestion_counts.values,
                    color_continuous_scale='Greens'
                )
                fig_suggestions.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_suggestions, width='stretch')
            
            # Suggestion length analysis
            st.subheader("📏 Suggestion Length Analysis")
            df['suggestion_length'] = df['suggestion'].str.len()
            fig_length = px.histogram(
                df, 
                x='suggestion_length',
                title="Distribution of Suggestion Lengths",
                labels={'x': 'Character Count', 'y': 'Frequency'},
                nbins=20
            )
            st.plotly_chart(fig_length, width='stretch')
        else:
            st.info("No suggestion data available for analysis.")
    
    with tab5:
        # Trends analysis
        st.subheader("📈 Processing Trends")
        
        # Flags over time (by page order)
        fig_trend = px.line(
            x=page_counts.index,
            y=page_counts.values,
            title="Flags Trend Across Pages",
            labels={'x': 'Page Number', 'y': 'Number of Flags'},
            markers=True
        )
        st.plotly_chart(fig_trend, width='stretch')
        
        # Cumulative flags
        cumulative_flags = page_counts.cumsum()
        fig_cumulative = px.line(
            x=cumulative_flags.index,
            y=cumulative_flags.values,
            title="Cumulative Flags Across Pages",
            labels={'x': 'Page Number', 'y': 'Cumulative Flags'},
            markers=True
        )
        st.plotly_chart(fig_cumulative, width='stretch')

def main():
    st.title("📝 Smart PDF Language Flagger - Advanced")
    st.markdown("Upload documents and configure language flagging settings with advanced analytics")
    
    # Initialize session state for results persistence with better error handling
    try:
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = None
        if 'processing_success' not in st.session_state:
            st.session_state.processing_success = False
        if 'processing_hits' not in st.session_state:
            st.session_state.processing_hits = []
        if 'processing_output' not in st.session_state:
            st.session_state.processing_output = ""
        if 'processing_timestamp' not in st.session_state:
            st.session_state.processing_timestamp = None
        if 'processing_outdir' not in st.session_state:
            st.session_state.processing_outdir = None
        if 'processing_duration' not in st.session_state:
            st.session_state.processing_duration = None
        
        # Ensure processing_hits is always a list to prevent serialization issues
        if not isinstance(st.session_state.processing_hits, list):
            st.session_state.processing_hits = []
            
    except Exception as e:
        st.error(f"Session state initialization error: {e}")
        # Reset session state
        st.session_state.clear()
        st.rerun()
    
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
        
        if api_type == "gemini":
            # Gemini model options
            model = st.selectbox(
                "Gemini Model",
                ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                index=0,
                help="Google Gemini model for language processing"
            )
        elif api_type == "openai":
            # OpenAI model options
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
                help="OpenAI model for language processing"
            )
        else:  # auto mode
            # Show both options with explanation
            model_type = st.radio(
                "Model Type",
                ["Gemini (Recommended)", "OpenAI"],
                index=0,
                help="Auto mode will prefer Gemini if available"
            )
            
            if model_type == "Gemini (Recommended)":
                model = st.selectbox(
                    "Gemini Model",
                    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                    index=0,
                    help="Google Gemini model (faster and cheaper)"
                )
            else:
                model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    index=0,
                    help="OpenAI model"
                )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # Processing settings
        st.subheader("Processing Settings")
        style = st.selectbox("Annotation Style", ["highlight", "underline"], index=0)
        
        # Skip terms configuration
        st.subheader("🚫 Skip Terms")
        skip_terms_text = st.text_area(
            "Terms to Skip (one per line)",
            value="",
            height=100,
            help="Enter terms to skip during processing. The system will automatically detect variations (e.g., 'disparity' will skip 'disparities', 'disparate', etc.)"
        )
        
        # Parse skip terms
        skip_terms = [term.strip().lower() for term in skip_terms_text.split('\n') if term.strip()]
        
        if skip_terms:
            st.info(f"📝 Will skip {len(skip_terms)} term(s) and their variations")
            with st.expander("🔍 Skip Terms Details", expanded=False):
                st.write("**Skip Terms:**")
                for i, term in enumerate(skip_terms, 1):
                    st.write(f"{i}. {term}")
                st.write("**Note:** The system will automatically detect variations (e.g., 'disparity' will skip 'disparities', 'disparate', etc.)")
        
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
    
    # Display persistent results if available
    if st.session_state.processing_success:
        st.markdown("---")
        st.header("📊 Processing Results")
        
        # Show processing summary with enhanced metrics
        st.subheader("📊 Processing Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Flagged Terms Found", len(st.session_state.processing_hits))
        with col2:
            processing_time = st.session_state.get('processing_duration', 'Unknown')
            st.metric("Processing Time", processing_time)
        with col3:
            st.metric("Output Files", "1 PDF + Reports")
        with col4:
            if st.session_state.processing_hits:
                unique_terms = len(set(hit.get('original_key', '') for hit in st.session_state.processing_hits))
                st.metric("Unique Terms", unique_terms)
            else:
                st.metric("Unique Terms", 0)
        
        # Success indicator
        if st.session_state.processing_hits:
            st.success(f"✅ Successfully processed document and found {len(st.session_state.processing_hits)} flagged terms!")
        else:
            st.warning("⚠️ No flagged terms found in the document.")
        
        # Load and display document analysis
        if st.session_state.processing_outdir:
            analysis_data = load_document_analysis(st.session_state.processing_outdir)
            if analysis_data:
                display_document_analysis(analysis_data)
        
        # Try to load results directly from file if session state is empty
        if not st.session_state.processing_hits and st.session_state.processing_outdir:
            csv_path = os.path.join(st.session_state.processing_outdir, "flag_report.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    hits = df.to_dict('records')
                    st.session_state.processing_hits = hits
                    st.success(f"✅ Loaded {len(hits)} hits directly from CSV file")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        if st.session_state.processing_hits and len(st.session_state.processing_hits) > 0:
            # Create visualizations (only if there are hits)
            st.subheader("📈 Analytics Dashboard")
            try:
                create_visualizations(st.session_state.processing_hits)
            except Exception as e:
                st.error(f"Error creating visualizations: {e}")
            
            # Display hits in a detailed table
            st.subheader("📋 Detailed Results Table")
            try:
                # Ensure we have valid data
                if not st.session_state.processing_hits:
                    st.warning("No hits data available")
                    return
                
                # Create DataFrame with error handling
                try:
                    df = pd.DataFrame(st.session_state.processing_hits)
                    if df.empty:
                        st.warning("No data to display")
                        return
                except Exception as df_error:
                    st.error(f"Error creating DataFrame: {df_error}")
                    return
                
                st.write(f"DataFrame columns: {list(df.columns)}")
                
                # Add search and filter capabilities with better error handling
                search_term = st.text_input(
                    "🔍 Search in results:", 
                    placeholder="Search by term, suggestion, or page...", 
                    key="search_results",
                    help="Search across all columns in the results table"
                )
                
                # Initialize display dataframe
                display_df = df.copy()
                
                # Apply search filter if provided
                if search_term and search_term.strip():
                    try:
                        # Clean the search term
                        clean_search = search_term.strip()
                        
                        # Create search mask with better error handling
                        search_mask = pd.Series([False] * len(display_df), index=display_df.index)
                        
                        for col in display_df.columns:
                            try:
                                col_mask = display_df[col].astype(str).str.contains(
                                    clean_search, 
                                    case=False, 
                                    na=False, 
                                    regex=False
                                )
                                search_mask = search_mask | col_mask
                            except Exception as col_error:
                                st.warning(f"Search error in column '{col}': {col_error}")
                                continue
                        
                        display_df = display_df[search_mask]
                        st.info(f"Showing {len(display_df)} results matching '{clean_search}'")
                        
                    except Exception as search_error:
                        st.warning(f"Search error: {search_error}. Showing all results.")
                        display_df = df
                
                # Display the dataframe with enhanced formatting
                try:
                    # Ensure we have data to display
                    if display_df.empty:
                        st.info("No results match your search criteria.")
                    else:
                        st.dataframe(
                            display_df, 
                            width='stretch',
                            column_config={
                                "page_num": st.column_config.NumberColumn("Page", help="Page number where the term was found"),
                                "original_key": st.column_config.TextColumn("Original Term", help="The flagged term"),
                                "matched_text": st.column_config.TextColumn("Matched Text", help="Actual text that was matched"),
                                "suggestion": st.column_config.TextColumn("Suggestion", help="LLM-generated suggestion"),
                                "reason": st.column_config.TextColumn("Reason", help="Explanation for the suggestion"),
                                "context": st.column_config.TextColumn("Context", help="Surrounding text context")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                except Exception as display_error:
                    st.error(f"Error displaying table: {display_error}")
                    # Fallback: show basic table without column config
                    try:
                        st.dataframe(display_df, width='stretch', hide_index=True, use_container_width=True)
                    except Exception as fallback_error:
                        st.error(f"Fallback display also failed: {fallback_error}")
                        st.write("Raw data preview:")
                        st.write(display_df.head())
            except Exception as e:
                st.error(f"Error creating table: {e}")
                st.write("Raw hits data:")
                st.write(st.session_state.processing_hits[:5])  # Show first 5 hits
        else:
            st.warning("⚠️ No hits data available in session state")
            st.info("ℹ️ No flagged terms found in the document.")
            st.info("💡 Try adjusting your flagged terms list, replacement map, or skip terms configuration.")
        
        # Debug information (collapsed by default)
        with st.expander("🔍 Debug Information", expanded=False):
            st.write(f"**Session State Debug:**")
            st.write(f"- Processing Success: {st.session_state.processing_success}")
            st.write(f"- Number of Hits: {len(st.session_state.processing_hits) if st.session_state.processing_hits else 0}")
            st.write(f"- Processing Output Length: {len(st.session_state.processing_output) if st.session_state.processing_output else 0}")
            st.write(f"- Output Directory: {st.session_state.processing_outdir}")
            st.write(f"- Hits Type: {type(st.session_state.processing_hits)}")
            st.write(f"- Hits is None: {st.session_state.processing_hits is None}")
            st.write(f"- Hits is Empty: {len(st.session_state.processing_hits) == 0 if st.session_state.processing_hits else 'N/A'}")
            
            if st.session_state.processing_hits and len(st.session_state.processing_hits) > 0:
                st.write(f"**Sample Hit:**")
                st.json(st.session_state.processing_hits[0])
                st.write(f"**Hit Keys:** {list(st.session_state.processing_hits[0].keys()) if st.session_state.processing_hits[0] else 'No keys'}")
            else:
                st.write("**No hits data available**")
            
            # Recovery options
            st.write("**Recovery Options:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Session State", help="Clear all session data and start fresh"):
                    st.session_state.clear()
                    st.rerun()
            with col2:
                if st.button("🔄 Reload Page", help="Refresh the page to recover from errors"):
                    st.rerun()
        
        # Display processing output if available
        if st.session_state.processing_output:
            with st.expander("📋 Processing Log", expanded=False):
                st.text_area("Processing Output", value=st.session_state.processing_output, height=300, help="Detailed processing log from the language flagging script")
        
        # Download options with enhanced UI
        st.subheader("📥 Download Results")
        st.markdown("Download your processed results in various formats:")
        
        # Create download buttons in a more organized layout
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            st.markdown("#### 📊 Data Reports")
            
            if st.session_state.processing_hits:
                df = pd.DataFrame(st.session_state.processing_hits)
                
                # CSV download
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_data,
                    file_name=f"flag_report_{st.session_state.processing_timestamp}.csv",
                    mime="text/csv",
                    help="Download results as a CSV spreadsheet"
                )
                
                # JSON download
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON Report",
                    data=json_data,
                    file_name=f"flag_report_{st.session_state.processing_timestamp}.json",
                    mime="application/json",
                    help="Download results as a JSON file for programmatic use"
                )
            else:
                st.info("No data available for download")
        
        with download_col2:
            st.markdown("#### 📄 Documents")
            
            # PDF Report Generation
            if st.session_state.processing_hits and len(st.session_state.processing_hits) > 0:
                if st.button("📊 Generate PDF Report", help="Generate a comprehensive PDF report with summaries and visualizations"):
                    try:
                        # Load document analysis if available
                        document_analysis = {}
                        if st.session_state.processing_outdir:
                            analysis_file = os.path.join(st.session_state.processing_outdir, "document_analysis.json")
                            if os.path.exists(analysis_file):
                                with open(analysis_file, 'r', encoding='utf-8') as f:
                                    document_analysis = json.load(f)
                        
                        # Generate PDF report
                        with st.spinner("Generating PDF report..."):
                            report_path = generate_pdf_report(
                                st.session_state.processing_hits,
                                document_analysis,
                                st.session_state.processing_outdir or tempfile.gettempdir()
                            )
                        
                        # Provide download link
                        if os.path.exists(report_path):
                            with open(report_path, 'rb') as f:
                                report_data = f.read()
                            
                            st.download_button(
                                label="📊 Download PDF Report",
                                data=report_data,
                                file_name=os.path.basename(report_path),
                                mime="application/pdf",
                                help="Download the comprehensive PDF report with summaries and visualizations"
                            )
                            
                            st.success("✅ PDF report generated successfully!")
                        else:
                            st.error("Failed to generate PDF report")
                            
                    except Exception as e:
                        st.error(f"Error generating PDF report: {e}")
                        st.exception(e)
            else:
                st.info("No data available for PDF report generation")
            
            # Annotated PDF download
            if st.session_state.processing_outdir:
                annotated_pdf_path = os.path.join(st.session_state.processing_outdir, "flagged_output.pdf")
                if os.path.exists(annotated_pdf_path):
                    with open(annotated_pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📄 Download Annotated PDF",
                        data=pdf_data,
                        file_name=f"flagged_output_{st.session_state.processing_timestamp}.pdf",
                        mime="application/pdf",
                        help="Download the PDF with highlighted flagged terms"
                    )
                else:
                    st.warning("Annotated PDF not found")
            else:
                st.info("No output directory available")
        
        # ZIP download (full width)
        st.markdown("#### 📦 Complete Package")
        if st.session_state.processing_outdir and os.path.exists(st.session_state.processing_outdir):
            try:
                zip_path = f"results_{st.session_state.processing_timestamp}.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, dirs, files in os.walk(st.session_state.processing_outdir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, st.session_state.processing_outdir))
                
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                st.download_button(
                    label="📦 Download All Files (ZIP Archive)",
                    data=zip_data,
                    file_name=zip_path,
                    mime="application/zip",
                    help="Download all output files in a single ZIP archive"
                )
                
                # Clean up the temporary zip file
                os.unlink(zip_path)
            except Exception as e:
                st.error(f"Error creating ZIP file: {e}")
        else:
            st.info("No output files available for ZIP download")
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Flags", len(st.session_state.processing_hits))
        
        with col2:
            if st.session_state.processing_hits:
                unique_terms = len(set(hit.get('original_key', '') for hit in st.session_state.processing_hits))
                st.metric("Unique Terms", unique_terms)
            else:
                st.metric("Unique Terms", 0)
        
        with col3:
            if st.session_state.processing_hits:
                pages = len(set(hit.get('page_num', 0) for hit in st.session_state.processing_hits))
                st.metric("Pages Affected", pages)
            else:
                st.metric("Pages Affected", 0)
        
        with col4:
            if st.session_state.processing_hits:
                avg_suggestions = len(st.session_state.processing_hits) / max(1, len(set(hit.get('page_num', 0) for hit in st.session_state.processing_hits)))
                st.metric("Avg Flags/Page", f"{avg_suggestions:.1f}")
            else:
                st.metric("Avg Flags/Page", "0.0")
        
        st.markdown("---")
    
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
            
            # Start Over button (only show if there are results)
            if st.session_state.processing_success:
                st.markdown("---")
                st.subheader("🔄 Session Management")
                if st.button("🗑️ Start Over / Process New Document", type="secondary", width='stretch'):
                    # Clear all session state
                    st.session_state.processing_results = None
                    st.session_state.processing_success = False
                    st.session_state.processing_hits = []
                    st.session_state.processing_output = ""
                    st.session_state.processing_timestamp = None
                    st.session_state.processing_duration = None
                    st.session_state.processing_outdir = None
                    st.rerun()
            
            # Process button
            if flagged_terms and replacements:
                if st.button("🚀 Process Document", type="primary", width='stretch'):
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
                        
                        # Start timing the processing
                        processing_start_time = datetime.now()
                        
                        success, output, hits = run_processing(
                            tmp_file_path, flagged_terms, replacements, 
                            outdir, style, model, temperature, api_type, skip_terms, progress_container
                        )
                        
                        # Calculate processing duration
                        processing_end_time = datetime.now()
                        processing_duration = processing_end_time - processing_start_time
                        processing_duration_str = str(processing_duration).split('.')[0]  # Remove microseconds
                        
                        # Step 5: Processing complete
                        status_text.text("✅ Processing complete!")
                        details_text.text(f"Found {len(hits) if hits else 0} flagged terms")
                        progress_bar.progress(100)
                        
                        # Display processing output
                        if output:
                            st.subheader("📋 Processing Log")
                            st.text_area("Processing Output", value=output, height=300, help="Detailed processing log from the language flagging script")
                        
                        if success:
                            # Store results in session state
                            st.session_state.processing_success = True
                            st.session_state.processing_hits = hits if hits else []
                            st.session_state.processing_output = output if output else ""
                            st.session_state.processing_timestamp = timestamp
                            st.session_state.processing_duration = processing_duration_str
                            st.session_state.processing_outdir = outdir
                            
                            
                            # Clear progress indicators
                            progress_container.empty()
                            status_container.empty()
                            
                            st.success("✅ Document processed successfully!")
                            
                            # Show processing summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Flagged Terms Found", len(hits) if hits else 0)
                            
                            with col2:
                                st.metric("Processing Time", processing_duration_str)
                            with col3:
                                st.metric("Output Files", "1 PDF + Reports")
                            
                            # Force page refresh to show updated results
                            st.rerun()
                            
                            # Results are now displayed persistently above
                            st.info("📊 Results are displayed above and will persist until you click 'Start Over'")
                            
                            if hits:
                                st.success(f"✅ Found {len(hits)} flagged terms - see results above!")
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