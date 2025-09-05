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

def load_json_file(file_path: str) -> dict:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return {}

def load_document_analysis(outdir: str) -> dict:
    """Load document analysis data if available."""
    analysis_file = os.path.join(outdir, "document_analysis.json")
    if os.path.exists(analysis_file):
        return load_json_file(analysis_file)
    return {}

def display_document_analysis(analysis_data: dict) -> None:
    """Display document analysis information."""
    if not analysis_data:
        return
    
    st.subheader("📋 Document Analysis")
    
    # NIH alignment
    nih_alignment = analysis_data.get('nih_alignment', '')
    if nih_alignment:
        st.write("**🏥 NIH Alignment:**")
        st.write(nih_alignment)
    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🏷️ Terms Analysis", "📄 Page Analysis", "💡 Suggestions", "📈 Trends", "🧠 Hybrid Analysis"])
    
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
        
        # Most common suggestions
        if 'suggestion' in df.columns:
            suggestion_counts = df['suggestion'].value_counts().head(10)
            if not suggestion_counts.empty:
                fig_suggestions = px.bar(
                    x=suggestion_counts.values,
                    y=suggestion_counts.index,
                    orientation='h',
                    title="Top 10 Suggestions",
                    labels={'x': 'Frequency', 'y': 'Suggestion'}
                )
                fig_suggestions.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_suggestions, width='stretch')
        
        # Method analysis
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            fig_methods = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title="Flags by Analysis Method"
            )
            st.plotly_chart(fig_methods, width='stretch')
    
    with tab5:
        # Trends analysis
        st.subheader("📈 Analysis Trends")
        
        # Flags over pages (trend)
        page_trend = df.groupby('page_num').size().reset_index(name='flags')
        fig_trend = px.line(
            page_trend, 
            x='page_num', 
            y='flags',
            title="Flagging Trend Across Pages",
            labels={'page_num': 'Page Number', 'flags': 'Number of Flags'}
        )
        st.plotly_chart(fig_trend, width='stretch')
        
        # Method effectiveness over pages
        if 'method' in df.columns:
            method_trend = df.groupby(['page_num', 'method']).size().reset_index(name='count')
            fig_method_trend = px.line(
                method_trend,
                x='page_num',
                y='count',
                color='method',
                title="Method Performance Across Pages",
                labels={'page_num': 'Page Number', 'count': 'Flags Found'}
            )
            st.plotly_chart(fig_method_trend, width='stretch')
    
    with tab6:
        # Hybrid analysis
        st.subheader("🧠 Hybrid Analysis")
        
        if 'method' in df.columns:
            # Method comparison
            method_counts = df['method'].value_counts()
            
            # Method effectiveness metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pattern_count = method_counts.get('pattern_matching', 0)
                st.metric("Pattern Matching Only", pattern_count)
            
            with col2:
                langextract_count = method_counts.get('langextract', 0)
                st.metric("LangExtract Only", langextract_count)
            
            with col3:
                both_count = method_counts.get('both_methods', 0)
                st.metric("Found by Both", both_count)
            
            # Method comparison pie chart
            st.subheader("🔍 Method Comparison")
            fig_methods = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title="Flags by Analysis Method",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_methods, width='stretch')
            
            # Method effectiveness over pages
            st.subheader("📊 Method Performance by Page")
            method_page_analysis = df.groupby(['page_num', 'method']).size().unstack(fill_value=0)
            
            if not method_page_analysis.empty:
                fig_method_pages = px.bar(
                    method_page_analysis,
                    title="Method Performance by Page",
                    labels={'page_num': 'Page Number', 'value': 'Flags Found'},
                    barmode='group'
                )
                fig_method_pages.update_layout(xaxis_title="Page Number", yaxis_title="Flags Found")
                st.plotly_chart(fig_method_pages, width='stretch')
            
            # Method overlap analysis
            st.subheader("🔄 Method Overlap Analysis")
            
            # Calculate overlap statistics
            total_flags = len(df)
            pattern_only = method_counts.get('pattern_matching', 0)
            langextract_only = method_counts.get('langextract', 0)
            both_methods = method_counts.get('both_methods', 0)
            
            overlap_stats = {
                'Pattern Only': pattern_only,
                'LangExtract Only': langextract_only,
                'Both Methods': both_methods
            }
            
            fig_overlap = px.bar(
                x=list(overlap_stats.keys()),
                y=list(overlap_stats.values()),
                title="Method Overlap Distribution",
                labels={'x': 'Method Category', 'y': 'Number of Flags'},
                color=list(overlap_stats.values()),
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_overlap, width='stretch')
            
            # Span grounding analysis
            if 'span_grounding' in df.columns:
                st.subheader("🎯 Span Grounding Analysis")
                
                grounding_counts = df['span_grounding'].value_counts()
                
                # Grounding method distribution
                fig_grounding = px.pie(
                    values=grounding_counts.values,
                    names=grounding_counts.index,
                    title="Span Grounding Methods Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_grounding, width='stretch')
                
                # Grounding accuracy by method
                if 'confidence' in df.columns:
                    grounding_confidence = df.groupby('span_grounding')['confidence'].agg(['mean', 'count']).reset_index()
                    grounding_confidence = grounding_confidence[grounding_confidence['count'] > 0]
                    
                    fig_grounding_conf = px.bar(
                        grounding_confidence,
                        x='span_grounding',
                        y='mean',
                        title="Average Confidence by Grounding Method",
                        labels={'mean': 'Average Confidence', 'span_grounding': 'Grounding Method'},
                        color='mean',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_grounding_conf.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_grounding_conf, width='stretch')
            
            # Method effectiveness summary
            st.subheader("📈 Method Effectiveness Summary")
            
            effectiveness_data = {
                'Metric': [
                    'Pattern Matching Coverage',
                    'LangExtract Coverage', 
                    'Combined Coverage',
                    'Overlap Rate',
                    'Unique Pattern Finds',
                    'Unique LangExtract Finds'
                ],
                'Value': [
                    f"{(pattern_only + both_methods) / total_flags * 100:.1f}%",
                    f"{(langextract_only + both_methods) / total_flags * 100:.1f}%",
                    f"{((pattern_only + langextract_only + both_methods) / total_flags) * 100:.1f}%",
                    f"{both_methods / total_flags * 100:.1f}%",
                    f"{pattern_only} ({pattern_only / total_flags * 100:.1f}%)",
                    f"{langextract_only} ({langextract_only / total_flags * 100:.1f}%)"
                ]
            }
            
            effectiveness_df = pd.DataFrame(effectiveness_data)
            st.dataframe(effectiveness_df, hide_index=True, use_container_width=True)
            
            # Method-specific term analysis
            st.subheader("🏷️ Method-Specific Term Analysis")
            
            # Terms found by each method
            pattern_terms = df[df['method'] == 'pattern_matching']['original_key'].value_counts().head(10)
            langextract_terms = df[df['method'] == 'langextract']['original_key'].value_counts().head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not pattern_terms.empty:
                    st.write("**Top Pattern Matching Terms:**")
                    fig_pattern = px.bar(
                        x=pattern_terms.values,
                        y=pattern_terms.index,
                        orientation='h',
                        title="Top Pattern Matching Terms",
                        labels={'x': 'Count', 'y': 'Term'}
                    )
                    fig_pattern.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    st.plotly_chart(fig_pattern, width='stretch')
                else:
                    st.info("No pattern matching terms found")
            
            with col2:
                if not langextract_terms.empty:
                    st.write("**Top LangExtract Terms:**")
                    fig_langextract = px.bar(
                        x=langextract_terms.values,
                        y=langextract_terms.index,
                        orientation='h',
                        title="Top LangExtract Terms",
                        labels={'x': 'Count', 'y': 'Term'}
                    )
                    fig_langextract.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                    st.plotly_chart(fig_langextract, width='stretch')
                else:
                    st.info("No LangExtract terms found")
        else:
            st.info("No method information available for hybrid analysis")

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
            ["Hybrid (Pattern + LangExtract)", "Standard Pattern Matching", "LangExtract Only"],
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
            
            # Confidence threshold settings for hybrid mode
            st.subheader("🎯 Confidence Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Only show suggestions with confidence above this threshold"
            )
            st.info(f"🎯 Showing suggestions with confidence ≥ {confidence_threshold:.1f}")
            
            # Hybrid strategy settings
            st.subheader("🔀 Hybrid Strategy")
            hybrid_strategy = st.selectbox(
                "Matching Strategy",
                ["conservative", "basic", "advanced"],
                index=1,
                help="How aggressively to merge pattern matching and LangExtract results"
            )
            # Store in session state for use in display_results
            st.session_state.hybrid_strategy = hybrid_strategy
            strategy_descriptions = {
                "conservative": "Only merge exact overlaps - safest but may miss related hits",
                "basic": "Merge overlaps and exact text matches - balanced approach",
                "advanced": "Merge overlaps, semantic similarity, and text containment - most comprehensive"
            }
            st.info(f"📋 {strategy_descriptions[hybrid_strategy]}")
            
            # Chunking settings for large documents
            st.subheader("📄 Document Chunking")
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=5000,
                max_value=20000,
                value=15000,
                step=1000,
                help="Size of text chunks for processing large documents"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=100,
                max_value=1000,
                value=750,
                step=50,
                help="Overlap between chunks to prevent missing hits at boundaries"
            )
            st.info(f"📄 Chunks: {chunk_size} chars with {chunk_overlap} overlap")
        else:
            st.info("🔍 Standard pattern matching mode")
            confidence_threshold = 0.5  # Default value
            hybrid_strategy = "basic"  # Default value
            # Store in session state for use in display_results
            st.session_state.hybrid_strategy = hybrid_strategy
            chunk_size = 10000  # Default value
            chunk_overlap = 500  # Default value
        
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
    col1, col2 = st.columns([3, 2])
    
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
            
            # Configuration Review Section
            st.markdown("---")
            st.header("📋 Configuration Review & Edit")
            st.markdown("Review and edit your flagged terms and replacement maps before processing:")
            
            # Get configuration files based on preset
            flagged_file, replacements_file = get_config_files(config_preset)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📝 Flagged Terms Configuration")
                
                # Load flagged terms based on preset
                if config_preset != "Custom" and os.path.exists(flagged_file):
                    with open(flagged_file, 'r') as f:
                        default_flagged_terms = json.load(f)
                else:
                    default_flagged_terms = []
                
                # Flagged terms as editable text area
                st.markdown("**Terms to Flag** (one per line):")
                
                flagged_terms_text = st.text_area(
                    "Flagged Terms",
                    value="\n".join(default_flagged_terms),
                    height=200,
                    help="Enter terms to flag, one per line",
                    key="flagged_terms_edit"
                )
                
                # Parse the terms
                flagged_terms = [term.strip() for term in flagged_terms_text.split('\n') if term.strip()]
                
                # Show count and preview
                st.info(f"📊 Total flagged terms: {len(flagged_terms)}")
                
                if flagged_terms:
                    st.markdown("**Preview (first 10 terms):**")
                    for i, term in enumerate(flagged_terms[:10]):
                        st.write(f"{i+1}. {term}")
                    if len(flagged_terms) > 10:
                        st.write(f"... and {len(flagged_terms) - 10} more")
            
            with col2:
                st.subheader("🔄 Replacement Map Configuration")
                
                # Load replacements based on preset
                if config_preset != "Custom" and os.path.exists(replacements_file):
                    with open(replacements_file, 'r') as f:
                        default_replacements = json.load(f)
                else:
                    default_replacements = {}
                
                # Replacements as editable text area
                st.markdown("**Term Replacements** (JSON format):")
                
                replacements_text = st.text_area(
                    "Replacements JSON",
                    value=json.dumps(default_replacements, indent=2),
                    height=200,
                    help="Enter replacements as JSON object (term: replacement)",
                    key="replacements_edit"
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
                
                # Show count and preview
                st.info(f"📊 Total replacements: {len(replacements)}")
                
                if replacements:
                    st.markdown("**Preview (first 10 replacements):**")
                    for i, (term, replacement) in enumerate(list(replacements.items())[:10]):
                        st.write(f"{i+1}. '{term}' → '{replacement}'")
                    if len(replacements) > 10:
                        st.write(f"... and {len(replacements) - 10} more")
            
            # Skip Terms Configuration
            st.subheader("⏭️ Skip Terms Configuration")
            skip_terms_text = st.text_area(
                "Skip Terms",
                value="\n".join(skip_terms),
                height=100,
                help="Enter terms to skip during analysis, one per line",
                key="skip_terms_edit"
            )
            
            # Parse skip terms
            skip_terms = [term.strip() for term in skip_terms_text.split('\n') if term.strip()]
            st.info(f"📊 Total skip terms: {len(skip_terms)}")
            
            # Processing Options Summary
            st.markdown("---")
            st.header("⚙️ Processing Options Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Analysis Mode:**")
                st.write(f"🔍 {analysis_mode}")
            
            with col2:
                st.markdown("**API Provider:**")
                st.write(f"🤖 {api_provider}")
            
            with col3:
                st.markdown("**Model:**")
                st.write(f"🧠 {model}")
            
            # Process Button
            st.markdown("---")
            if st.button("🚀 Process Document", type="primary", help="Begin document analysis with current configuration"):
                process_document(input_file, analysis_mode, api_provider, model, temperature, skip_terms, config_preset, flagged_terms, replacements, confidence_threshold, hybrid_strategy, chunk_size, chunk_overlap)
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Display session state stats
        if 'processing_success' in st.session_state and st.session_state.processing_success:
            # Main metrics in a more compact layout
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric("Total Flags", len(st.session_state.processing_hits))
                st.metric("Pages Processed", len(set(hit.get('page_num', 0) for hit in st.session_state.processing_hits)))
            
            with col2b:
                st.metric("Processing Time", f"{st.session_state.processing_duration:.1f}s" if st.session_state.processing_duration else "N/A")
                
                if st.session_state.processing_hits:
                    # Analysis method breakdown
                    methods = {}
                    for hit in st.session_state.processing_hits:
                        method = hit.get('method', 'unknown')
                        methods[method] = methods.get(method, 0) + 1
                    
                    if methods:
                        st.markdown("**Analysis Methods:**")
                        for method, count in methods.items():
                            st.write(f"• {method.replace('_', ' ').title()}: {count}")
        else:
            st.info("📄 Upload and process a document to see statistics")

def process_document(input_file: str, analysis_mode: str, api_provider: str, model: str, 
                    temperature: float, skip_terms: List[str], config_preset: str, 
                    flagged_terms: List[str] = None, replacements: Dict[str, str] = None,
                    confidence_threshold: float = 0.5, hybrid_strategy: str = "advanced",
                    chunk_size: int = 10000, chunk_overlap: int = 500):
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
    
    # Get configuration files based on preset (fallback if not provided)
    if flagged_terms is None or replacements is None:
        flagged_file, replacements_file = get_config_files(config_preset)
    else:
        # Use provided terms and create temporary files
        flagged_file = None
        replacements_file = None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"hybrid_output_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    
    # Prepare command
    if flagged_file and replacements_file:
        # Use preset files
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
    else:
        # Use edited terms - create temporary files
        import tempfile
        
        # Create temporary flagged terms file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(flagged_terms, f, indent=2)
            flagged_file = f.name
        
        # Create temporary replacements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(replacements, f, indent=2)
            replacements_file = f.name
        
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
                
                # Apply confidence filtering if confidence column exists
                if 'confidence' in df.columns and confidence_threshold > 0:
                    original_count = len(hits)
                    hits = [hit for hit in hits if hit.get('confidence', 0.0) >= confidence_threshold]
                    filtered_count = len(hits)
                    if original_count != filtered_count:
                        st.info(f"🎯 Confidence filtering: {original_count} → {filtered_count} hits (threshold: {confidence_threshold:.1f})")
                
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
    
    # Clean up temporary files if they were created
    if flagged_file and flagged_file.startswith('/tmp'):
        try:
            os.unlink(flagged_file)
        except:
            pass
    
    if replacements_file and replacements_file.startswith('/tmp'):
        try:
            os.unlink(replacements_file)
        except:
            pass

def display_results():
    """Display comprehensive processing results."""
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
            
            # Hybrid-specific metrics
            if st.session_state.processing_hits:
                methods = {}
                for hit in st.session_state.processing_hits:
                    method = hit.get('method', 'unknown')
                    methods[method] = methods.get(method, 0) + 1
                
                if methods:
                    st.subheader("🔍 Hybrid Analysis Breakdown")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pattern_count = methods.get('pattern_matching', 0)
                        st.metric("Pattern Matching", pattern_count)
                    
                    with col2:
                        langextract_count = methods.get('langextract', 0)
                        st.metric("LangExtract", langextract_count)
                    
                    with col3:
                        both_count = methods.get('both_methods', 0)
                        st.metric("Found by Both", both_count)
                    
                    # Show method effectiveness
                    total_hits = len(st.session_state.processing_hits)
                    if total_hits > 0:
                        st.info(f"📊 **Method Effectiveness:** Pattern matching found {pattern_count} ({pattern_count/total_hits*100:.1f}%), LangExtract found {langextract_count} ({langextract_count/total_hits*100:.1f}%), Both methods found {both_count} ({both_count/total_hits*100:.1f}%)")
                        
                        # Show confidence statistics
                        confidences = [hit.get('confidence', 0.0) for hit in st.session_state.processing_hits if 'confidence' in hit]
                        if confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                            high_conf = len([c for c in confidences if c >= 0.8])
                            medium_conf = len([c for c in confidences if 0.5 <= c < 0.8])
                            low_conf = len([c for c in confidences if c < 0.5])
                            
                            st.subheader("🎯 Confidence Analysis")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                            with col2:
                                st.metric("High Confidence (≥0.8)", high_conf)
                            with col3:
                                st.metric("Medium Confidence (0.5-0.8)", medium_conf)
                            with col4:
                                st.metric("Low Confidence (<0.5)", low_conf)
                            
                            # Show hybrid strategy effectiveness
                            st.subheader("🔀 Hybrid Strategy Effectiveness")
                            try:
                                from hybrid_language_flagger import HybridLanguageFlagger
                                
                                # Create a temporary hybrid flagger for analysis
                                temp_flagger = HybridLanguageFlagger(
                                    flagged_terms=[],
                                    replacement_map={},
                                    hybrid_strategy=st.session_state.get('hybrid_strategy', 'basic')
                                )
                                
                                effectiveness = temp_flagger.analyze_hybrid_effectiveness(st.session_state.processing_hits)
                                
                                if "error" not in effectiveness:
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Pattern Coverage", f"{effectiveness['coverage_metrics']['pattern_coverage']:.1%}")
                                    with col2:
                                        st.metric("LangExtract Coverage", f"{effectiveness['coverage_metrics']['langextract_coverage']:.1%}")
                                    with col3:
                                        st.metric("Overlap Rate", f"{effectiveness['coverage_metrics']['overlap_rate']:.1%}")
                                    with col4:
                                        strategy_quality = "High" if effectiveness['strategy_effectiveness']['high_overlap'] else "Medium"
                                        st.metric("Strategy Quality", strategy_quality)
                                    
                                    # Show effectiveness indicators
                                    indicators = []
                                    if effectiveness['strategy_effectiveness']['high_overlap']:
                                        indicators.append("✅ High method overlap")
                                    if effectiveness['strategy_effectiveness']['balanced_coverage']:
                                        indicators.append("✅ Balanced coverage")
                                    if effectiveness['strategy_effectiveness']['high_confidence']:
                                        indicators.append("✅ High confidence")
                                    
                                    if indicators:
                                        st.success(" | ".join(indicators))
                                    else:
                                        st.warning("⚠️ Consider adjusting hybrid strategy settings")
                                        
                            except Exception as e:
                                st.warning(f"Could not analyze hybrid effectiveness: {e}")
                            
                            # Show span grounding statistics
                            if 'span_grounding' in df.columns and not df['span_grounding'].isna().all():
                                grounding_counts = df['span_grounding'].value_counts()
                                st.subheader("🎯 Span Grounding Methods")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Direct LangExtract", grounding_counts.get('langextract_direct', 0))
                                with col2:
                                    st.metric("Fuzzy Matching", grounding_counts.get('fuzzy_exact', 0) + grounding_counts.get('fuzzy_word_match', 0))
                                with col3:
                                    st.metric("Pattern Matching", grounding_counts.get('pattern_matching_exact', 0))
                                with col4:
                                    st.metric("Simple Find", grounding_counts.get('simple_find', 0))
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
                                "context": st.column_config.TextColumn("Context", help="Surrounding text context"),
                                "method": st.column_config.TextColumn("Method", help="Analysis method used"),
                                "confidence": st.column_config.NumberColumn("Confidence", help="Confidence score (0.0-1.0)", format="%.2f"),
                                "char_start": st.column_config.NumberColumn("Start", help="Character start position"),
                                "char_end": st.column_config.NumberColumn("End", help="Character end position"),
                                "span_length": st.column_config.NumberColumn("Length", help="Span length in characters"),
                                "span_grounding": st.column_config.TextColumn("Grounding", help="Method used for span detection")
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
            if PDF_REPORT_AVAILABLE:
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
            else:
                st.info("📊 PDF Report generation not available in this environment")
            
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
                            arcname = os.path.relpath(file_path, st.session_state.processing_outdir)
                            zipf.write(file_path, arcname)
                
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                st.download_button(
                    label="📦 Download Complete Package",
                    data=zip_data,
                    file_name=zip_path,
                    mime="application/zip",
                    help="Download all output files as a ZIP archive"
                )
                
                # Clean up the zip file
                os.remove(zip_path)
                
            except Exception as e:
                st.error(f"Error creating ZIP file: {e}")
        else:
            st.info("No output directory available for ZIP download")

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
