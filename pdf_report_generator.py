#!/usr/bin/env python3
"""
PDF Report Generator for Smart PDF Language Flagger
Generates professional PDF reports with summaries, visualizations, and recommendations
"""

import os
import json
import tempfile
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import io
import base64

# Optional imports for chart generation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Charts will be skipped in PDF reports.")

class PDFReportGenerator:
    """Generate professional PDF reports for language flagging results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86AB')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#2E86AB'),
            borderWidth=1,
            borderColor=colors.HexColor('#2E86AB'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#A23B72')
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86AB')
        ))
    
    def create_chart_image(self, chart_type: str, data: Dict, title: str, 
                          width: int = 8, height: int = 6) -> str:
        """Create a chart and return it as a base64 encoded image."""
        if not CHART_AVAILABLE:
            # Return a placeholder if matplotlib is not available
            return self.create_text_placeholder(title, data)
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(width, height))
            
            if chart_type == 'bar':
                terms = list(data.keys())[:10]  # Top 10 terms
                counts = list(data.values())[:10]
                bars = ax.bar(terms, counts, color='#2E86AB', alpha=0.7)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Terms', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                
            elif chart_type == 'pie':
                # Take top 8 terms for pie chart
                terms = list(data.keys())[:8]
                counts = list(data.values())[:8]
                colors_list = plt.cm.Set3(range(len(terms)))
                wedges, texts, autotexts = ax.pie(counts, labels=terms, autopct='%1.1f%%', 
                                                colors=colors_list, startangle=90)
                ax.set_title(title, fontsize=14, fontweight='bold')
                
            elif chart_type == 'page_distribution':
                pages = list(data.keys())
                counts = list(data.values())
                ax.plot(pages, counts, marker='o', linewidth=2, markersize=6, color='#2E86AB')
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Page Number', fontsize=12)
                ax.set_ylabel('Number of Flags', fontsize=12)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Error creating chart: {e}")
            return self.create_text_placeholder(title, data)
    
    def create_text_placeholder(self, title: str, data: Dict) -> str:
        """Create a text-based placeholder when charts are not available."""
        # Create a simple text representation
        text_content = f"{title}\n\n"
        
        if len(data) > 0:
            # Sort by value and take top items
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            for i, (key, value) in enumerate(sorted_items[:10]):
                text_content += f"{i+1}. {key}: {value}\n"
        else:
            text_content += "No data available"
        
        # Create a simple text image using reportlab
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.lib.colors import black
        
        # Create a simple drawing
        d = Drawing(400, 300)
        d.add(String(50, 250, title, fontSize=14, fillColor=black))
        
        # Add data as text
        y_pos = 220
        for i, (key, value) in enumerate(sorted(data.items(), key=lambda x: x[1], reverse=True)[:8]):
            text = f"{key}: {value}"
            d.add(String(50, y_pos, text, fontSize=10, fillColor=black))
            y_pos -= 20
        
        # Convert to base64
        buffer = io.BytesIO()
        renderPDF.drawToFile(d, buffer, "PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
    
    def create_cover_page(self, doc, hits: List[Dict], document_analysis: Dict) -> None:
        """Create the cover page of the report."""
        # Title
        title = Paragraph("Language Analysis Report", self.styles['CustomTitle'])
        doc.build([title, Spacer(1, 0.5*inch)])
        
        # Document info
        doc_date = datetime.now().strftime("%B %d, %Y")
        doc_info = f"""
        <b>Analysis Date:</b> {doc_date}<br/>
        <b>Document Type:</b> {document_analysis.get('document_type', 'Unknown').title()}<br/>
        <b>Total Flags Found:</b> {len(hits)}<br/>
        <b>NIH Alignment Score:</b> {document_analysis.get('nih_alignment_score', 0):.1f}%<br/>
        <b>Project 2025 Score:</b> {document_analysis.get('project_2025_score', 0):.1f}%<br/>
        <b>Alabama SB 129 Score:</b> {document_analysis.get('alabama_sb129_score', 0):.1f}%
        """
        
        info_para = Paragraph(doc_info, self.styles['CustomBodyText'])
        doc.build([info_para, Spacer(1, 0.3*inch)])
        
        # Key themes
        themes = document_analysis.get('main_themes', [])
        if themes:
            themes_text = f"<b>Key Themes:</b> {', '.join(themes[:5])}"
            themes_para = Paragraph(themes_text, self.styles['CustomBodyText'])
            doc.build([themes_para])
        
        doc.build([PageBreak()])
    
    def create_executive_summary(self, doc, hits: List[Dict], document_analysis: Dict) -> None:
        """Create the executive summary section."""
        # Section header
        header = Paragraph("Executive Summary", self.styles['SectionHeader'])
        doc.build([header])
        
        # Key metrics
        total_flags = len(hits)
        unique_terms = len(set(hit.get('original_key', '') for hit in hits))
        pages_affected = len(set(hit.get('page_num', 0) for hit in hits))
        
        # Calculate average flags per page
        avg_flags_per_page = total_flags / max(1, pages_affected)
        
        # Most common terms
        term_counts = {}
        for hit in hits:
            term = hit.get('original_key', '')
            term_counts[term] = term_counts.get(term, 0) + 1
        
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_terms_text = ', '.join([f"{term} ({count})" for term, count in top_terms])
        
        summary_text = f"""
        This report presents the results of a comprehensive language analysis performed on the submitted document. 
        The analysis identified <b>{total_flags}</b> instances of flagged language across <b>{pages_affected}</b> pages, 
        involving <b>{unique_terms}</b> unique terms or phrases. On average, <b>{avg_flags_per_page:.1f}</b> flags 
        were found per page.
        
        The most frequently flagged terms were: <b>{top_terms_text}</b>.
        
        The document shows <b>{document_analysis.get('nih_alignment_score', 0):.1f}%</b> alignment with current NIH priorities, 
        <b>{document_analysis.get('project_2025_score', 0):.1f}%</b> alignment with Project 2025 healthcare principles, 
        and <b>{document_analysis.get('alabama_sb129_score', 0):.1f}%</b> compliance with Alabama State Bill 129.
        """
        
        summary_para = Paragraph(summary_text, self.styles['CustomBodyText'])
        doc.build([summary_para, Spacer(1, 0.2*inch)])
        
        # Key recommendations
        recommendations = document_analysis.get('strategic_recommendations', [])
        if recommendations:
            rec_header = Paragraph("Key Recommendations", self.styles['SubsectionHeader'])
            doc.build([rec_header])
            
            for i, rec in enumerate(recommendations[:3], 1):
                rec_text = f"<b>{i}.</b> {rec}"
                rec_para = Paragraph(rec_text, self.styles['CustomBodyText'])
                doc.build([rec_para])
        
        doc.build([PageBreak()])
    
    def create_document_analysis_section(self, doc, document_analysis: Dict) -> None:
        """Create the document analysis section."""
        header = Paragraph("Document Analysis", self.styles['SectionHeader'])
        doc.build([header])
        
        # Document type and themes
        doc_type = document_analysis.get('document_type', 'Unknown').title()
        themes = document_analysis.get('main_themes', [])
        scientific_focus = document_analysis.get('scientific_focus', [])
        
        analysis_text = f"""
        <b>Document Type:</b> {doc_type}<br/><br/>
        <b>Main Themes:</b> {', '.join(themes[:5]) if themes else 'Not identified'}<br/><br/>
        <b>Scientific Focus:</b> {', '.join(scientific_focus[:5]) if scientific_focus else 'Not identified'}<br/><br/>
        <b>Target Population:</b> {document_analysis.get('target_population', 'Not specified')}<br/><br/>
        <b>Intervention Approach:</b> {document_analysis.get('intervention_approach', 'Not specified')}
        """
        
        analysis_para = Paragraph(analysis_text, self.styles['CustomBodyText'])
        doc.build([analysis_para, Spacer(1, 0.2*inch)])
        
        # Alignment scores
        scores_header = Paragraph("Policy Alignment Scores", self.styles['SubsectionHeader'])
        doc.build([scores_header])
        
        # Create scores table
        scores_data = [
            ['Policy Framework', 'Alignment Score'],
            ['NIH Priorities', f"{document_analysis.get('nih_alignment_score', 0):.1f}%"],
            ['Project 2025', f"{document_analysis.get('project_2025_score', 0):.1f}%"],
            ['Alabama SB 129', f"{document_analysis.get('alabama_sb129_score', 0):.1f}%"]
        ]
        
        scores_table = Table(scores_data, colWidths=[2*inch, 1.5*inch])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        doc.build([scores_table, Spacer(1, 0.2*inch)])
        
        # NIH priorities alignment
        nih_priorities = document_analysis.get('nih_priority_alignment', [])
        if nih_priorities:
            nih_header = Paragraph("NIH Priority Alignment", self.styles['SubsectionHeader'])
            doc.build([nih_header])
            
            nih_text = f"<b>Aligned Priorities:</b> {', '.join(nih_priorities[:5])}"
            nih_para = Paragraph(nih_text, self.styles['CustomBodyText'])
            doc.build([nih_para])
        
        doc.build([PageBreak()])
    
    def create_flagged_terms_analysis(self, doc, hits: List[Dict]) -> None:
        """Create the flagged terms analysis section with charts."""
        header = Paragraph("Flagged Terms Analysis", self.styles['SectionHeader'])
        doc.build([header])
        
        # Calculate term statistics
        term_counts = {}
        page_distribution = {}
        
        for hit in hits:
            term = hit.get('original_key', '')
            page = hit.get('page_num', 0)
            
            term_counts[term] = term_counts.get(term, 0) + 1
            page_distribution[page] = page_distribution.get(page, 0) + 1
        
        # Top terms chart
        if term_counts:
            top_terms = dict(sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            chart_image = self.create_chart_image('bar', top_terms, 'Top 10 Flagged Terms')
            
            # Save chart to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(base64.b64decode(chart_image))
                chart_path = tmp_file.name
            
            # Add chart to document
            chart_img = Image(chart_path, width=6*inch, height=4*inch)
            doc.build([chart_img, Spacer(1, 0.2*inch)])
            
            # Clean up
            os.unlink(chart_path)
        
        # Page distribution chart
        if page_distribution:
            sorted_pages = dict(sorted(page_distribution.items()))
            chart_image = self.create_chart_image('page_distribution', sorted_pages, 'Flags Distribution by Page')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(base64.b64decode(chart_image))
                chart_path = tmp_file.name
            
            chart_img = Image(chart_path, width=6*inch, height=4*inch)
            doc.build([chart_img, Spacer(1, 0.2*inch)])
            
            os.unlink(chart_path)
        
        # Summary statistics
        total_terms = len(term_counts)
        total_flags = sum(term_counts.values())
        avg_flags_per_term = total_flags / max(1, total_terms)
        
        stats_text = f"""
        <b>Summary Statistics:</b><br/>
        • Total unique terms flagged: {total_terms}<br/>
        • Total flag instances: {total_flags}<br/>
        • Average flags per term: {avg_flags_per_term:.1f}<br/>
        • Pages with flags: {len(page_distribution)}
        """
        
        stats_para = Paragraph(stats_text, self.styles['CustomBodyText'])
        doc.build([stats_para, PageBreak()])
    
    def create_detailed_results_table(self, doc, hits: List[Dict]) -> None:
        """Create a detailed results table."""
        header = Paragraph("Detailed Results", self.styles['SectionHeader'])
        doc.build([header])
        
        if not hits:
            no_data = Paragraph("No flagged terms found.", self.styles['CustomBodyText'])
            doc.build([no_data])
            return
        
        # Prepare table data
        table_data = [['Page', 'Original Term', 'Matched Text', 'Suggestion', 'Reason']]
        
        for hit in hits[:50]:  # Limit to first 50 results for readability
            page = str(hit.get('page_num', ''))
            original = hit.get('original_key', '')[:30] + '...' if len(hit.get('original_key', '')) > 30 else hit.get('original_key', '')
            matched = hit.get('matched_text', '')[:30] + '...' if len(hit.get('matched_text', '')) > 30 else hit.get('matched_text', '')
            suggestion = hit.get('suggestion', '')[:40] + '...' if len(hit.get('suggestion', '')) > 40 else hit.get('suggestion', '')
            reason = hit.get('reason', '')[:50] + '...' if len(hit.get('reason', '')) > 50 else hit.get('reason', '')
            
            table_data.append([page, original, matched, suggestion, reason])
        
        # Create table
        results_table = Table(table_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1.5*inch, 2*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        doc.build([results_table])
        
        if len(hits) > 50:
            note = Paragraph(f"<i>Note: Showing first 50 of {len(hits)} total results.</i>", self.styles['CustomBodyText'])
            doc.build([Spacer(1, 0.1*inch), note])
    
    def generate_report(self, hits: List[Dict], document_analysis: Dict, 
                       output_path: str) -> str:
        """Generate the complete PDF report."""
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build the report
            self.create_cover_page(doc, hits, document_analysis)
            self.create_executive_summary(doc, hits, document_analysis)
            self.create_document_analysis_section(doc, document_analysis)
            self.create_flagged_terms_analysis(doc, hits)
            self.create_detailed_results_table(doc, hits)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating PDF report: {str(e)}")

def generate_pdf_report(hits: List[Dict], document_analysis: Dict, 
                       output_dir: str) -> str:
    """Generate a PDF report for the language flagging results."""
    generator = PDFReportGenerator()
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"language_analysis_report_{timestamp}.pdf")
    
    # Generate the report
    generator.generate_report(hits, document_analysis, output_path)
    
    return output_path
