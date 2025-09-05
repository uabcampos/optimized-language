#!/usr/bin/env python3
"""
Hybrid Language Flagging System
Combines traditional pattern matching with LangExtract semantic analysis
"""

import os
import json
import textwrap
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

# Optional LangExtract import
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("Warning: LangExtract not available. Hybrid mode will use pattern matching only.")

class HybridLanguageFlagger:
    """Hybrid language flagging system combining pattern matching and LangExtract."""
    
    def __init__(self, flagged_terms: List[str], replacement_map: Dict[str, str], 
                 skip_terms: List[str] = None, use_langextract: bool = True,
                 confidence_threshold: float = 0.5, hybrid_strategy: str = "advanced"):
        self.flagged_terms = flagged_terms
        self.replacement_map = replacement_map
        self.skip_terms = skip_terms or []
        self.use_langextract = use_langextract and LANGEXTRACT_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.hybrid_strategy = hybrid_strategy  # "basic", "advanced", "conservative"
        
        # LangExtract examples for different types of problematic language
        self.langextract_examples = self._setup_langextract_examples()
        
        print(f"🔧 Hybrid Language Flagger initialized:")
        print(f"   - Pattern matching: ✅ Enabled")
        print(f"   - LangExtract: {'✅ Enabled' if self.use_langextract else '❌ Disabled'}")
        print(f"   - Flagged terms: {len(self.flagged_terms)}")
        print(f"   - Skip terms: {len(self.skip_terms)}")
        print(f"   - Confidence threshold: {self.confidence_threshold}")
        print(f"   - Hybrid strategy: {self.hybrid_strategy}")
    
    def _setup_langextract_examples(self) -> List[Any]:
        """Setup few-shot examples for LangExtract, focusing on semantic issues not covered by pattern matching."""
        if not LANGEXTRACT_AVAILABLE:
            return []
        
        # Create comprehensive skip list to avoid conflicts with pattern matching
        skip_terms_comprehensive = set(self.skip_terms)
        skip_terms_comprehensive.update(self.flagged_terms)
        skip_terms_comprehensive.update(self.replacement_map.keys())
        skip_terms_comprehensive.update(self.replacement_map.values())
        
        # Focus on semantic issues that pattern matching might miss
        examples = [
            lx.data.ExampleData(
                text="The program targets at-risk youth who are struggling with addiction",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="at-risk youth",
                        attributes={
                            "suggestion": "youth facing challenges",
                            "reason": "Deficit-framing language, focuses on problems rather than potential",
                            "severity": "medium",
                            "category": "deficit_framing"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="stigmatizing_language",
                        extraction_text="struggling with addiction",
                        attributes={
                            "suggestion": "experiencing substance use challenges",
                            "reason": "More person-centered language that reduces stigma",
                            "severity": "high",
                            "category": "stigmatizing_language"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="The intervention addresses the needs of marginalized communities",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="marginalized communities",
                        attributes={
                            "suggestion": "underserved communities",
                            "reason": "More empowering language that focuses on service gaps rather than marginalization",
                            "severity": "medium",
                            "category": "deficit_framing"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="The study examines the impact on hard-to-reach populations",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="hard-to-reach populations",
                        attributes={
                            "suggestion": "populations with limited access",
                            "reason": "Less deficit-framing language that focuses on systemic barriers rather than population characteristics",
                            "severity": "low",
                            "category": "deficit_framing"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="The program serves low-income families and their children",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="low-income families",
                        attributes={
                            "suggestion": "families experiencing economic challenges",
                            "reason": "More person-centered language that focuses on circumstances rather than defining characteristics",
                            "severity": "low",
                            "category": "deficit_framing"
                        }
                    )
                ]
            )
        ]
        
        # Only add examples that don't conflict with our flagged terms
        filtered_examples = []
        for example in examples:
            # Check if any extraction text conflicts with our flagged terms
            has_conflict = False
            for extraction in example.extractions:
                if any(term.lower() in extraction.extraction_text.lower() for term in skip_terms_comprehensive):
                    has_conflict = True
                    break
            
            if not has_conflict:
                filtered_examples.append(example)
        
        return filtered_examples
    
    def _validate_suggestion(self, suggestion: str) -> Tuple[str, str]:
        """Validate suggestion to ensure it doesn't contain flagged terms."""
        if not suggestion:
            return "Consider alternative phrasing", "No suggestion provided"
        
        # Create skip list for flagged terms only (not replacements)
        skip_terms_comprehensive = set(self.skip_terms)
        skip_terms_comprehensive.update(self.flagged_terms)
        skip_terms_comprehensive.update(self.replacement_map.keys())  # Only original flagged terms
        
        # Check if suggestion contains any flagged terms (word boundary matching)
        suggestion_lower = suggestion.lower()
        import re
        
        for term in skip_terms_comprehensive:
            term_lower = term.lower()
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(term_lower) + r'\b', suggestion_lower):
                # Try to find a better suggestion from replacement map
                for original, replacement in self.replacement_map.items():
                    if re.search(r'\b' + re.escape(original.lower()) + r'\b', suggestion_lower):
                        # Replace the flagged term with its approved alternative
                        updated_suggestion = re.sub(
                            r'\b' + re.escape(original.lower()) + r'\b', 
                            replacement, 
                            suggestion_lower, 
                            flags=re.IGNORECASE
                        )
                        return updated_suggestion, f"Replaced flagged term '{original}' with approved alternative"
                
                # If no direct replacement, provide a generic alternative
                return "Consider alternative phrasing", f"Suggestion contained flagged term '{term}'"
        
        return suggestion, "Validated suggestion"
    
    def _extract_span_info(self, extraction, text: str) -> Dict[str, Any]:
        """Extract precise span information for LangExtract extraction."""
        extraction_text = extraction.extraction_text
        
        # Try multiple methods to find the span
        span_info = {
            "found": False,
            "start": 0,
            "end": 0,
            "grounding_method": "none"
        }
        
        # Method 1: Check if LangExtract provides direct span information
        if hasattr(extraction, 'start') and hasattr(extraction, 'end'):
            span_info.update({
                "found": True,
                "start": extraction.start,
                "end": extraction.end,
                "grounding_method": "langextract_direct"
            })
            return span_info
        
        # Method 2: Check attributes for span information
        span_attrs = ['start', 'end', 'span', 'position', 'offset', 'start_char', 'end_char', 'char_start', 'char_end']
        for attr in span_attrs:
            if attr in extraction.attributes:
                value = extraction.attributes[attr]
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    span_info.update({
                        "found": True,
                        "start": value[0],
                        "end": value[1],
                        "grounding_method": f"langextract_attr_{attr}"
                    })
                    return span_info
                elif isinstance(value, dict) and 'start' in value and 'end' in value:
                    span_info.update({
                        "found": True,
                        "start": value['start'],
                        "end": value['end'],
                        "grounding_method": f"langextract_attr_{attr}_dict"
                    })
                    return span_info
        
        # Method 3: Check metadata for span information
        if hasattr(extraction, 'metadata') and extraction.metadata:
            for key, value in extraction.metadata.items():
                if any(span_term in key.lower() for span_term in ['span', 'position', 'offset', 'start', 'end']):
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        span_info.update({
                            "found": True,
                            "start": value[0],
                            "end": value[1],
                            "grounding_method": f"langextract_metadata_{key}"
                        })
                        return span_info
        
        # Method 4: Enhanced text matching with fuzzy search
        span_info = self._fuzzy_span_search(extraction_text, text)
        if span_info["found"]:
            return span_info
        
        # Method 5: Fallback to simple find
        pos = text.lower().find(extraction_text.lower())
        if pos != -1:
            span_info.update({
                "found": True,
                "start": pos,
                "end": pos + len(extraction_text),
                "grounding_method": "simple_find"
            })
        
        return span_info
    
    def _fuzzy_span_search(self, extraction_text: str, text: str) -> Dict[str, Any]:
        """Enhanced fuzzy search for finding text spans with better accuracy."""
        import re
        
        # Clean the extraction text for better matching
        clean_extraction = re.sub(r'\s+', ' ', extraction_text.strip())
        clean_text = re.sub(r'\s+', ' ', text)
        
        # Try exact match first
        pos = clean_text.lower().find(clean_extraction.lower())
        if pos != -1:
            return {
                "found": True,
                "start": pos,
                "end": pos + len(clean_extraction),
                "grounding_method": "fuzzy_exact"
            }
        
        # Try word-by-word matching for better accuracy
        extraction_words = clean_extraction.lower().split()
        text_words = clean_text.lower().split()
        
        for i in range(len(text_words) - len(extraction_words) + 1):
            window = text_words[i:i + len(extraction_words)]
            if window == extraction_words:
                # Calculate character positions
                start_char = len(' '.join(text_words[:i]))
                if i > 0:
                    start_char += 1  # Account for space
                end_char = start_char + len(clean_extraction)
                
                return {
                    "found": True,
                    "start": start_char,
                    "end": end_char,
                    "grounding_method": "fuzzy_word_match"
                }
        
        # Try partial matching for cases with slight variations
        for i in range(len(text_words)):
            for j in range(i + 1, min(i + len(extraction_words) + 2, len(text_words) + 1)):
                window_text = ' '.join(text_words[i:j])
                if self._text_similarity(clean_extraction, window_text) > 0.8:
                    start_char = len(' '.join(text_words[:i]))
                    if i > 0:
                        start_char += 1
                    end_char = start_char + len(window_text)
                    
                    return {
                        "found": True,
                        "start": start_char,
                        "end": end_char,
                        "grounding_method": "fuzzy_partial_match"
                    }
        
        return {"found": False, "start": 0, "end": 0, "grounding_method": "none"}
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        longer = text1 if len(text1) > len(text2) else text2
        shorter = text2 if len(text1) > len(text2) else text1
        
        if len(longer) == 0:
            return 1.0
        
        matches = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matches / len(longer)
    
    def _pattern_match_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Traditional pattern matching analysis."""
        hits = []
        text_lower = text.lower()
        
        for term in self.flagged_terms:
            if term.lower() in self.skip_terms:
                continue
                
            if term.lower() in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(term.lower(), start)
                    if pos == -1:
                        break
                    
                    # Get context around the match
                    context_start = max(0, pos - 50)
                    context_end = min(len(text), pos + len(term) + 50)
                    context = text[context_start:context_end]
                    
                    # Get matched text (preserve original case)
                    matched_text = text[pos:pos + len(term)]
                    
                    # Calculate confidence for pattern matching (high confidence for exact matches)
                    confidence = 0.95  # High confidence for exact pattern matches
                    
                    hits.append({
                        "original_key": term,
                        "matched_text": matched_text,
                        "context": context,
                        "position": pos,
                        "char_start": pos,
                        "char_end": pos + len(term),
                        "span_length": len(term),
                        "method": "pattern_matching",
                        "suggestion": self.replacement_map.get(term, "Consider alternative"),
                        "reason": "Matched flagged term",
                        "severity": "medium",
                        "confidence": confidence,
                        "span_grounding": "pattern_matching_exact"
                    })
                    
                    start = pos + 1
        
        return hits
    
    def _langextract_analysis(self, text: str) -> List[Dict[str, Any]]:
        """LangExtract semantic analysis."""
        if not self.use_langextract:
            return []
        
        try:
            print(f"      📝 Preparing LangExtract prompt...")
            prompt = textwrap.dedent("""
            Extract instances of deficit-framing language, bias, or non-inclusive terminology that may not be caught by traditional pattern matching.
            Focus on semantic issues like:
            - Deficit-framing language that emphasizes problems over potential
            - Stigmatizing language that reduces people to their circumstances
            - Problem-focused language that emphasizes obstacles over solutions
            - Language that implies helplessness or lack of agency
            
            Use exact text for extractions. Do not paraphrase or overlap entities.
            Provide meaningful attributes including suggestions and reasoning.
            Avoid terms that are already covered by traditional flagged term lists.
            """)
            
            print(f"      🚀 Calling LangExtract API...")
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=self.langextract_examples,
                model_id="gemini-2.5-flash"
            )
            
            print(f"      🔄 Processing LangExtract results...")
            hits = []
            for i, extraction in enumerate(result.extractions):
                # Get precise span information
                span_info = self._extract_span_info(extraction, text)
                
                if span_info["found"]:
                    # Get context around the extraction
                    context_start = max(0, span_info["start"] - 50)
                    context_end = min(len(text), span_info["end"] + 50)
                    context = text[context_start:context_end]
                    
                    # Validate the suggestion
                    original_suggestion = extraction.attributes.get("suggestion", "Consider alternative")
                    validated_suggestion, validation_reason = self._validate_suggestion(original_suggestion)
                    
                    # Extract confidence score from attributes or calculate default
                    confidence = extraction.attributes.get("confidence", 0.8)  # Default confidence
                    
                    # If confidence is not provided, try to extract from Gemini response metadata
                    if hasattr(extraction, 'confidence') and extraction.confidence is not None:
                        confidence = extraction.confidence
                    elif hasattr(extraction, 'metadata') and extraction.metadata:
                        confidence = extraction.metadata.get("confidence", confidence)
                    
                    hits.append({
                        "original_key": extraction.extraction_text,
                        "matched_text": extraction.extraction_text,
                        "context": context,
                        "position": span_info["start"],
                        "char_start": span_info["start"],
                        "char_end": span_info["end"],
                        "span_length": span_info["end"] - span_info["start"],
                        "method": "langextract",
                        "suggestion": validated_suggestion,
                        "reason": f"{extraction.attributes.get('reason', 'Semantic analysis')} ({validation_reason})",
                        "severity": extraction.attributes.get("severity", "medium"),
                        "category": extraction.attributes.get("category", "unknown"),
                        "confidence": confidence,
                        "span_grounding": span_info["grounding_method"]
                    })
                
                if (i + 1) % 5 == 0:  # Progress update every 5 extractions
                    print(f"         Processed {i + 1}/{len(result.extractions)} extractions...")
            
            print(f"      ✅ LangExtract processing complete!")
            return hits
            
        except Exception as e:
            print(f"      ❌ LangExtract analysis failed: {e}")
            return []
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using hybrid approach."""
        print(f"🔍 Analyzing text with hybrid approach...")
        print(f"   Text length: {len(text)} characters")
        
        # Pattern matching analysis
        print(f"   🔧 Running pattern matching analysis...")
        pattern_hits = self._pattern_match_analysis(text)
        print(f"   ✅ Pattern matching found: {len(pattern_hits)} hits")
        
        # LangExtract analysis
        langextract_hits = []
        if self.use_langextract:
            print(f"   🧠 Running LangExtract semantic analysis...")
            print(f"   ⏳ This may take 10-30 seconds for complex text...")
            langextract_hits = self._langextract_analysis(text)
            print(f"   ✅ LangExtract found: {len(langextract_hits)} hits")
        else:
            print(f"   ⏭️  Skipping LangExtract (not available)")
        
        # Combine and deduplicate results
        print(f"   🔄 Combining and deduplicating results...")
        all_hits = pattern_hits + langextract_hits
        deduplicated_hits = self._deduplicate_hits(all_hits)
        
        # Analysis summary
        pattern_count = len([h for h in deduplicated_hits if h["method"] == "pattern_matching"])
        langextract_count = len([h for h in deduplicated_hits if h["method"] == "langextract"])
        both_count = len([h for h in deduplicated_hits if "both" in h.get("method", "")])
        
        print(f"   📊 Final results: {len(deduplicated_hits)} unique hits")
        print(f"      - Pattern matching only: {pattern_count}")
        print(f"      - LangExtract only: {langextract_count}")
        print(f"      - Found by both: {both_count}")
        
        # Calculate confidence statistics
        confidences = [h.get("confidence", 0.0) for h in deduplicated_hits]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        high_confidence_hits = len([h for h in deduplicated_hits if h.get("confidence", 0.0) >= 0.8])
        medium_confidence_hits = len([h for h in deduplicated_hits if 0.5 <= h.get("confidence", 0.0) < 0.8])
        low_confidence_hits = len([h for h in deduplicated_hits if h.get("confidence", 0.0) < 0.5])
        
        return {
            "hits": deduplicated_hits,
            "summary": {
                "total_hits": len(deduplicated_hits),
                "pattern_matching_hits": pattern_count,
                "langextract_hits": langextract_count,
                "both_methods_hits": both_count,
                "text_length": len(text),
                "analysis_method": "hybrid",
                "confidence_stats": {
                    "average_confidence": round(avg_confidence, 3),
                    "high_confidence_hits": high_confidence_hits,
                    "medium_confidence_hits": medium_confidence_hits,
                    "low_confidence_hits": low_confidence_hits
                }
            }
        }
    
    def _deduplicate_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced deduplication using hybrid match strategy."""
        if not hits:
            return []
        
        print(f"   🔄 Advanced deduplication: {len(hits)} hits")
        
        # Group hits by method for better processing
        pattern_hits = [h for h in hits if h["method"] == "pattern_matching"]
        langextract_hits = [h for h in hits if h["method"] == "langextract"]
        
        print(f"      - Pattern matching: {len(pattern_hits)} hits")
        print(f"      - LangExtract: {len(langextract_hits)} hits")
        
        # Start with pattern matching hits (high confidence, exact matches)
        deduplicated = pattern_hits.copy()
        
        # Process LangExtract hits with hybrid matching
        for langextract_hit in langextract_hits:
            merged = False
            
            # Try to find overlapping or similar hits
            for i, existing_hit in enumerate(deduplicated):
                if self._should_merge_hits(existing_hit, langextract_hit):
                    # Merge the hits
                    merged_hit = self._merge_hits(existing_hit, langextract_hit)
                    deduplicated[i] = merged_hit
                    merged = True
                    print(f"      🔗 Merged: '{langextract_hit['matched_text']}' with '{existing_hit['matched_text']}'")
                    break
            
            if not merged:
                # Add as new hit
                deduplicated.append(langextract_hit)
                print(f"      ➕ Added: '{langextract_hit['matched_text']}' (LangExtract only)")
        
        print(f"   ✅ Deduplication complete: {len(deduplicated)} unique hits")
        return deduplicated
    
    def _should_merge_hits(self, hit1: Dict[str, Any], hit2: Dict[str, Any]) -> bool:
        """Determine if two hits should be merged based on hybrid strategy."""
        
        # Same method - don't merge
        if hit1["method"] == hit2["method"]:
            return False
        
        # Strategy-specific merging logic
        if self.hybrid_strategy == "conservative":
            # Only merge exact overlaps
            return self._hits_overlap(hit1, hit2)
        
        elif self.hybrid_strategy == "basic":
            # Merge overlaps and exact text matches
            return (self._hits_overlap(hit1, hit2) or 
                    self._hits_semantically_similar(hit1, hit2))
        
        else:  # advanced
            # Merge overlaps, semantic similarity, and text containment
            return (self._hits_overlap(hit1, hit2) or 
                    self._hits_semantically_similar(hit1, hit2) or
                    self._hits_text_contained(hit1, hit2))
    
    def _hits_overlap(self, hit1: Dict[str, Any], hit2: Dict[str, Any]) -> bool:
        """Check if two hits have overlapping character spans."""
        start1, end1 = hit1.get("char_start", 0), hit1.get("char_end", 0)
        start2, end2 = hit2.get("char_start", 0), hit2.get("char_end", 0)
        
        # Check for any overlap
        return not (end1 <= start2 or end2 <= start1)
    
    def _hits_semantically_similar(self, hit1: Dict[str, Any], hit2: Dict[str, Any]) -> bool:
        """Check if two hits are semantically similar."""
        text1 = hit1["matched_text"].lower().strip()
        text2 = hit2["matched_text"].lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
        
        # High similarity threshold for semantic matching
        similarity = self._text_similarity(text1, text2)
        return similarity > 0.85
    
    def _hits_text_contained(self, hit1: Dict[str, Any], hit2: Dict[str, Any]) -> bool:
        """Check if one hit's text is contained in the other."""
        text1 = hit1["matched_text"].lower().strip()
        text2 = hit2["matched_text"].lower().strip()
        
        return text1 in text2 or text2 in text1
    
    def _merge_hits(self, hit1: Dict[str, Any], hit2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two hits using hybrid strategy."""
        
        # Determine which hit to use as base (prefer pattern matching for exactness)
        base_hit = hit1 if hit1["method"] == "pattern_matching" else hit2
        other_hit = hit2 if hit1["method"] == "pattern_matching" else hit1
        
        # Create merged hit
        merged = base_hit.copy()
        merged["method"] = "both_methods"
        
        # Use the longer/more specific text
        if len(other_hit["matched_text"]) > len(base_hit["matched_text"]):
            merged["matched_text"] = other_hit["matched_text"]
        
        # Use the better suggestion (prefer LangExtract for semantic quality)
        if other_hit["method"] == "langextract" and other_hit.get("suggestion"):
            merged["suggestion"] = other_hit["suggestion"]
        
        # Combine reasons
        reason1 = hit1.get("reason", "")
        reason2 = hit2.get("reason", "")
        if reason1 and reason2 and reason1 != reason2:
            merged["reason"] = f"{reason1} | {reason2}"
        elif reason2:
            merged["reason"] = reason2
        
        # Use higher confidence
        conf1 = hit1.get("confidence", 0.0)
        conf2 = hit2.get("confidence", 0.0)
        merged["confidence"] = max(conf1, conf2)
        
        # Use more specific category if available
        if other_hit.get("category") and other_hit["category"] != "unknown":
            merged["category"] = other_hit["category"]
        
        # Use higher severity
        severity_map = {"low": 1, "medium": 2, "high": 3}
        sev1 = severity_map.get(hit1.get("severity", "medium"), 2)
        sev2 = severity_map.get(hit2.get("severity", "medium"), 2)
        if sev2 > sev1:
            merged["severity"] = hit2["severity"]
        
        # Update span information to cover both hits
        start1, end1 = hit1.get("char_start", 0), hit1.get("char_end", 0)
        start2, end2 = hit2.get("char_start", 0), hit2.get("char_end", 0)
        merged["char_start"] = min(start1, start2)
        merged["char_end"] = max(end1, end2)
        merged["span_length"] = merged["char_end"] - merged["char_start"]
        merged["position"] = merged["char_start"]
        
        # Update grounding method
        merged["span_grounding"] = f"hybrid_{base_hit.get('span_grounding', 'unknown')}_{other_hit.get('span_grounding', 'unknown')}"
        
        return merged
    
    def analyze_hybrid_effectiveness(self, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the effectiveness of the hybrid matching strategy."""
        if not hits:
            return {"error": "No hits to analyze"}
        
        # Count hits by method
        method_counts = {}
        for hit in hits:
            method = hit.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Calculate effectiveness metrics
        total_hits = len(hits)
        pattern_only = method_counts.get("pattern_matching", 0)
        langextract_only = method_counts.get("langextract", 0)
        both_methods = method_counts.get("both_methods", 0)
        
        # Calculate coverage metrics
        pattern_coverage = (pattern_only + both_methods) / total_hits if total_hits > 0 else 0
        langextract_coverage = (langextract_only + both_methods) / total_hits if total_hits > 0 else 0
        overlap_rate = both_methods / total_hits if total_hits > 0 else 0
        
        # Calculate confidence metrics by method
        confidence_by_method = {}
        for method in ["pattern_matching", "langextract", "both_methods"]:
            method_hits = [h for h in hits if h.get("method") == method]
            if method_hits:
                confidences = [h.get("confidence", 0.0) for h in method_hits]
                confidence_by_method[method] = {
                    "count": len(method_hits),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences)
                }
        
        return {
            "total_hits": total_hits,
            "method_distribution": method_counts,
            "coverage_metrics": {
                "pattern_coverage": round(pattern_coverage, 3),
                "langextract_coverage": round(langextract_coverage, 3),
                "overlap_rate": round(overlap_rate, 3)
            },
            "confidence_analysis": confidence_by_method,
            "strategy_effectiveness": {
                "high_overlap": overlap_rate > 0.3,
                "balanced_coverage": abs(pattern_coverage - langextract_coverage) < 0.2,
                "high_confidence": all(
                    conf_data["avg_confidence"] > 0.7 
                    for conf_data in confidence_by_method.values()
                ) if confidence_by_method else False
            }
        }
    
    def filter_by_confidence(self, hits: List[Dict[str, Any]], 
                           threshold: float = None) -> List[Dict[str, Any]]:
        """Filter hits by confidence threshold."""
        if threshold is None:
            threshold = self.confidence_threshold
        
        filtered_hits = [h for h in hits if h.get("confidence", 0.0) >= threshold]
        
        print(f"   🔍 Confidence filtering: {len(hits)} -> {len(filtered_hits)} hits (threshold: {threshold})")
        
        return filtered_hits
    
    def generate_suggestions(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate enhanced suggestions for hits."""
        enhanced_hits = []
        
        for hit in hits:
            # Use LangExtract suggestion if available, otherwise use replacement map
            suggestion = hit.get("suggestion")
            if not suggestion:
                suggestion = self.replacement_map.get(hit["original_key"], "Consider alternative")
            
            # Enhance the reason based on method
            reason = hit.get("reason", "Language analysis")
            if hit["method"] == "langextract":
                reason = f"Semantic analysis: {reason}"
            elif hit["method"] == "pattern_matching":
                reason = f"Pattern matching: {reason}"
            elif hit["method"] == "both_methods":
                reason = f"Both methods: {reason}"
            
            enhanced_hit = hit.copy()
            enhanced_hit.update({
                "suggestion": suggestion,
                "reason": reason,
                "enhanced": True
            })
            enhanced_hits.append(enhanced_hit)
        
        return enhanced_hits

def test_hybrid_system():
    """Test the hybrid system with sample text."""
    print("🧪 Testing Hybrid Language Flagging System")
    print("=" * 50)
    
    # Sample data
    flagged_terms = ["burden", "vulnerable", "disadvantaged", "barriers", "problem", "limited", "inadequate"]
    replacement_map = {
        "burden": "challenge",
        "vulnerable": "underserved",
        "disadvantaged": "underserved",
        "barriers": "challenges",
        "problem": "issue",
        "limited": "constrained",
        "inadequate": "developing"
    }
    skip_terms = ["determinant", "cohort"]
    
    # Sample text
    sample_text = """
    The burden of disease affects vulnerable populations in rural communities. 
    These disadvantaged groups face significant barriers to healthcare access. 
    The problem is compounded by limited resources and inadequate infrastructure. 
    We must address these challenges through community-based interventions that 
    empower local stakeholders and build capacity for sustainable solutions.
    """
    
    # Initialize hybrid system
    hybrid_flagger = HybridLanguageFlagger(
        flagged_terms=flagged_terms,
        replacement_map=replacement_map,
        skip_terms=skip_terms,
        use_langextract=True
    )
    
    # Analyze text
    results = hybrid_flagger.analyze_text(sample_text)
    
    # Generate enhanced suggestions
    enhanced_hits = hybrid_flagger.generate_suggestions(results["hits"])
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"Total hits: {results['summary']['total_hits']}")
    print(f"Pattern matching: {results['summary']['pattern_matching_hits']}")
    print(f"LangExtract: {results['summary']['langextract_hits']}")
    print(f"Both methods: {results['summary']['both_methods_hits']}")
    
    print(f"\n🔍 Detailed Results:")
    for i, hit in enumerate(enhanced_hits, 1):
        print(f"{i:2d}. '{hit['matched_text']}' -> '{hit['suggestion']}'")
        print(f"    Method: {hit['method']}")
        print(f"    Reason: {hit['reason']}")
        print(f"    Context: ...{hit['context'][:50]}...")
        print()
    
    return results

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("LANGEXTRACT_API_KEY"):
        print("⚠️ LANGEXTRACT_API_KEY not found. LangExtract features will be disabled.")
        print("   Set LANGEXTRACT_API_KEY to enable full hybrid functionality.")
    
    test_hybrid_system()

