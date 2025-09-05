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
                 skip_terms: List[str] = None, use_langextract: bool = True):
        self.flagged_terms = flagged_terms
        self.replacement_map = replacement_map
        self.skip_terms = skip_terms or []
        self.use_langextract = use_langextract and LANGEXTRACT_AVAILABLE
        
        # LangExtract examples for different types of problematic language
        self.langextract_examples = self._setup_langextract_examples()
        
        print(f"🔧 Hybrid Language Flagger initialized:")
        print(f"   - Pattern matching: ✅ Enabled")
        print(f"   - LangExtract: {'✅ Enabled' if self.use_langextract else '❌ Disabled'}")
        print(f"   - Flagged terms: {len(self.flagged_terms)}")
        print(f"   - Skip terms: {len(self.skip_terms)}")
    
    def _setup_langextract_examples(self) -> List[Any]:
        """Setup few-shot examples for LangExtract."""
        if not LANGEXTRACT_AVAILABLE:
            return []
        
        return [
            lx.data.ExampleData(
                text="The burden of disease affects vulnerable populations in rural communities",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="burden",
                        attributes={
                            "suggestion": "challenge",
                            "reason": "deficit-framing language",
                            "severity": "medium",
                            "category": "burden_terms"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="vulnerable",
                        attributes={
                            "suggestion": "underserved",
                            "reason": "deficit-framing language",
                            "severity": "high",
                            "category": "vulnerability_terms"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="These disadvantaged groups face significant barriers to healthcare access",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="disadvantaged",
                        attributes={
                            "suggestion": "underserved",
                            "reason": "deficit-framing language",
                            "severity": "high",
                            "category": "vulnerability_terms"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="deficit_language",
                        extraction_text="barriers",
                        attributes={
                            "suggestion": "challenges",
                            "reason": "deficit-framing language",
                            "severity": "medium",
                            "category": "obstacle_terms"
                        }
                    )
                ]
            ),
            lx.data.ExampleData(
                text="The problem is compounded by limited resources and inadequate infrastructure",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="problem_language",
                        extraction_text="problem",
                        attributes={
                            "suggestion": "challenge",
                            "reason": "problem-focused language",
                            "severity": "medium",
                            "category": "problem_terms"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="problem_language",
                        extraction_text="limited",
                        attributes={
                            "suggestion": "constrained",
                            "reason": "deficit-framing language",
                            "severity": "low",
                            "category": "limitation_terms"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="problem_language",
                        extraction_text="inadequate",
                        attributes={
                            "suggestion": "developing",
                            "reason": "deficit-framing language",
                            "severity": "medium",
                            "category": "limitation_terms"
                        }
                    )
                ]
            )
        ]
    
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
                    
                    hits.append({
                        "original_key": term,
                        "matched_text": matched_text,
                        "context": context,
                        "position": pos,
                        "method": "pattern_matching",
                        "suggestion": self.replacement_map.get(term, "Consider alternative"),
                        "reason": "Matched flagged term",
                        "severity": "medium"
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
            Extract instances of deficit-framing language, bias, or non-inclusive terminology.
            Focus on terms that emphasize problems, weaknesses, or negative aspects.
            Use exact text for extractions. Do not paraphrase or overlap entities.
            Provide meaningful attributes including suggestions and reasoning.
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
                # Get context around the extraction
                pos = text.lower().find(extraction.extraction_text.lower())
                if pos != -1:
                    context_start = max(0, pos - 50)
                    context_end = min(len(text), pos + len(extraction.extraction_text) + 50)
                    context = text[context_start:context_end]
                    
                    hits.append({
                        "original_key": extraction.extraction_text,
                        "matched_text": extraction.extraction_text,
                        "context": context,
                        "position": pos,
                        "method": "langextract",
                        "suggestion": extraction.attributes.get("suggestion", "Consider alternative"),
                        "reason": extraction.attributes.get("reason", "Semantic analysis"),
                        "severity": extraction.attributes.get("severity", "medium"),
                        "category": extraction.attributes.get("category", "unknown")
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
        
        return {
            "hits": deduplicated_hits,
            "summary": {
                "total_hits": len(deduplicated_hits),
                "pattern_matching_hits": pattern_count,
                "langextract_hits": langextract_count,
                "both_methods_hits": both_count,
                "text_length": len(text),
                "analysis_method": "hybrid"
            }
        }
    
    def _deduplicate_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate hits based on position and text."""
        seen = set()
        deduplicated = []
        
        for hit in hits:
            # Create a key based on position and text
            key = (hit["position"], hit["matched_text"].lower())
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(hit)
            else:
                # If we've seen this before, mark it as found by both methods
                for existing_hit in deduplicated:
                    if (existing_hit["position"] == hit["position"] and 
                        existing_hit["matched_text"].lower() == hit["matched_text"].lower()):
                        existing_hit["method"] = "both_methods"
                        break
        
        return deduplicated
    
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
