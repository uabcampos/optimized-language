#!/usr/bin/env python3
"""
Simplified LangExtract comparison test
"""

import os
import textwrap
import langextract as lx

def test_langextract_vs_current():
    """Compare LangExtract with our current system."""
    print("🔍 LangExtract vs Current System Comparison")
    print("=" * 60)
    
    # Sample text
    sample_text = """
    The burden of disease affects vulnerable populations in rural communities. 
    These disadvantaged groups face significant barriers to healthcare access. 
    The problem is compounded by limited resources and inadequate infrastructure. 
    We must address these challenges through community-based interventions that 
    empower local stakeholders and build capacity for sustainable solutions.
    """
    
    # Current system terms
    current_terms = [
        "burden", "vulnerable", "disadvantaged", "barriers", "problem", 
        "limited", "inadequate", "challenges"
    ]
    
    print(f"📝 Sample Text:")
    print(f'"{sample_text.strip()}"')
    print()
    
    # Test LangExtract
    print("🤖 LangExtract Analysis:")
    print("-" * 30)
    
    # Simple example for deficit language
    examples = [
        lx.data.ExampleData(
            text="The burden of disease affects vulnerable populations",
            extractions=[
                lx.data.Extraction(
                    extraction_class="deficit_language",
                    extraction_text="burden",
                    attributes={
                        "suggestion": "challenge",
                        "reason": "deficit-framing language",
                        "severity": "medium"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="deficit_language",
                    extraction_text="vulnerable",
                    attributes={
                        "suggestion": "underserved",
                        "reason": "deficit-framing language",
                        "severity": "high"
                    }
                )
            ]
        )
    ]
    
    prompt = """
    Extract instances of deficit-framing language, bias, or non-inclusive terminology.
    Focus on terms that emphasize problems, weaknesses, or negative aspects.
    Use exact text for extractions. Provide suggestions and reasoning.
    """
    
    try:
        result = lx.extract(
            text_or_documents=sample_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash"
        )
        
        langextract_terms = [ext.extraction_text for ext in result.extractions]
        langextract_suggestions = {ext.extraction_text: ext.attributes.get('suggestion', 'N/A') 
                                 for ext in result.extractions}
        
        print(f"✅ Found {len(result.extractions)} extractions:")
        for i, ext in enumerate(result.extractions, 1):
            print(f"  {i:2d}. '{ext.extraction_text}' -> '{ext.attributes.get('suggestion', 'N/A')}'")
            print(f"      Reason: {ext.attributes.get('reason', 'N/A')}")
        
    except Exception as e:
        print(f"❌ LangExtract error: {e}")
        langextract_terms = []
        langextract_suggestions = {}
    
    # Test current system
    print(f"\n🔧 Current System Analysis:")
    print("-" * 30)
    
    current_matches = []
    for term in current_terms:
        if term.lower() in sample_text.lower():
            current_matches.append(term)
    
    print(f"✅ Found {len(current_matches)} matches:")
    for i, term in enumerate(current_matches, 1):
        print(f"  {i:2d}. '{term}'")
    
    # Comparison
    print(f"\n📊 Comparison Results:")
    print("=" * 30)
    
    overlap = set(langextract_terms) & set(current_matches)
    langextract_only = set(langextract_terms) - set(current_matches)
    current_only = set(current_matches) - set(langextract_terms)
    
    print(f"LangExtract found:     {len(langextract_terms):2d} terms")
    print(f"Current system found:  {len(current_matches):2d} terms")
    print(f"Overlap:              {len(overlap):2d} terms")
    print(f"LangExtract only:     {len(langextract_only):2d} terms")
    print(f"Current only:         {len(current_only):2d} terms")
    
    if overlap:
        print(f"\n🤝 Terms found by both systems:")
        for term in sorted(overlap):
            suggestion = langextract_suggestions.get(term, 'N/A')
            print(f"  • '{term}' -> '{suggestion}'")
    
    if langextract_only:
        print(f"\n🆕 Terms found only by LangExtract:")
        for term in sorted(langextract_only):
            suggestion = langextract_suggestions.get(term, 'N/A')
            print(f"  • '{term}' -> '{suggestion}'")
    
    if current_only:
        print(f"\n🔍 Terms found only by current system:")
        for term in sorted(current_only):
            print(f"  • '{term}'")
    
    # Analysis
    print(f"\n📈 Analysis:")
    print("=" * 20)
    
    if len(langextract_terms) > len(current_matches):
        print(f"✅ LangExtract found {len(langextract_terms) - len(current_matches)} more terms")
        print("   This suggests LangExtract has better semantic understanding")
    
    if len(langextract_only) > 0:
        print(f"✅ LangExtract found {len(langextract_only)} additional terms not in our list")
        print("   This could help expand our flagged terms database")
    
    if len(current_only) > 0:
        print(f"⚠️  Current system found {len(current_only)} terms LangExtract missed")
        print("   This suggests our current system has some advantages")
    
    print(f"\n💡 Recommendations:")
    print("=" * 20)
    print("1. LangExtract shows promise for semantic understanding")
    print("2. Consider hybrid approach: LangExtract + current system")
    print("3. Use LangExtract suggestions to improve our replacement map")
    print("4. Test on larger documents to validate performance")

if __name__ == "__main__":
    # Check API key
    if not os.getenv("LANGEXTRACT_API_KEY"):
        print("❌ LANGEXTRACT_API_KEY not found. Please set your API key.")
        print("   You can set it with: export LANGEXTRACT_API_KEY='your-key-here'")
        exit(1)
    
    test_langextract_vs_current()
