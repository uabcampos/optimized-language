#!/usr/bin/env python3
"""
Test module for LangExtract integration with our language flagging app
This is a proof-of-concept to explore how LangExtract could enhance our current system
"""

import os
import json
import textwrap
from typing import List, Dict, Any
import langextract as lx

# Sample text from our flagged terms for testing
SAMPLE_TEXT = """
The burden of disease affects vulnerable populations in rural communities. 
These disadvantaged groups face significant barriers to healthcare access. 
The problem is compounded by limited resources and inadequate infrastructure. 
We must address these challenges through community-based interventions that 
empower local stakeholders and build capacity for sustainable solutions.
"""

# Our current flagged terms for comparison
CURRENT_FLAGGED_TERMS = [
    "burden", "vulnerable", "disadvantaged", "barriers", "problem", 
    "limited", "inadequate", "challenges"
]

class LangExtractTester:
    """Test LangExtract integration for language flagging."""
    
    def __init__(self):
        self.setup_examples()
    
    def setup_examples(self):
        """Setup few-shot examples for different types of problematic language."""
        
        # Deficit-framing language examples
        self.deficit_examples = [
            lx.data.ExampleData(
                text="The burden of disease affects vulnerable populations",
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
                text="These disadvantaged groups face significant barriers",
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
            )
        ]
        
        # Problem-focused language examples
        self.problem_examples = [
            lx.data.ExampleData(
                text="The problem is compounded by limited resources",
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
                    )
                ]
            )
        ]
    
    def test_deficit_language_extraction(self, text: str) -> Dict[str, Any]:
        """Test extraction of deficit-framing language."""
        prompt = textwrap.dedent("""
        Extract instances of deficit-framing language, bias, or non-inclusive terminology.
        Focus on terms that emphasize problems, weaknesses, or negative aspects.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes including suggestions and reasoning.
        """)
        
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=self.deficit_examples,
                model_id="gemini-2.5-flash"  # Using flash for faster testing
            )
            return {
                "success": True,
                "result": result,
                "extractions": result.extractions if hasattr(result, 'extractions') else []
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extractions": []
            }
    
    def test_problem_language_extraction(self, text: str) -> Dict[str, Any]:
        """Test extraction of problem-focused language."""
        prompt = textwrap.dedent("""
        Extract instances of problem-focused language that emphasizes challenges or limitations.
        Look for terms that frame situations negatively or focus on what's wrong.
        Use exact text for extractions. Provide suggestions for more constructive alternatives.
        """)
        
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=self.problem_examples,
                model_id="gemini-2.5-flash"
            )
            return {
                "success": True,
                "result": result,
                "extractions": result.extractions if hasattr(result, 'extractions') else []
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extractions": []
            }
    
    def compare_with_current_system(self, text: str) -> Dict[str, Any]:
        """Compare LangExtract results with our current system."""
        print("🔍 Testing LangExtract vs Current System")
        print("=" * 50)
        
        # Test LangExtract
        print("\n📊 LangExtract Results:")
        deficit_result = self.test_deficit_language_extraction(text)
        problem_result = self.test_problem_language_extraction(text)
        
        # Current system simulation (simple pattern matching)
        print("\n🔧 Current System Results:")
        current_matches = []
        for term in CURRENT_FLAGGED_TERMS:
            if term.lower() in text.lower():
                current_matches.append({
                    "term": term,
                    "found": True,
                    "position": text.lower().find(term.lower())
                })
        
        # Combine results
        all_langextract_extractions = []
        if deficit_result["success"]:
            all_langextract_extractions.extend(deficit_result["extractions"])
        if problem_result["success"]:
            all_langextract_extractions.extend(problem_result["extractions"])
        
        # Analysis
        langextract_terms = [ext.extraction_text for ext in all_langextract_extractions]
        current_terms = [match["term"] for match in current_matches]
        
        print(f"\n📈 Comparison Summary:")
        print(f"Current system found: {len(current_matches)} terms")
        print(f"LangExtract found: {len(all_langextract_extractions)} extractions")
        print(f"Overlap: {len(set(langextract_terms) & set(current_terms))} terms")
        print(f"LangExtract only: {len(set(langextract_terms) - set(current_terms))} terms")
        print(f"Current only: {len(set(current_terms) - set(langextract_terms))} terms")
        
        return {
            "langextract_results": {
                "deficit": deficit_result,
                "problem": problem_result,
                "all_extractions": all_langextract_extractions
            },
            "current_system_results": current_matches,
            "comparison": {
                "langextract_count": len(all_langextract_extractions),
                "current_count": len(current_matches),
                "overlap": len(set(langextract_terms) & set(current_terms)),
                "langextract_only": len(set(langextract_terms) - set(current_terms)),
                "current_only": len(set(current_terms) - set(langextract_terms))
            }
        }
    
    def create_visualization_demo(self, text: str, output_file: str = "langextract_demo.html"):
        """Create an interactive visualization of LangExtract results."""
        print(f"\n🎨 Creating visualization demo: {output_file}")
        
        # Get extractions
        deficit_result = self.test_deficit_language_extraction(text)
        problem_result = self.test_problem_language_extraction(text)
        
        if not deficit_result["success"] and not problem_result["success"]:
            print("❌ No successful extractions to visualize")
            return None
        
        # Combine results
        all_extractions = []
        if deficit_result["success"]:
            all_extractions.extend(deficit_result["extractions"])
        if problem_result["success"]:
            all_extractions.extend(problem_result["extractions"])
        
        if not all_extractions:
            print("❌ No extractions found")
            return None
        
        # Create a simple document for visualization
        doc = lx.data.AnnotatedDocument(
            text=text,
            extractions=all_extractions
        )
        
        try:
            # Save to JSONL
            lx.io.save_annotated_documents([doc], output_name="temp_extractions.jsonl")
            
            # Generate HTML visualization
            html_content = lx.visualize("temp_extractions.jsonl")
            
            # Save to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Clean up temp file
            if os.path.exists("temp_extractions.jsonl"):
                os.remove("temp_extractions.jsonl")
            
            print(f"✅ Visualization saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None

def main():
    """Run the LangExtract integration test."""
    print("🚀 LangExtract Integration Test")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv("LANGEXTRACT_API_KEY"):
        print("❌ LANGEXTRACT_API_KEY not found. Please set your API key.")
        print("   You can set it with: export LANGEXTRACT_API_KEY='your-key-here'")
        return
    
    # Initialize tester
    tester = LangExtractTester()
    
    # Test with sample text
    print(f"\n📝 Sample Text:")
    print(f'"{SAMPLE_TEXT.strip()}"')
    
    # Run comparison
    results = tester.compare_with_current_system(SAMPLE_TEXT)
    
    # Display detailed results
    print(f"\n📊 Detailed LangExtract Results:")
    for i, extraction in enumerate(results["langextract_results"]["all_extractions"], 1):
        print(f"{i}. '{extraction.extraction_text}' -> '{extraction.attributes.get('suggestion', 'N/A')}'")
        print(f"   Reason: {extraction.attributes.get('reason', 'N/A')}")
        print(f"   Category: {extraction.attributes.get('category', 'N/A')}")
        print()
    
    # Create visualization
    viz_file = tester.create_visualization_demo(SAMPLE_TEXT)
    if viz_file:
        print(f"🌐 Open {viz_file} in your browser to see the interactive visualization!")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"✅ LangExtract successfully extracted {results['comparison']['langextract_count']} instances")
    print(f"✅ Current system found {results['comparison']['current_count']} terms")
    print(f"✅ Overlap: {results['comparison']['overlap']} terms found by both")
    print(f"✅ LangExtract found {results['comparison']['langextract_only']} additional instances")
    print(f"✅ Current system found {results['comparison']['current_only']} terms not caught by LangExtract")

if __name__ == "__main__":
    main()
