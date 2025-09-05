#!/usr/bin/env python3
"""
Simple LangExtract visualization test
"""

import os
import textwrap
import langextract as lx

def create_simple_visualization():
    """Create a simple visualization of LangExtract results."""
    print("🎨 Creating LangExtract Visualization Demo")
    print("=" * 50)
    
    # Sample text
    sample_text = "The burden of disease affects vulnerable populations in rural communities."
    
    # Example
    examples = [
        lx.data.ExampleData(
            text="The burden of disease affects vulnerable populations",
            extractions=[
                lx.data.Extraction(
                    extraction_class="deficit_language",
                    extraction_text="burden",
                    attributes={"suggestion": "challenge", "reason": "deficit-framing"}
                ),
                lx.data.Extraction(
                    extraction_class="deficit_language",
                    extraction_text="vulnerable",
                    attributes={"suggestion": "underserved", "reason": "deficit-framing"}
                )
            ]
        )
    ]
    
    prompt = "Extract deficit-framing language from the text."
    
    try:
        # Extract
        result = lx.extract(
            text_or_documents=sample_text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemini-2.5-flash"
        )
        
        print(f"✅ Extracted {len(result.extractions)} terms from: '{sample_text}'")
        
        # Save results
        lx.io.save_annotated_documents([result], output_name="demo_extractions.jsonl")
        print("✅ Saved extractions to demo_extractions.jsonl")
        
        # Generate HTML visualization
        html_content = lx.visualize("demo_extractions.jsonl")
        
        with open("langextract_demo.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Created langextract_demo.html")
        print("🌐 Open langextract_demo.html in your browser to see the interactive visualization!")
        
        # Clean up
        if os.path.exists("demo_extractions.jsonl"):
            os.remove("demo_extractions.jsonl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False

if __name__ == "__main__":
    if not os.getenv("LANGEXTRACT_API_KEY"):
        print("❌ LANGEXTRACT_API_KEY not found. Please set your API key.")
        exit(1)
    
    create_simple_visualization()
