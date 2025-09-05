#!/usr/bin/env python3
"""
Test script for CSV enrichment functionality using the new clean API.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import using the new package structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice import TableEnricher, LLMChain, FieldManager, EnrichmentConfig


def main():
    """Test the CSV enrichment functionality with the new API."""
    
    # Set up API key (you'll need to set this)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("⚠️  Warning: OpenAI API key not set. Using mock responses.")
        print("   Set OPENAI_API_KEY environment variable to test with real LLM.")
    
    print("🚀 Testing CSV Enrichment Tool (New Architecture)")
    print("=" * 60)
    
    # Initialize components with new clean API
    try:
        # Load field manager using new API (files now in examples/)
        field_manager = FieldManager.from_csv("../examples/field_categories.csv")
        print("✅ Field manager loaded successfully")
        
        # Create enrichment chain using factory method with proper max_tokens
        chain = LLMChain.openai(api_key=api_key, temperature=0.5, max_tokens=4000)
        print("✅ LLM chain created successfully")
        
        # Create enrichment config using optimized preset
        config = EnrichmentConfig.for_development()  # Optimized for testing
        print("✅ Configuration created successfully")
        
        # Create table enricher with new simplified API
        enricher = TableEnricher(
            chain=chain,
            field_manager=field_manager,
            config=config
        )
        print("✅ Table enricher initialized successfully")
        print(f"📊 Enricher info: {enricher.get_enrichment_info()['field_manager_info']}")
        
        # Load sample data (now in examples/)
        df = pd.read_csv("../examples/sample_data.csv")
        print(f"✅ Loaded {len(df)} rows of sample data")
        print("\nSample data:")
        print(df.head())
        
        # Process the business_analysis category using new API
        print("\n🔄 Processing business_analysis category...")
        enriched_df = enricher.enrich_dataframe(
            df=df,
            category="business_analysis",
            overwrite_fields=True
        )
        
        print("\n✅ Enrichment completed!")
        print("\nEnriched data:")
        print(enriched_df)
        
        # Save results
        output_file = "enriched_sample_data.csv"
        enriched_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Show enrichment summary
        info = enricher.get_enrichment_info()
        print(f"\n📈 Enrichment Summary:")
        print(f"   Chain Type: {info['chain_type']}")
        print(f"   Categories Available: {', '.join(info['categories'])}")
        print(f"   Total Fields: {info['total_fields']}")
        print(f"   Configuration: {info['config']['batch_size']} batch size, "
              f"{info['config']['max_workers']} workers, "
              f"{info['config']['row_delay']}s delay")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()