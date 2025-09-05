#!/usr/bin/env python3
"""
Example script demonstrating WebEnrichedLLMChain functionality.

Shows how to use web search capabilities to enhance CSV data enrichment
with real-time information from the internet.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import using the package structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice import TableEnricher, FieldManager, EnrichmentConfig
from lattice.chains import WebEnrichedLLMChain
from lattice.utils.logger import setup_logging


def main():
    """Demonstrate web-enriched CSV enrichment functionality."""
    
    # Set up colored logging for better console output
    setup_logging(level="INFO", format_type="console", include_timestamp=False)
    
    print("🌐 Testing Web-Enhanced CSV Enrichment Tool")
    print("=" * 60)
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or openai_key == "your-api-key-here":
        print("⚠️  Warning: OpenAI API key not set.")
        print("   Set OPENAI_API_KEY environment variable to test with real LLM.")
        return
        
    if not tavily_key:
        print("⚠️  Warning: Tavily API key not set.")
        print("   Set TAVILY_API_KEY environment variable to enable web search.")
        print("   Get a free API key at: https://tavily.com")
        return
    
    try:
        print("🔧 Initializing components...")
        
        # Load field manager with web intelligence fields
        field_manager = FieldManager.from_csv("examples/field_categories.csv")
        print("✅ Field manager loaded with categories:", field_manager.get_categories())
        
        # Create web-enriched chain with optimal settings
        web_chain = WebEnrichedLLMChain.create(
            api_key=openai_key,
            tavily_api_key=tavily_key
            # Uses defaults: gpt-4o, 8000 tokens, temperature=0.3, 5 search results, advanced depth
        )
        print("✅ Web-enriched LLM chain created successfully")
        
        # Create enrichment config optimized for web searches
        config = EnrichmentConfig.for_development()
        config.row_delay = 3.0  # Longer delay to respect rate limits
        print("✅ Configuration created (3s delay between rows)")
        
        # Create table enricher
        enricher = TableEnricher(
            chain=web_chain,
            field_manager=field_manager,
            config=config
        )
        print("✅ Table enricher initialized")
        
        # Load sample data
        df = pd.read_csv("examples/sample_data.csv")
        print(f"📊 Loaded {len(df)} rows of sample data")
        print("\nSample data preview:")
        print(df.head())
        
        # Demo 1: Traditional business analysis (no web search needed)
        print("\n" + "="*60)
        print("📈 Demo 1: Traditional Business Analysis")
        print("="*60)
        
        # Create a regular chain for comparison
        from lattice.chains import LLMChain
        regular_chain = LLMChain.openai(api_key=openai_key, temperature=0.3)
        regular_enricher = TableEnricher(
            chain=regular_chain,
            field_manager=field_manager,
            config=config
        )
        
        traditional_result = regular_enricher.enrich_dataframe(
            df=df.head(2),  # Just first 2 rows for demo
            category="business_analysis",
            overwrite_fields=True
        )
        
        print("Traditional Analysis Results:")
        print(traditional_result[["name", "market_size", "competition_level", "growth_potential"]])
        
        # Demo 2: Web-enhanced intelligence gathering
        print("\n" + "="*60)
        print("🌐 Demo 2: Web-Enhanced Intelligence Gathering")
        print("="*60)
        
        web_enhanced_result = enricher.enrich_dataframe(
            df=df.head(2),  # Just first 2 rows for demo
            category="web_intelligence",
            overwrite_fields=True
        )
        
        print("Web-Enhanced Results:")
        print(web_enhanced_result[["name", "recent_funding", "current_news", "company_valuation"]])
        
        # Save results
        output_file = "web_enriched_data/sample_data.csv"
        
        # Combine both traditional and web-enhanced results
        combined_df = df.head(2).copy()
        
        # Add traditional analysis columns
        for col in ["market_size", "competition_level", "growth_potential"]:
            if col in traditional_result.columns:
                combined_df[f"analysis_{col}"] = traditional_result[col]
        
        # Add web intelligence columns  
        for col in ["recent_funding", "current_news", "company_valuation", "market_trends"]:
            if col in web_enhanced_result.columns:
                combined_df[f"web_{col}"] = web_enhanced_result[col]
        
        combined_df.to_csv(output_file, index=False)
        print(f"\n💾 Combined results saved to: {output_file}")
        
        # Show detailed comparison
        print("\n" + "="*60)
        print("🔍 Detailed Comparison: Traditional vs Web-Enhanced")
        print("="*60)
        
        for idx, row in df.head(2).iterrows():
            company_name = row['name']
            print(f"\n🏢 Company: {company_name}")
            print("-" * 40)
            
            # Traditional analysis
            if idx < len(traditional_result):
                trad_row = traditional_result.iloc[idx]
                print("📊 Traditional Analysis:")
                print(f"   Market Size: {trad_row.get('market_size', 'N/A')}")
                print(f"   Competition: {trad_row.get('competition_level', 'N/A')}")
            
            # Web-enhanced intelligence
            if idx < len(web_enhanced_result):
                web_row = web_enhanced_result.iloc[idx]
                print("🌐 Web-Enhanced Intelligence:")
                print(f"   Recent Funding: {web_row.get('recent_funding', 'N/A')}")
                print(f"   Latest News: {web_row.get('current_news', 'N/A')}")
                print(f"   Valuation: {web_row.get('company_valuation', 'N/A')}")
        
        print("\n✅ Web enrichment demonstration completed successfully!")
        print("\n💡 Key Benefits of Web Enhancement:")
        print("   • Real-time funding information")
        print("   • Current news and developments") 
        print("   • Latest market valuations")
        print("   • Up-to-date executive team info")
        print("   • Current market trends and dynamics")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """Quick test of web search functionality without full enrichment."""
    
    # Set up logging for quick test
    setup_logging(level="INFO", format_type="console", include_timestamp=False)
    
    print("🧪 Quick Web Search Test")
    print("=" * 30)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("⚠️  API keys not available for quick test")
        return
    
    try:
        # Create web-enriched chain
        chain = WebEnrichedLLMChain.create(
            api_key=openai_key,
            tavily_api_key=tavily_key,
            max_search_results=2
        )
        
        # Test search query generation
        test_data = {
            "row_data": {"name": "OpenAI", "industry": "AI"},
            "fields": {"funding": {"prompt": "latest funding news"}}
        }
        
        queries = chain._build_search_queries(test_data)
        print(f"Generated queries: {queries}")
        
        # Test web search (this will hit the actual API)
        search_result = chain._perform_web_search(test_data)
        print(f"Search results preview: {search_result[:200]}...")
        
        print("✅ Quick test passed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()