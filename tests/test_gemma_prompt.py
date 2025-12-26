"""Test script for GemmaExtractor prompt with a specific dish description."""

import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.gemma_extractor import GemmaExtractor

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Single test case - Bánh Cuốn
TEST_DESCRIPTION = """[Rice flour batter] 200 g rice flour ,40 g tapioca starch ,40 g potato starch ,1320 g filtered water ,1 tsp salt ,3 tbsp neutral cooking oil for wrapping cooked bánh cuốn | [Filling] 15 g dried woodear mushroom ,filtered hot water ,2 tbsp neutral cooking oil ,127 g onion peeled and finely chopped ,300 g ground pork ,1 tsp salt ,1/2 tsp MSG (monosodium glutamate) ,1 tsp ground pepper ,172 g jicama peeled and finely chopped ,29 g green onions optional | [Sides] crunchy Japanese or Persian cucumbers ,bean sprouts blanched ,fried shallots ,cha lua (Vietnamese pork sausage) ,mint chopped ,fish sauce-based dipping sauce nước chấm"""


def main():
    """Run single test."""
    print("🚀 Testing GemmaExtractor with Bánh Cuốn description")
    print("=" * 80)
    
    print("\n📝 Input description:")
    print(TEST_DESCRIPTION[:200] + "...")
    
    # Initialize extractor
    print("\n⏳ Initializing GemmaExtractor...")
    extractor = GemmaExtractor(preload=True)
    print("✅ Model loaded successfully!")
    
    # Extract
    print("\n🔍 Extracting...")
    result = extractor.extract_from_description(TEST_DESCRIPTION)
    
    # Print results
    print("\n" + "=" * 80)
    print("� EXTRACTION RESULT:")
    print("=" * 80)
    print(f"\n🍽️  Dish Name:   {result.get('dish_name')}")
    print(f"🌍 Cuisine:     {result.get('cuisine')}")
    print(f"📊 Confidence:  {result.get('confidence')}")
    print(f"\n🥗 Ingredients ({len(result.get('ingredients', []))} items):")
    for i, ing in enumerate(result.get('ingredients', []), 1):
        print(f"   {i:2}. {ing}")
    
    # Save to JSON
    output_file = "tests/prompt_test_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Result saved to: {output_file}")


if __name__ == "__main__":
    main()
