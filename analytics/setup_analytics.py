"""
Complete Analytics Setup Script
Run this to set up your 4-tier analytics framework completely
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure the parent directory is in sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_setup():
    """Complete setup process"""
    
    print("🚀 COMPLETE ANALYTICS FRAMEWORK SETUP")
    print("=" * 50)
    
    # 1. Install requirements
    print("\n📦 Step 1: Installing Required Packages")
    print("-" * 30)
    
    requirements = [
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0", 
        "statsmodels>=0.14.0",
        # Optional but recommended
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ]
    
    print("Installing core ML/analytics packages...")
    for req in requirements:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {req} - Installation failed: {e}")
    
    # Optional packages
    optional_requirements = [
        "prophet>=1.1.4",
        "xgboost>=2.0.0", 
        "lightgbm>=4.0.0",
        "plotly>=5.15.0"
    ]
    
    print("\nInstalling optional advanced packages...")
    for req in optional_requirements:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {req} - May require additional system dependencies")
    
    # 2. Test the framework
    print("\n🧪 Step 2: Testing Analytics Framework")
    print("-" * 30)
    
    try:
        # Import test
        from analytics import (
            DescriptiveAnalytics, DiagnosticAnalytics,
            PredictiveAnalytics, PrescriptiveAnalytics,
            AnalyticsPromptTemplates
        )
        print("✅ All analytics classes import successfully")
        
        # Quick test
        descriptive = DescriptiveAnalytics("financial")
        queries = descriptive.get_supported_queries()
        print(f"✅ Descriptive analytics supports {len(queries)} query types")
        
        # Test prompts
        prompt = AnalyticsPromptTemplates.get_financial_analytics_prompt()
        print(f"✅ System prompts working (length: {len(prompt)} chars)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    # 3. Integration instructions
    print("\n🔧 Step 3: Integration Instructions")
    print("-" * 30)
    
    print("""
✅ Your analytics framework is set up! Here's what to do next:

🔹 UPDATE YOUR AGENTS:
   1. Update agent imports to use new analytics classes
   2. Replace old analytics calls with new 4-tier system
   3. Use AnalyticsPromptTemplates for system prompts

🔹 DOMAIN-SPECIFIC ANALYTICS:
   • Sales: SalesDescriptiveAnalytics, SalesDiagnosticAnalytics, etc.
   • Inventory: InventoryDescriptiveAnalytics, etc.
   • Purchase: PurchaseDescriptiveAnalytics, etc.

🔹 EXAMPLE USAGE:
   ```python
   from analytics import SalesDescriptiveAnalytics
   
   sales_analytics = SalesDescriptiveAnalytics()
   result = sales_analytics.analyze(
       "Show sales performance", 
       sales_data, 
       {'period': 'monthly'}
   )
   print(result.insights)
   print(result.recommendations)
   ```

🔹 SYSTEM PROMPTS:
   ```python
   from analytics import AnalyticsPromptTemplates
   
   prompt = AnalyticsPromptTemplates.get_sales_analytics_prompt(context)
   ```

🔹 RUN TESTS:
   python analytics/test_analytics.py

🔹 INTEGRATION HELPER:
   python analytics/integration_helper.py
""")
    
    return True


if __name__ == "__main__":
    success = run_setup()
    if success:
        print("\n🎉 SETUP COMPLETE! Your 4-tier analytics framework is ready!")
    else:
        print("\n❌ Setup encountered issues. Check the errors above.")
