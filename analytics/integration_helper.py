"""
Analytics Integration Helper
Updates your existing agents to use the new 4-tier analytics framework
"""

import os
import sys
from pathlib import Path


class AnalyticsIntegrator:
    """Helper class to integrate analytics into existing agents"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.agents_dir = self.project_root / "agents"
        self.analytics_dir = self.project_root / "analytics"
    
    def update_base_agent_imports(self):
        """Update base agent imports to use new analytics"""
        base_agent_path = self.agents_dir / "base.py"
        
        # Read current content
        with open(base_agent_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace old analytics imports
        old_import = """# Analytics imports
try:
    from analytics.base_analytics import (
        DescriptiveAnalytics, DiagnosticAnalytics, 
        PredictiveAnalytics, PrescriptiveAnalytics,
        AnalyticsResult
    )
except ImportError:
    from ..analytics.base_analytics import (
        DescriptiveAnalytics, DiagnosticAnalytics, 
        PredictiveAnalytics, PrescriptiveAnalytics,
        AnalyticsResult
    )"""
        
        new_import = """# Analytics imports
try:
    from analytics import (
        DescriptiveAnalytics, DiagnosticAnalytics, 
        PredictiveAnalytics, PrescriptiveAnalytics,
        AnalyticsResponse, AnalyticsPromptTemplates
    )
except ImportError:
    from ..analytics import (
        DescriptiveAnalytics, DiagnosticAnalytics, 
        PredictiveAnalytics, PrescriptiveAnalytics,
        AnalyticsResponse, AnalyticsPromptTemplates
    )"""
        
        # Replace in content
        content = content.replace(old_import, new_import)
        
        # Update AnalyticsResult to AnalyticsResponse
        content = content.replace("AnalyticsResult", "AnalyticsResponse")
        
        # Update system prompt method to use new prompt templates
        old_prompt_method = """    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        \"\"\"Build domain-specific system prompt for the agent\"\"\"
        if context is None:
            context = {}
            
        return f\"\"\"
You are a {self.name} specialized in {self.agent_type.value} analysis for a Tally ERP system.

Your core responsibilities:
- Analyze {self.agent_type.value} data from Tally database
- Provide accurate, actionable business insights
- Access data from tables: {', '.join(self.tables)}
- Execute 4-tier analytics: Descriptive, Diagnostic, Predictive, Prescriptive
- Ensure data accuracy and business context

Database Context:
- Tally ERP database with master (mst_*) and transaction (trn_*) tables
- GUID-based primary keys throughout
- Date fields in datetime format
- Amount fields as decimal/float
- Real business data with accounting conventions

Current Analysis Context: {json.dumps(context, default=str, indent=2)}

Business Guidelines:
- Always validate date ranges and parameters
- Provide business context for technical results  
- Highlight actionable insights and recommendations
- Format currency amounts in INR (₹) format
- Handle errors gracefully with helpful messages
- Focus on practical business value

Analytics Capabilities:
- Descriptive: Historical summaries and KPIs
- Diagnostic: Root cause analysis and pattern identification  
- Predictive: Trend forecasting and future projections
- Prescriptive: Strategic recommendations and optimization

Remember: You're providing business intelligence to help make informed decisions.
\"\"\""""

        new_prompt_method = """    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        \"\"\"Build domain-specific system prompt for the agent\"\"\"
        return AnalyticsPromptTemplates.get_prompt_for_agent_type(
            self.agent_type.value, context
        )"""
        
        # Replace the method
        content = content.replace(old_prompt_method, new_prompt_method)
        
        # Write back the updated content
        with open(base_agent_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Updated {base_agent_path}")
    
    def create_domain_specific_agents(self):
        """Create domain-specific analytics agent classes"""
        
        # Sales Agent Enhancement
        sales_agent_content = '''"""
Enhanced Sales Analytics Agent
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from .base import BaseAgent
from ..models.responses import QueryType, StructuredResponse
from ..analytics import (
    SalesDescriptiveAnalytics, 
    SalesDiagnosticAnalytics,
    SalesPredictiveAnalytics, 
    SalesPrescriptiveAnalytics
)


class SalesAnalyticsAgent(BaseAgent):
    """Sales-focused analytics agent with 4-tier intelligence"""
    
    def __init__(self):
        super().__init__(
            name="SalesAnalytics",
            agent_type=QueryType.SALES,
            tables=[
                "trn_sales", "trn_voucher", "trn_accounting",
                "mst_ledger", "mst_party", "mst_stock_item"
            ]
        )
        
        # Initialize sales-specific analytics
        self.sales_descriptive = SalesDescriptiveAnalytics()
        self.sales_diagnostic = SalesDiagnosticAnalytics() 
        self.sales_predictive = SalesPredictiveAnalytics()
        self.sales_prescriptive = SalesPrescriptiveAnalytics()
    
    def analyze_sales_performance(
        self, 
        query: str,
        date_from: str = None,
        date_to: str = None,
        customer_name: str = None,
        product_category: str = None
    ) -> StructuredResponse:
        """Comprehensive sales performance analysis"""
        try:
            # Get sales data
            data = self._get_sales_data(date_from, date_to, customer_name, product_category)
            
            # Determine analytics type based on query
            analytics_type = self._classify_query_intent(query)
            params = {
                'date_from': date_from,
                'date_to': date_to,
                'customer_name': customer_name,
                'product_category': product_category
            }
            
            # Execute appropriate analytics
            if analytics_type == 'descriptive':
                result = self.sales_descriptive.analyze(query, data, params)
            elif analytics_type == 'diagnostic':
                result = self.sales_diagnostic.analyze(query, data, params)
            elif analytics_type == 'predictive':
                result = self.sales_predictive.analyze(query, data, params)
            elif analytics_type == 'prescriptive':
                result = self.sales_prescriptive.analyze(query, data, params)
            else:
                result = self.sales_descriptive.analyze(query, data, params)
            
            return StructuredResponse(
                query_type=QueryType.SALES,
                status="success",
                data=result.results,
                insights=result.insights,
                recommendations=result.recommendations,
                metadata=result.metadata
            )
            
        except Exception as e:
            return StructuredResponse(
                query_type=QueryType.SALES,
                status="error", 
                error=f"Sales analysis failed: {str(e)}"
            )
    
    def _get_sales_data(self, date_from=None, date_to=None, customer_name=None, product_category=None):
        """Get sales data for analysis"""
        # Build query conditions
        conditions = []
        if date_from:
            conditions.append(f"v.date >= '{date_from}'")
        if date_to:
            conditions.append(f"v.date <= '{date_to}'")
        if customer_name:
            conditions.append(f"p.name LIKE '%{customer_name}%'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT 
            v.date,
            v.voucher_number,
            v.voucher_type,
            p.name as party_name,
            si.name as item,
            si.parent as item_group,
            a.amount,
            COALESCE(inv.quantity, 0) as quantity,
            COALESCE(inv.rate, 0) as rate
        FROM trn_voucher v
        JOIN trn_accounting a ON v.guid = a.guid
        LEFT JOIN mst_party p ON a.ledger = p.name
        LEFT JOIN trn_inventory inv ON v.guid = inv.guid
        LEFT JOIN mst_stock_item si ON inv.item = si.name
        WHERE v.voucher_type IN ('Sales', 'Sales Invoice') 
        AND a.amount > 0
        AND {where_clause}
        ORDER BY v.date DESC
        """
        
        result = self.execute_database_query(query)
        return pd.DataFrame(result) if result else pd.DataFrame()
'''

        # Write sales agent
        sales_agent_path = self.agents_dir / "sales_analytics.py"
        with open(sales_agent_path, 'w', encoding='utf-8') as f:
            f.write(sales_agent_content)
        
        print(f"✅ Created {sales_agent_path}")
    
    def install_requirements(self):
        """Install additional analytics requirements"""
        import subprocess
        
        requirements = [
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
            "statsmodels>=0.14.0",
            "prophet>=1.1.4",
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0"
        ]
        
        print("Installing analytics requirements...")
        for req in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                print(f"✅ Installed {req}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to install {req}: {e}")
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Starting Analytics Integration...")
        
        # 1. Update base agent
        print("\n1. Updating base agent imports...")
        self.update_base_agent_imports()
        
        # 2. Create domain-specific agents
        print("\n2. Creating domain-specific analytics agents...")
        self.create_domain_specific_agents()
        
        # 3. Install requirements
        print("\n3. Installing additional requirements...")
        # self.install_requirements()  # Uncomment to auto-install
        
        print("\n✅ Analytics Integration Complete!")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r analytics_requirements.txt")
        print("2. Update your manager to use new analytics agents")
        print("3. Test the new analytics capabilities")


if __name__ == "__main__":
    # Run integration
    project_root = os.path.dirname(os.path.dirname(__file__))
    integrator = AnalyticsIntegrator(project_root)
    integrator.run_integration()
