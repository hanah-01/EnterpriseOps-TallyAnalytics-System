"""
Enhanced Manager Agent with Analytics Integration
Bridges your existing Google ADK agents with new analytics-enhanced agents
"""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from typing import Dict, List, Any, Optional
import logging

# Your existing sub-agents
from .sub_agents.financial_analyst.agent import financial_analyst
from .sub_agents.inventory_analyst.agent import inventory_analyst
from .sub_agents.cost_management_analyst.agent import cost_management_analyst
from .sub_agents.tax_and_compliance_analyst.agent import tax_and_compliance_analyst
from .sub_agents.banking_analyst.agent import banking_analyst
from .tools.tools import get_current_time_tool

# New enhanced agents with analytics
try:
    from agents.financial import FinancialAgent
    from models.responses import AnalysisResponse, AgentCapability, QueryType
    from analytics.base_analytics import AnalyticsType
    ENHANCED_AGENTS_AVAILABLE = True
except ImportError:
    try:
        from multiagents.agents.financial import FinancialAgent
        from multiagents.models.responses import AnalysisResponse, AgentCapability, QueryType
        from multiagents.analytics.base_analytics import AnalyticsType
        ENHANCED_AGENTS_AVAILABLE = True
    except ImportError:
        ENHANCED_AGENTS_AVAILABLE = False
        logging.warning("Enhanced agents not available, using existing agents only")

class EnhancedManagerAgent:
    """Enhanced manager with both ADK agents and analytics agents"""
    
    def __init__(self):
        self.adk_agents = {
            'financial_analyst': financial_analyst,
            'inventory_analyst': inventory_analyst,
            'cost_management_analyst': cost_management_analyst,
            'tax_and_compliance_analyst': tax_and_compliance_analyst,
            'banking_analyst': banking_analyst
        }
        
        # Initialize enhanced agents if available
        self.enhanced_agents = {}
        if ENHANCED_AGENTS_AVAILABLE:
            try:
                self.enhanced_agents['financial'] = FinancialAgent()
                logging.info("Enhanced Financial Agent initialized")
            except Exception as e:
                logging.error(f"Failed to initialize enhanced agents: {e}")
    
    def get_analytics(self, domain: str, analytics_type: str, query: str, **kwargs) -> Dict[str, Any]:
        """Get analytics from enhanced agents"""
        if domain in self.enhanced_agents:
            try:
                agent = self.enhanced_agents[domain]
                result = agent.get_analytics(analytics_type, query, **kwargs)
                return {
                    "status": "success",
                    "agent": f"enhanced_{domain}",
                    "analytics_type": analytics_type,
                    "result": result.dict()
                }
            except Exception as e:
                return {
                    "status": "error",
                    "agent": f"enhanced_{domain}",
                    "error": str(e)
                }
        else:
            return {
                "status": "error",
                "error": f"Enhanced {domain} agent not available"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all agents"""
        status = {
            "adk_agents": {},
            "enhanced_agents": {},
            "overall_status": "healthy"
        }
        
        # Check ADK agents (basic check)
        for name, agent in self.adk_agents.items():
            status["adk_agents"][name] = "available"
        
        # Check enhanced agents
        for name, agent in self.enhanced_agents.items():
            try:
                health = agent.health_check()
                status["enhanced_agents"][name] = health
            except Exception as e:
                status["enhanced_agents"][name] = {"status": "error", "error": str(e)}
                status["overall_status"] = "degraded"
        
        return status

# Create enhanced manager instance
enhanced_manager = EnhancedManagerAgent()

# Your existing root agent with enhanced capabilities
root_agent = Agent(
    name="enhanced_manager",
    model="gemini-2.0-flash", 
    description="Enhanced Manager agent with analytics capabilities",
    instruction=f"""
    You are an enhanced manager agent with both traditional business intelligence and advanced 4-tier analytics capabilities.

    Database Schema Overview:
    The tally.db database contains master tables (mst_*) and transaction tables (trn_*):
    
    Master Tables:
    1. mst_ledger (Chart of Accounts) - 298 accounts
    2. mst_stock_item (Stock Items) 
    3. mst_cost_centre (Cost Centers)
    4. mst_godown (Warehouses/Locations) - 2 locations
    5. mst_group (Account Groups)
    6. mst_uom (Units of Measure) - 3 units
    
    Transaction Tables:
    1. trn_accounting (Accounting Entries) - 32,474 entries
    2. trn_inventory (Inventory Transactions) - 9,773 transactions
    3. trn_voucher (Voucher Headers) - 8,765 vouchers
    4. trn_bank (Bank Transactions)
    5. trn_bill (Bill Information)

    AGENT DELEGATION STRATEGY:

    Traditional Agents (Google ADK):
    - financial_analyst: Basic financial queries, account lookups
    - inventory_analyst: Basic inventory queries, stock levels
    - cost_management_analyst: Cost center analysis
    - tax_and_compliance_analyst: Tax calculations, compliance
    - banking_analyst: Bank transaction analysis

    Enhanced Analytics Agents (when available):
    - enhanced_financial: Advanced financial analytics with 4-tier analysis
    - enhanced_inventory: Advanced inventory analytics (coming soon)
    - enhanced_sales: Advanced sales analytics (coming soon)
    - enhanced_purchase: Advanced purchase analytics (coming soon)

    ANALYTICS TIERS AVAILABLE:
    1. Descriptive: "What happened?" - Historical summaries, KPIs, trends
    2. Diagnostic: "Why did it happen?" - Root cause analysis, anomaly detection
    3. Predictive: "What will happen?" - Forecasting, trend prediction
    4. Prescriptive: "What should we do?" - Recommendations, optimization

    DELEGATION RULES:
    - For basic queries: Use traditional ADK agents
    - For advanced analytics: Use enhanced agents if available
    - For multi-domain analysis: Coordinate between multiple agents
    - For 4-tier analytics: Always use enhanced agents

    Enhanced agents available: {ENHANCED_AGENTS_AVAILABLE}

    Example Queries:
    - "Show me cash flow analysis" → Enhanced Financial Agent (Descriptive)
    - "Why are expenses increasing?" → Enhanced Financial Agent (Diagnostic)  
    - "Predict next quarter revenue" → Enhanced Financial Agent (Predictive)
    - "How to optimize costs?" → Enhanced Financial Agent (Prescriptive)
    - "Simple account balance for Cash" → Traditional Financial Analyst

    Always choose the most appropriate agent based on query complexity and analytics requirements.
    """,
    sub_agents=[
        financial_analyst,
        inventory_analyst, 
        cost_management_analyst,
        tax_and_compliance_analyst,
        banking_analyst,
    ],
    tools=[
        get_current_time_tool,
    ],
)
