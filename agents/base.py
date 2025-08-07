"""
Enhanced Base Agent Class with Analytics Integration
Built for your existing Tally multi-agent system
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Your existing imports
import google.generativeai as genai

# Import your existing database connection
try:
    from database.connection import get_query_engine, QueryEngine
except ImportError:
    from ..database.connection import get_query_engine, QueryEngine

# Analytics imports
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
    )

# Models imports  
try:
    from models.responses import (
        StructuredResponse, ResponseStatus, QueryType, ErrorDetail
    )
except ImportError:
    from ..models.responses import (
        StructuredResponse, ResponseStatus, QueryType, ErrorDetail
    )

# Set up logging (fallback if structlog not available)
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Enhanced base class for all specialized agents with analytics integration"""
    
    def __init__(
        self,
        name: str,
        agent_type: QueryType,
        tables: List[str],
        model: str = "gemini-2.5-flash"
    ):
        self.name = name
        self.agent_type = agent_type
        self.tables = tables
        self.model = model
        self.client = None
        self.query_engine: Optional[QueryEngine] = None
        
        # Initialize 4-tier analytics
        self.descriptive_analytics = DescriptiveAnalytics(agent_type.value)
        self.diagnostic_analytics = DiagnosticAnalytics(agent_type.value)
        self.predictive_analytics = PredictiveAnalytics(agent_type.value)
        self.prescriptive_analytics = PrescriptiveAnalytics(agent_type.value)
        
        # Initialize tools and connections
        self.tools = self._initialize_tools()
        self._initialize_gemini_client()
        self.initialize()
        
        logger.info(f"{self.name} agent initialized with 4-tier analytics")
    
    def _initialize_gemini_client(self):
        """Initialize Google AI SDK client"""
        try:
            # Try to get API key from environment
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning(f"No API key found for {self.name} agent")
                return
            
            # Configure client
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
            
            logger.info(f"Gemini client initialized for {self.name} agent")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client for {self.name}: {e}")
            self.client = None
    
    @abstractmethod
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize function calling tools for this agent"""
        pass
    
    def initialize(self):
        """Initialize agent with database connection"""
        try:
            self.query_engine = get_query_engine()
            logger.info(f"{self.name} agent database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed for {self.name}: {e}")
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """Build domain-specific system prompt for the agent"""
        if context is None:
            context = {}
            
        return f"""
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
"""
    
    def execute_database_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute database query with validation and error handling"""
        if not self.query_engine:
            raise RuntimeError(f"Query engine not initialized for {self.name}")
        
        try:
            return self.query_engine.execute_query(
                query=query,
                params=params,
                cache_key=cache_key
            )
        except Exception as e:
            logger.error(f"Database query failed in {self.name}: {e}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Parameters: {params}")
            raise
    
    def get_analytics(
        self,
        analytics_type: str,
        query: str,
        date_from: str = None,
        date_to: str = None,
        parameters: Dict[str, Any] = None
    ) -> AnalyticsResult:
        """Execute 4-tier analytics with data from database"""
        try:
            logger.info(f"Running {analytics_type} analytics for {self.name}: {query}")
            
            # Get data for analytics
            data = self._get_analytics_data(date_from, date_to, parameters)
            
            if data.empty:
                return AnalyticsResult(
                    analytics_type, query,
                    {"error": "No data available for analysis"},
                    ["No data found for the specified criteria"],
                    ["Check date range and filters"], 0.0
                )
            
            # Route to appropriate analytics
            analytics_params = parameters or {}
            analytics_params.update({
                'date_from': date_from,
                'date_to': date_to,
                'agent_type': self.agent_type.value
            })
            
            if analytics_type.lower() == "descriptive":
                return self.descriptive_analytics.analyze(query, data, analytics_params)
            elif analytics_type.lower() == "diagnostic":
                return self.diagnostic_analytics.analyze(query, data, analytics_params)
            elif analytics_type.lower() == "predictive":
                return self.predictive_analytics.analyze(query, data, analytics_params)
            elif analytics_type.lower() == "prescriptive":
                return self.prescriptive_analytics.analyze(query, data, analytics_params)
            else:
                # Default to descriptive
                return self.descriptive_analytics.analyze(query, data, analytics_params)
                
        except Exception as e:
            logger.error(f"Analytics execution failed in {self.name}: {e}")
            return AnalyticsResult(
                analytics_type, query,
                {"error": str(e)}, [], [], 0.0
            )
    
    def _get_analytics_data(
        self, 
        date_from: str = None, 
        date_to: str = None,
        parameters: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Get data for analytics - to be customized by each agent"""
        try:
            # Default query - subclasses should override this
            where_conditions = []
            
            if date_from:
                where_conditions.append(f"v.date >= '{date_from}'")
            if date_to:
                where_conditions.append(f"v.date <= '{date_to}'")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Basic query that works for most agents
            query = f"""
            SELECT 
                v.guid,
                v.date,
                v.voucher_type,
                v.voucher_number,
                v.party_name,
                a.ledger,
                a.amount
            FROM trn_voucher v
            LEFT JOIN trn_accounting a ON v.guid = a.guid
            WHERE {where_clause}
            ORDER BY v.date DESC
            LIMIT 1000
            """
            
            result = self.execute_database_query(query)
            return pd.DataFrame(result) if result else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Data retrieval failed in {self.name}: {e}")
            return pd.DataFrame()
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple[bool, str]:
        """Validate date range parameters"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start > end:
                return False, "Start date cannot be after end date"
            
            # Check if date range is reasonable (not more than 2 years)
            if (end - start).days > 730:
                return False, "Date range cannot exceed 2 years"
            
            return True, "Date range is valid"
            
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"
    
    def format_currency(self, amount: float, currency: str = "INR") -> str:
        """Format currency amount"""
        if currency == "INR":
            return f"₹{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not accessible by {self.name} agent")
        
        try:
            schema = self.query_engine.get_table_schema(table_name)
            return {
                "table_name": table_name,
                "schema": schema,
                "agent_access": True
            }
        except Exception as e:
            logger.error(f"Schema retrieval failed for {table_name}: {e}")
            return {
                "table_name": table_name,
                "error": str(e),
                "agent_access": True
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        try:
            # Test database connection
            db_status = False
            if self.query_engine:
                try:
                    db_status = self.query_engine.test_connection()
                except:
                    db_status = False
            
            # Test Gemini client
            gemini_status = self.client is not None
            
            # Test analytics modules
            analytics_status = all([
                self.descriptive_analytics is not None,
                self.diagnostic_analytics is not None,
                self.predictive_analytics is not None,
                self.prescriptive_analytics is not None
            ])
            
            return {
                "agent_name": self.name,
                "agent_type": self.agent_type.value,
                "database_connected": db_status,
                "gemini_client_ready": gemini_status,
                "analytics_ready": analytics_status,
                "tables_accessible": len(self.tables),
                "tools_available": len(self.tools),
                "status": "healthy" if db_status and analytics_status else "unhealthy"
            }
            
        except Exception as e:
            return {
                "agent_name": self.name,
                "status": "error", 
                "error": str(e)
            }
    
    # Compatibility methods for your existing system
    def get_summary(self, **kwargs) -> Dict[str, Any]:
        """Get summary - compatibility method"""
        try:
            # Default implementation - subclasses should override
            result = self.get_analytics("descriptive", "summary analysis", **kwargs)
            return {
                "summary": result.results,
                "insights": result.insights,
                "agent": self.name
            }
        except Exception as e:
            return {"error": str(e), "agent": self.name}
