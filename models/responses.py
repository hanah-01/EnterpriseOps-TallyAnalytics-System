"""
Response models for structured agent communication
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    NO_DATA = "no_data"
    PARTIAL = "partial"

class QueryType(str, Enum):
    FINANCIAL = "financial"
    INVENTORY = "inventory"
    SALES = "sales"
    PURCHASE = "purchase"
    GENERAL = "general"
    ERROR = "error"

class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str
    error_message: str
    error_context: Optional[Dict[str, Any]] = None

class StructuredResponse(BaseModel):
    """Base structured response for all agents"""
    status: ResponseStatus
    data: Union[Dict[str, Any], List[Dict[str, Any]], ErrorDetail]
    message: str
    query_type: QueryType = QueryType.GENERAL
    execution_time_ms: int = 0
    agent_name: str = "Unknown"
    metadata: Optional[Dict[str, Any]] = None

class FinancialData(BaseModel):
    """Financial data model"""
    account_name: str
    balance: float
    currency: str = "INR"
    as_of_date: datetime
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    total_debits: float = 0.0
    total_credits: float = 0.0

class FinancialResponse(StructuredResponse):
    """Financial agent response"""
    data: Union[FinancialData, List[FinancialData], ErrorDetail]
    query_type: QueryType = QueryType.FINANCIAL

class InventoryData(BaseModel):
    """Inventory data model"""
    item_name: str
    current_stock: float
    unit_of_measure: str
    last_updated: datetime
    cost_price: float = 0.0
    location: Optional[str] = None

class InventoryResponse(StructuredResponse):
    """Inventory agent response"""
    data: Union[InventoryData, List[InventoryData], ErrorDetail]
    query_type: QueryType = QueryType.INVENTORY

class SalesData(BaseModel):
    """Sales data model"""
    customer_name: str
    total_sales: float
    transaction_count: int
    last_transaction_date: datetime
    avg_order_value: float = 0.0

class SalesResponse(StructuredResponse):
    """Sales agent response"""
    data: Union[SalesData, List[SalesData], ErrorDetail]
    query_type: QueryType = QueryType.SALES

class PurchaseData(BaseModel):
    """Purchase data model"""
    supplier_name: str
    total_purchases: float
    transaction_count: int
    last_purchase_date: datetime
    avg_order_value: float = 0.0

class PurchaseResponse(StructuredResponse):
    """Purchase agent response"""
    data: Union[PurchaseData, List[PurchaseData], ErrorDetail]
    query_type: QueryType = QueryType.PURCHASE

class MultiAgentResponse(StructuredResponse):
    """Multi-agent coordination response"""
    agent_results: Dict[str, Any]
    synthesis: Optional[Dict[str, Any]] = None
    coordination_plan: Optional[Dict[str, Any]] = None

class AnalyticsResult(BaseModel):
    """Analytics result model"""
    analytics_type: str
    query: str
    results: Dict[str, Any]
    insights: List[str] = []
    recommendations: List[str] = []
    confidence_level: float = 0.0
    execution_time_ms: int = 0