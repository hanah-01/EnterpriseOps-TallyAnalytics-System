"""
Analytics Framework for 4-Tier Business Intelligence
"""

from .base import AnalyticsBase, AnalyticsResponse
from .descriptive import DescriptiveAnalytics
from .diagnostic import DiagnosticAnalytics
from .predictive import PredictiveAnalytics
from .prescriptive import PrescriptiveAnalytics
from .models import FinancialAnalyticsModels
from .pipeline import AnalyticsDataPipeline
from .system_prompts import AnalyticsPromptTemplates

# Domain-specific analytics
from .inventory_descriptive import InventoryDescriptiveAnalytics
from .inventory_diagnostic import InventoryDiagnosticAnalytics
from .inventory_predictive import InventoryPredictiveAnalytics
from .inventory_prescriptive import InventoryPrescriptiveAnalytics

from .sales_descriptive import SalesDescriptiveAnalytics
from .sales_diagnostic import SalesDiagnosticAnalytics
from .sales_predictive import SalesPredictiveAnalytics
from .sales_prescriptive import SalesPrescriptiveAnalytics

from .purchase_descriptive import PurchaseDescriptiveAnalytics
from .purchase_diagnostic import PurchaseDiagnosticAnalytics
from .purchase_predictive import PurchasePredictiveAnalytics
from .purchase_prescriptive import PurchasePrescriptiveAnalytics

__all__ = [
    # Base classes
    'AnalyticsBase',
    'AnalyticsResponse', 
    'AnalyticsPromptTemplates',
    
    # Generic analytics
    'DescriptiveAnalytics',
    'DiagnosticAnalytics',
    'PredictiveAnalytics',
    'PrescriptiveAnalytics',
    'FinancialAnalyticsModels',
    'AnalyticsDataPipeline',
    
    # Inventory analytics
    'InventoryDescriptiveAnalytics',
    'InventoryDiagnosticAnalytics',
    'InventoryPredictiveAnalytics',
    'InventoryPrescriptiveAnalytics',
    
    # Sales analytics
    'SalesDescriptiveAnalytics',
    'SalesDiagnosticAnalytics', 
    'SalesPredictiveAnalytics',
    'SalesPrescriptiveAnalytics',
    
    # Purchase analytics
    'PurchaseDescriptiveAnalytics',
    'PurchaseDiagnosticAnalytics',
    'PurchasePredictiveAnalytics',
    'PurchasePrescriptiveAnalytics',
]