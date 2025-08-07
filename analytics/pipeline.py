"""
Analytics Data Pipeline for 4-Tier Business Intelligence
Supports all agent types: Financial, Inventory, Sales, Tax, Purchase
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import logging
import structlog

logger = structlog.get_logger(__name__)


class DataExtractor:
    """Base class for data extraction from different sources"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.logger = structlog.get_logger(f"{__name__}.{agent_type}")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data from source"""
        raise NotImplementedError("Subclasses must implement extract method")


class FinancialDataExtractor(DataExtractor):
    """Extract financial data from Tally database"""
    
    def __init__(self):
        super().__init__("financial")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract financial data based on query type"""
        if query == "cash_flow":
            return self.extract_cash_flow_data(params)
        elif query == "account_balances":
            return self.extract_account_balance_data(params)
        elif query == "transactions":
            return self.extract_transaction_data(params)
        elif query == "payments":
            return self.extract_payment_data(params)
        else:
            return self.extract_general_financial_data(params)
    
    def extract_cash_flow_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract cash flow data"""
        # This would typically connect to the database
        # For now, return sample structure
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'amount': np.random.randn(100) * 1000,
            'account_name': np.random.choice(['Cash', 'Bank', 'Receivables', 'Payables'], 100),
            'transaction_type': np.random.choice(['Inflow', 'Outflow'], 100)
        })
    
    def extract_account_balance_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract account balance data"""
        return pd.DataFrame({
            'account_name': ['Cash', 'Bank', 'Receivables', 'Payables', 'Inventory'],
            'balance': [10000, -5000, 25000, -15000, 30000],
            'account_type': ['Asset', 'Asset', 'Asset', 'Liability', 'Asset']
        })
    
    def extract_transaction_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract transaction data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=200, freq='D'),
            'amount': np.random.exponential(scale=1000, size=200),
            'transaction_type': np.random.choice(['Payment', 'Receipt', 'Transfer'], 200),
            'account_name': np.random.choice(['Cash', 'Bank', 'Receivables', 'Payables'], 200),
            'customer_name': np.random.choice(['Customer A', 'Customer B', 'Customer C'], 200)
        })
    
    def extract_payment_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract payment data"""
        return pd.DataFrame({
            'customer_name': np.random.choice(['Customer A', 'Customer B', 'Customer C'], 100),
            'amount': np.random.exponential(scale=5000, size=100),
            'due_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'payment_date': pd.date_range(start='2024-01-05', periods=100, freq='D'),
            'payment_status': np.random.choice(['Paid', 'Overdue', 'Pending'], 100)
        })
    
    def extract_general_financial_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract general financial data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'revenue': np.random.exponential(scale=10000, size=365),
            'expenses': np.random.exponential(scale=8000, size=365),
            'net_income': np.random.randn(365) * 2000
        })


class InventoryDataExtractor(DataExtractor):
    """Extract inventory data from Tally database"""
    
    def __init__(self):
        super().__init__("inventory")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract inventory data based on query type"""
        if query == "stock_levels":
            return self.extract_stock_level_data(params)
        elif query == "product_movement":
            return self.extract_product_movement_data(params)
        elif query == "demand_history":
            return self.extract_demand_history_data(params)
        else:
            return self.extract_general_inventory_data(params)
    
    def extract_stock_level_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract current stock levels"""
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'product_name': ['Samsung Charger', 'iPhone Case', 'Headphones', 'Cable', 'Battery'],
            'current_stock': [100, 50, 75, 200, 25],
            'reorder_point': [20, 10, 15, 50, 5],
            'max_stock': [500, 200, 300, 1000, 100],
            'unit_cost': [25.0, 15.0, 100.0, 10.0, 50.0],
            'category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', 'Electronics']
        })
    
    def extract_product_movement_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract product movement data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=300, freq='D'),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 300),
            'movement_type': np.random.choice(['IN', 'OUT'], 300),
            'quantity': np.random.randint(1, 20, 300),
            'unit_price': np.random.uniform(10, 100, 300)
        })
    
    def extract_demand_history_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract demand history data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 365),
            'quantity_demanded': np.random.poisson(lam=10, size=365),
            'fulfilled_quantity': np.random.poisson(lam=9, size=365)
        })
    
    def extract_general_inventory_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract general inventory data"""
        return pd.DataFrame({
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 500),
            'transaction_date': pd.date_range(start='2024-01-01', periods=500, freq='D'),
            'quantity': np.random.randint(1, 50, 500),
            'value': np.random.uniform(100, 5000, 500)
        })


class SalesDataExtractor(DataExtractor):
    """Extract sales data from Tally database"""
    
    def __init__(self):
        super().__init__("sales")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract sales data based on query type"""
        if query == "sales_transactions":
            return self.extract_sales_transaction_data(params)
        elif query == "customer_data":
            return self.extract_customer_data(params)
        elif query == "product_sales":
            return self.extract_product_sales_data(params)
        else:
            return self.extract_general_sales_data(params)
    
    def extract_sales_transaction_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract sales transaction data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=1000, freq='D'),
            'customer_id': np.random.choice(['C001', 'C002', 'C003', 'C004', 'C005'], 1000),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 1000),
            'quantity': np.random.randint(1, 10, 1000),
            'unit_price': np.random.uniform(10, 500, 1000),
            'total_amount': np.random.uniform(100, 5000, 1000),
            'discount': np.random.uniform(0, 100, 1000)
        })
    
    def extract_customer_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract customer data"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'customer_name': ['Customer A', 'Customer B', 'Customer C', 'Customer D', 'Customer E'],
            'total_purchases': [150, 200, 75, 300, 125],
            'total_spent': [15000, 25000, 8000, 40000, 12000],
            'first_purchase_date': pd.date_range(start='2023-01-01', periods=5, freq='30D'),
            'last_purchase_date': pd.date_range(start='2024-01-01', periods=5, freq='30D'),
            'customer_type': ['Premium', 'Regular', 'New', 'Premium', 'Regular']
        })
    
    def extract_product_sales_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract product sales data"""
        return pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'product_name': ['Samsung Charger', 'iPhone Case', 'Headphones', 'Cable', 'Battery'],
            'total_sold': [1000, 500, 750, 2000, 250],
            'total_revenue': [25000, 7500, 75000, 20000, 12500],
            'avg_selling_price': [25.0, 15.0, 100.0, 10.0, 50.0],
            'margin_percent': [40.0, 60.0, 30.0, 50.0, 35.0]
        })
    
    def extract_general_sales_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract general sales data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'sales_amount': np.random.exponential(scale=5000, size=365),
            'units_sold': np.random.poisson(lam=20, size=365),
            'profit_margin': np.random.uniform(0.2, 0.6, 365)
        })


class TaxDataExtractor(DataExtractor):
    """Extract tax data from Tally database"""
    
    def __init__(self):
        super().__init__("tax")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract tax data based on query type"""
        if query == "tax_transactions":
            return self.extract_tax_transaction_data(params)
        elif query == "gst_data":
            return self.extract_gst_data(params)
        elif query == "compliance_data":
            return self.extract_compliance_data(params)
        else:
            return self.extract_general_tax_data(params)
    
    def extract_tax_transaction_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract tax transaction data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=500, freq='D'),
            'transaction_id': [f'TXN{i:06d}' for i in range(500)],
            'taxable_amount': np.random.uniform(1000, 50000, 500),
            'tax_rate': np.random.choice([0.05, 0.12, 0.18, 0.28], 500),
            'tax_amount': np.random.uniform(50, 5000, 500),
            'tax_type': np.random.choice(['CGST', 'SGST', 'IGST'], 500)
        })
    
    def extract_gst_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract GST data"""
        return pd.DataFrame({
            'month': pd.date_range(start='2024-01-01', periods=12, freq='M'),
            'input_tax': np.random.uniform(10000, 50000, 12),
            'output_tax': np.random.uniform(12000, 60000, 12),
            'net_tax_liability': np.random.uniform(2000, 10000, 12),
            'return_status': np.random.choice(['Filed', 'Pending', 'Late'], 12)
        })
    
    def extract_compliance_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract compliance data"""
        return pd.DataFrame({
            'compliance_type': ['GST Return', 'TDS Return', 'Advance Tax', 'Annual Return'],
            'due_date': pd.date_range(start='2024-01-01', periods=4, freq='90D'),
            'filing_date': pd.date_range(start='2024-01-05', periods=4, freq='90D'),
            'status': ['Completed', 'Pending', 'Completed', 'Overdue'],
            'penalty_amount': [0, 0, 0, 5000]
        })
    
    def extract_general_tax_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract general tax data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'total_income': np.random.uniform(5000, 50000, 365),
            'tax_amount': np.random.uniform(500, 5000, 365),
            'effective_tax_rate': np.random.uniform(0.1, 0.3, 365)
        })


class PurchaseDataExtractor(DataExtractor):
    """Extract purchase data from Tally database"""
    
    def __init__(self):
        super().__init__("purchase")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract purchase data based on query type"""
        if query == "purchase_transactions":
            return self.extract_purchase_transaction_data(params)
        elif query == "supplier_data":
            return self.extract_supplier_data(params)
        elif query == "cost_analysis":
            return self.extract_cost_analysis_data(params)
        else:
            return self.extract_general_purchase_data(params)
    
    def extract_purchase_transaction_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract purchase transaction data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=800, freq='D'),
            'supplier_id': np.random.choice(['S001', 'S002', 'S003', 'S004', 'S005'], 800),
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], 800),
            'quantity': np.random.randint(10, 100, 800),
            'unit_cost': np.random.uniform(5, 200, 800),
            'total_amount': np.random.uniform(500, 20000, 800),
            'payment_terms': np.random.choice(['30 days', '45 days', '60 days'], 800)
        })
    
    def extract_supplier_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract supplier data"""
        return pd.DataFrame({
            'supplier_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'supplier_name': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E'],
            'total_orders': [50, 75, 30, 100, 40],
            'total_spent': [50000, 75000, 30000, 100000, 40000],
            'on_time_deliveries': [48, 70, 28, 95, 38],
            'total_deliveries': [50, 75, 30, 100, 40],
            'quality_rating': [4.5, 4.2, 3.8, 4.8, 4.0],
            'average_payment_days': [30, 35, 28, 32, 40]
        })
    
    def extract_cost_analysis_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract cost analysis data"""
        return pd.DataFrame({
            'cost_category': ['Raw Materials', 'Labor', 'Overhead', 'Logistics', 'Quality'],
            'amount': [100000, 50000, 30000, 15000, 10000],
            'percentage': [48.8, 24.4, 14.6, 7.3, 4.9],
            'variability': ['High', 'Medium', 'Low', 'Medium', 'Low'],
            'optimization_potential': ['High', 'Medium', 'Low', 'High', 'Medium']
        })
    
    def extract_general_purchase_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Extract general purchase data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'purchase_amount': np.random.exponential(scale=8000, size=365),
            'supplier_count': np.random.randint(1, 5, 365),
            'cost_per_unit': np.random.uniform(10, 100, 365)
        })


class MarketDataExtractor(DataExtractor):
    """Extract market data from external sources"""
    
    def __init__(self):
        super().__init__("market")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract market data"""
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=365, freq='D'),
            'market_index': np.random.uniform(95, 105, 365),
            'sector_performance': np.random.uniform(90, 110, 365),
            'economic_indicator': np.random.uniform(0.95, 1.05, 365)
        })


class CustomerDataExtractor(DataExtractor):
    """Extract customer data from CRM systems"""
    
    def __init__(self):
        super().__init__("customer")
    
    def extract(self, source: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract customer data"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'acquisition_date': pd.date_range(start='2023-01-01', periods=5, freq='60D'),
            'satisfaction_score': np.random.uniform(3.0, 5.0, 5),
            'support_tickets': np.random.randint(0, 10, 5),
            'referrals_made': np.random.randint(0, 5, 5)
        })


class DataTransformer:
    """Transform raw data for analytics consumption"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.logger = structlog.get_logger(f"{__name__}.{agent_type}")
    
    def transform(self, data: pd.DataFrame, transformation_type: str = "standard") -> pd.DataFrame:
        """Apply transformations to data"""
        transformed_data = data.copy()
        
        # Handle missing values
        transformed_data = self.handle_missing_values(transformed_data)
        
        # Encode categorical variables
        transformed_data = self.encode_categorical_variables(transformed_data)
        
        # Scale numerical variables
        transformed_data = self.scale_numerical_variables(transformed_data, transformation_type)
        
        # Create derived features
        transformed_data = self.create_derived_features(transformed_data)
        
        # Feature selection
        transformed_data = self.select_features(transformed_data)
        
        return transformed_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        # Numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if 'numerical' not in self.imputers:
                self.imputers['numerical'] = SimpleImputer(strategy='median')
                data[numerical_cols] = self.imputers['numerical'].fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.imputers['numerical'].transform(data[numerical_cols])
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            if 'categorical' not in self.imputers:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = self.imputers['categorical'].fit_transform(data[categorical_cols])
            else:
                data[categorical_cols] = self.imputers['categorical'].transform(data[categorical_cols])
        
        return data
    
    def encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['date']:  # Skip date columns
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[f'{col}_encoded'] = self.encoders[col].transform(data[col].astype(str))
        
        return data
    
    def scale_numerical_variables(self, data: pd.DataFrame, transformation_type: str) -> pd.DataFrame:
        """Scale numerical variables"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            if transformation_type == "standard":
                if 'standard' not in self.scalers:
                    self.scalers['standard'] = StandardScaler()
                    data[numerical_cols] = self.scalers['standard'].fit_transform(data[numerical_cols])
                else:
                    data[numerical_cols] = self.scalers['standard'].transform(data[numerical_cols])
            elif transformation_type == "minmax":
                if 'minmax' not in self.scalers:
                    self.scalers['minmax'] = MinMaxScaler()
                    data[numerical_cols] = self.scalers['minmax'].fit_transform(data[numerical_cols])
                else:
                    data[numerical_cols] = self.scalers['minmax'].transform(data[numerical_cols])
        
        return data
    
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features specific to agent type"""
        if self.agent_type == "financial":
            return self.create_financial_features(data)
        elif self.agent_type == "inventory":
            return self.create_inventory_features(data)
        elif self.agent_type == "sales":
            return self.create_sales_features(data)
        elif self.agent_type == "tax":
            return self.create_tax_features(data)
        elif self.agent_type == "purchase":
            return self.create_purchase_features(data)
        else:
            return data
    
    def create_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create financial-specific derived features"""
        # Add time-based features if date column exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['day_of_week'] = data['date'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Add financial ratios if amount columns exist
        if 'amount' in data.columns:
            data['amount_abs'] = data['amount'].abs()
            data['amount_log'] = np.log1p(data['amount_abs'])
            
            # Rolling statistics
            if len(data) > 7:
                data['amount_rolling_mean_7'] = data['amount'].rolling(window=7, min_periods=1).mean()
                data['amount_rolling_std_7'] = data['amount'].rolling(window=7, min_periods=1).std()
        
        return data
    
    def create_inventory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create inventory-specific derived features"""
        # Stock turnover features
        if 'current_stock' in data.columns and 'quantity' in data.columns:
            data['stock_turnover'] = data['quantity'] / (data['current_stock'] + 1)
        
        # Reorder indicators
        if 'current_stock' in data.columns and 'reorder_point' in data.columns:
            data['needs_reorder'] = (data['current_stock'] <= data['reorder_point']).astype(int)
            data['stock_ratio'] = data['current_stock'] / data['reorder_point']
        
        return data
    
    def create_sales_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sales-specific derived features"""
        # Customer lifetime value features
        if 'total_spent' in data.columns and 'total_purchases' in data.columns:
            data['avg_order_value'] = data['total_spent'] / data['total_purchases']
        
        # Time-based features
        if 'first_purchase_date' in data.columns and 'last_purchase_date' in data.columns:
            data['customer_lifetime_days'] = (
                pd.to_datetime(data['last_purchase_date']) - 
                pd.to_datetime(data['first_purchase_date'])
            ).dt.days
        
        return data
    
    def create_tax_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create tax-specific derived features"""
        # Tax rate features
        if 'tax_amount' in data.columns and 'taxable_amount' in data.columns:
            data['effective_tax_rate'] = data['tax_amount'] / data['taxable_amount']
        
        # Compliance features
        if 'due_date' in data.columns and 'filing_date' in data.columns:
            data['filing_delay'] = (
                pd.to_datetime(data['filing_date']) - 
                pd.to_datetime(data['due_date'])
            ).dt.days
            data['is_late'] = (data['filing_delay'] > 0).astype(int)
        
        return data
    
    def create_purchase_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create purchase-specific derived features"""
        # Supplier performance features
        if 'on_time_deliveries' in data.columns and 'total_deliveries' in data.columns:
            data['delivery_performance'] = data['on_time_deliveries'] / data['total_deliveries']
        
        # Cost efficiency features
        if 'total_spent' in data.columns and 'total_orders' in data.columns:
            data['avg_order_value'] = data['total_spent'] / data['total_orders']
        
        return data
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for analysis"""
        # This is a simplified feature selection
        # In practice, you'd use more sophisticated methods
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        missing_ratio = data.isnull().sum() / len(data)
        columns_to_keep = missing_ratio[missing_ratio < missing_threshold].index
        
        return data[columns_to_keep]


class FeatureEngineer:
    """Engineer features for analytics"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.logger = structlog.get_logger(f"{__name__}.{agent_type}")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for the specific agent type"""
        if self.agent_type == "financial":
            return self.engineer_financial_features(data)
        elif self.agent_type == "inventory":
            return self.engineer_inventory_features(data)
        elif self.agent_type == "sales":
            return self.engineer_sales_features(data)
        elif self.agent_type == "tax":
            return self.engineer_tax_features(data)
        elif self.agent_type == "purchase":
            return self.engineer_purchase_features(data)
        else:
            return data
    
    def engineer_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer financial-specific features"""
        # Cash flow patterns
        if 'amount' in data.columns and 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            
            # Lag features
            for lag in [1, 3, 7, 30]:
                data[f'amount_lag_{lag}'] = data['amount'].shift(lag)
            
            # Moving averages
            for window in [7, 30, 90]:
                data[f'amount_ma_{window}'] = data['amount'].rolling(window=window).mean()
                data[f'amount_std_{window}'] = data['amount'].rolling(window=window).std()
            
            # Volatility measures
            data['amount_volatility'] = data['amount'].rolling(window=30).std()
            data['amount_trend'] = data['amount'].rolling(window=30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        return data
    
    def engineer_inventory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer inventory-specific features"""
        # ABC analysis features
        if 'quantity' in data.columns and 'unit_cost' in data.columns:
            data['inventory_value'] = data['quantity'] * data['unit_cost']
            
            # Categorize by value
            data['value_category'] = pd.cut(
                data['inventory_value'], 
                bins=3, 
                labels=['C', 'B', 'A']
            )
        
        # Stock movement patterns
        if 'date' in data.columns and 'quantity' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            
            # Velocity features
            data['quantity_velocity'] = data['quantity'].diff()
            data['quantity_acceleration'] = data['quantity_velocity'].diff()
        
        return data
    
    def engineer_sales_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer sales-specific features"""
        # RFM analysis
        if 'customer_id' in data.columns:
            current_date = pd.Timestamp.now()
            
            if 'last_purchase_date' in data.columns:
                data['recency'] = (current_date - pd.to_datetime(data['last_purchase_date'])).dt.days
            
            if 'total_purchases' in data.columns:
                data['frequency'] = data['total_purchases']
            
            if 'total_spent' in data.columns:
                data['monetary'] = data['total_spent']
        
        # Seasonal features
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['season'] = data['date'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        return data
    
    def engineer_tax_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer tax-specific features"""
        # Tax burden features
        if 'tax_amount' in data.columns and 'total_income' in data.columns:
            data['tax_burden'] = data['tax_amount'] / data['total_income']
        
        # Compliance risk features
        if 'filing_date' in data.columns and 'due_date' in data.columns:
            data['filing_timeliness'] = (
                pd.to_datetime(data['due_date']) - 
                pd.to_datetime(data['filing_date'])
            ).dt.days
            data['compliance_score'] = np.where(data['filing_timeliness'] >= 0, 1, 0)
        
        return data
    
    def engineer_purchase_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer purchase-specific features"""
        # Supplier concentration
        if 'supplier_id' in data.columns and 'total_amount' in data.columns:
            supplier_total = data.groupby('supplier_id')['total_amount'].sum()
            total_spend = data['total_amount'].sum()
            data['supplier_concentration'] = data['supplier_id'].map(supplier_total / total_spend)
        
        # Cost efficiency
        if 'quantity' in data.columns and 'total_amount' in data.columns:
            data['cost_per_unit'] = data['total_amount'] / data['quantity']
        
        return data


class DataNormalizer:
    """Normalize data for analytics consumption"""
    
    def __init__(self):
        self.normalizers = {}
    
    def normalize(self, data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Normalize data using specified method"""
        normalized_data = data.copy()
        numerical_cols = normalized_data.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            if 'standard' not in self.normalizers:
                self.normalizers['standard'] = StandardScaler()
                normalized_data[numerical_cols] = self.normalizers['standard'].fit_transform(normalized_data[numerical_cols])
            else:
                normalized_data[numerical_cols] = self.normalizers['standard'].transform(normalized_data[numerical_cols])
        elif method == "minmax":
            if 'minmax' not in self.normalizers:
                self.normalizers['minmax'] = MinMaxScaler()
                normalized_data[numerical_cols] = self.normalizers['minmax'].fit_transform(normalized_data[numerical_cols])
            else:
                normalized_data[numerical_cols] = self.normalizers['minmax'].transform(normalized_data[numerical_cols])
        
        return normalized_data


class DataAggregator:
    """Aggregate data for analytics"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
    
    def aggregate(self, data: pd.DataFrame, aggregation_type: str = "daily") -> pd.DataFrame:
        """Aggregate data based on type"""
        if 'date' not in data.columns:
            return data
        
        data['date'] = pd.to_datetime(data['date'])
        
        if aggregation_type == "daily":
            return data.groupby(data['date'].dt.date).agg(self.get_aggregation_functions()).reset_index()
        elif aggregation_type == "weekly":
            return data.groupby(data['date'].dt.to_period('W')).agg(self.get_aggregation_functions()).reset_index()
        elif aggregation_type == "monthly":
            return data.groupby(data['date'].dt.to_period('M')).agg(self.get_aggregation_functions()).reset_index()
        else:
            return data
    
    def get_aggregation_functions(self) -> Dict[str, List[str]]:
        """Get aggregation functions for different column types"""
        return {
            'amount': ['sum', 'mean', 'count', 'std'],
            'quantity': ['sum', 'mean', 'count'],
            'price': ['mean', 'min', 'max']
        }


class AnalyticsDataPipeline:
    """Complete data pipeline for analytics"""
    
    def __init__(self, agent_type: str = "financial"):
        self.agent_type = agent_type
        self.extractors = self.initialize_extractors()
        self.transformers = self.initialize_transformers()
        self.logger = structlog.get_logger(f"{__name__}.{agent_type}")
    
    def initialize_extractors(self) -> Dict[str, DataExtractor]:
        """Initialize data extractors"""
        return {
            'financial': FinancialDataExtractor(),
            'inventory': InventoryDataExtractor(),
            'sales': SalesDataExtractor(),
            'tax': TaxDataExtractor(),
            'purchase': PurchaseDataExtractor(),
            'market': MarketDataExtractor(),
            'customer': CustomerDataExtractor()
        }
    
    def initialize_transformers(self) -> Dict[str, Any]:
        """Initialize data transformers"""
        return {
            'feature_engineering': FeatureEngineer(self.agent_type),
            'normalization': DataNormalizer(),
            'aggregation': DataAggregator(self.agent_type),
            'transformation': DataTransformer(self.agent_type)
        }
    
    def process_for_analytics(self, data_type: str, query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Process raw data for analytics consumption"""
        # Extract data
        extractor = self.extractors.get(data_type)
        if not extractor:
            raise ValueError(f"No extractor available for data type: {data_type}")
        
        raw_data = extractor.extract("database", query, params)
        
        # Transform data
        transformer = self.transformers['transformation']
        transformed_data = transformer.transform(raw_data)
        
        # Engineer features
        feature_engineer = self.transformers['feature_engineering']
        engineered_data = feature_engineer.engineer_features(transformed_data)
        
        # Normalize data
        normalizer = self.transformers['normalization']
        normalized_data = normalizer.normalize(engineered_data)
        
        self.logger.info(f"Processed {len(normalized_data)} records for {data_type} analytics")
        
        return normalized_data
    
    def get_processed_data(self, data_sources: List[str], query: str = None, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Get processed data from multiple sources"""
        processed_datasets = []
        
        for source in data_sources:
            try:
                processed_data = self.process_for_analytics(source, query, params)
                processed_datasets.append(processed_data)
            except Exception as e:
                self.logger.error(f"Error processing {source}: {str(e)}")
                continue
        
        if not processed_datasets:
            raise ValueError("No data could be processed from the provided sources")
        
        # Combine datasets if multiple sources
        if len(processed_datasets) == 1:
            return processed_datasets[0]
        else:
            # Simple concatenation - in practice, you'd use more sophisticated merging
            return pd.concat(processed_datasets, ignore_index=True)


# Factory function to create pipelines for different agents
def create_analytics_pipeline(agent_type: str) -> AnalyticsDataPipeline:
    """Factory function to create analytics pipeline for different agent types"""
    return AnalyticsDataPipeline(agent_type)


# Utility functions
def get_pipeline_info(agent_type: str) -> Dict[str, Any]:
    """Get information about pipeline capabilities for an agent type"""
    pipeline = create_analytics_pipeline(agent_type)
    
    return {
        'agent_type': agent_type,
        'available_extractors': list(pipeline.extractors.keys()),
        'available_transformers': list(pipeline.transformers.keys()),
        'supported_data_types': list(pipeline.extractors.keys())
    }


def validate_pipeline_data(data: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate that data meets pipeline requirements"""
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, missing_columns
    else:
        return True, []