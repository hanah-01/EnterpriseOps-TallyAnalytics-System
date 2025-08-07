"""
Purchase Descriptive Analytics - "What happened?"
Historical purchase data analysis and current state reporting
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import AnalyticsBase, AnalyticsResponse


class PurchaseDescriptiveAnalytics(AnalyticsBase):
    """Descriptive analytics for purchase historical data analysis"""
    
    def __init__(self, agent_type: str = "purchase"):
        super().__init__("PurchaseDescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'procurement_analysis',
            'supplier_segmentation',
            'purchase_trends',
            'cost_analysis',
            'transaction_patterns',
            'voucher_type_analysis',
            'warehouse_distribution',
            'time_series_analysis'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform descriptive analytics on purchase data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'procurement_analysis':
            results = self.analyze_procurement_patterns(data, params)
        elif analysis_type == 'supplier_segmentation':
            results = self.analyze_supplier_segments(data, params)
        elif analysis_type == 'purchase_trends':
            results = self.analyze_purchase_trends(data, params)
        elif analysis_type == 'cost_analysis':
            results = self.analyze_cost_patterns(data, params)
        elif analysis_type == 'transaction_patterns':
            results = self.analyze_transaction_patterns(data, params)
        elif analysis_type == 'voucher_type_analysis':
            results = self.analyze_voucher_types(data, params)
        elif analysis_type == 'warehouse_distribution':
            results = self.analyze_warehouse_distribution(data, params)
        elif analysis_type == 'time_series_analysis':
            results = self.analyze_time_series_patterns(data, params)
        else:
            results = self.analyze_general_purchase_summary(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_procurement_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze procurement patterns and spending distribution"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for positive amounts (purchases)
            procurement_data = data[data['amount'] > 0]
            
            # Calculate procurement metrics
            total_procurement = procurement_data['amount'].sum()
            avg_transaction_value = procurement_data['amount'].mean()
            median_transaction_value = procurement_data['amount'].median()
            procurement_std = procurement_data['amount'].std()
            
            # Procurement distribution
            procurement_quartiles = procurement_data['amount'].quantile([0.25, 0.5, 0.75])
            
            # Daily procurement analysis
            procurement_data['date'] = pd.to_datetime(procurement_data['date'])
            daily_procurement = procurement_data.groupby(procurement_data['date'].dt.date)['amount'].sum()
            
            # Top procurement days
            top_procurement_days = daily_procurement.nlargest(10)
            
            return {
                "total_procurement": round(total_procurement, 2),
                "avg_transaction_value": round(avg_transaction_value, 2),
                "median_transaction_value": round(median_transaction_value, 2),
                "procurement_std": round(procurement_std, 2),
                "procurement_quartiles": {
                    "q1": round(procurement_quartiles[0.25], 2),
                    "q2": round(procurement_quartiles[0.5], 2),
                    "q3": round(procurement_quartiles[0.75], 2)
                },
                "daily_procurement_stats": {
                    "avg_daily_procurement": round(daily_procurement.mean(), 2),
                    "max_daily_procurement": round(daily_procurement.max(), 2),
                    "min_daily_procurement": round(daily_procurement.min(), 2)
                },
                "top_procurement_days": top_procurement_days.to_dict(),
                "total_procurement_transactions": len(procurement_data)
            }
            
        except Exception as e:
            return {"error": f"Procurement analysis failed: {str(e)}"}
    
    def analyze_supplier_segments(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supplier segmentation and performance"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for suppliers with positive amounts
            supplier_data = data[data['amount'] > 0].copy()
            
            # Supplier metrics
            supplier_metrics = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': 'sum',
                'date': ['min', 'max', 'nunique']
            }).round(2)
            
            # Flatten column names
            supplier_metrics.columns = ['_'.join(col).strip() for col in supplier_metrics.columns]
            
            # Supplier segmentation using spend and frequency analysis
            supplier_metrics['total_spending'] = supplier_metrics['amount_sum']
            supplier_metrics['transaction_count'] = supplier_metrics['amount_count']
            supplier_metrics['avg_transaction_value'] = supplier_metrics['amount_mean']
            
            # Segmentation thresholds
            spending_threshold = supplier_metrics['total_spending'].quantile(0.8)
            frequency_threshold = supplier_metrics['transaction_count'].quantile(0.8)
            
            # Create segments
            supplier_metrics['segment'] = 'Standard'
            supplier_metrics.loc[
                (supplier_metrics['total_spending'] >= spending_threshold) & 
                (supplier_metrics['transaction_count'] >= frequency_threshold), 'segment'
            ] = 'Strategic'
            supplier_metrics.loc[
                (supplier_metrics['total_spending'] >= spending_threshold) & 
                (supplier_metrics['transaction_count'] < frequency_threshold), 'segment'
            ] = 'High Value'
            supplier_metrics.loc[
                (supplier_metrics['total_spending'] < spending_threshold) & 
                (supplier_metrics['transaction_count'] >= frequency_threshold), 'segment'
            ] = 'Frequent'
            
            # Segment analysis
            segment_stats = supplier_metrics.groupby('segment').agg({
                'total_spending': ['sum', 'mean', 'count'],
                'transaction_count': 'mean',
                'avg_transaction_value': 'mean'
            }).round(2)
            
            return {
                "total_suppliers": len(supplier_metrics),
                "supplier_segments": {
                    "Strategic": len(supplier_metrics[supplier_metrics['segment'] == 'Strategic']),
                    "High_Value": len(supplier_metrics[supplier_metrics['segment'] == 'High Value']),
                    "Frequent": len(supplier_metrics[supplier_metrics['segment'] == 'Frequent']),
                    "Standard": len(supplier_metrics[supplier_metrics['segment'] == 'Standard'])
                },
                "segment_statistics": segment_stats.to_dict('index'),
                "top_suppliers": supplier_metrics.nlargest(10, 'total_spending')[['total_spending', 'transaction_count', 'segment']].to_dict('index'),
                "segmentation_thresholds": {
                    "spending_threshold": round(spending_threshold, 2),
                    "frequency_threshold": frequency_threshold
                }
            }
            
        except Exception as e:
            return {"error": f"Supplier segmentation failed: {str(e)}"}
    
    def analyze_purchase_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze purchase trends over time"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for positive amounts
            purchase_data = data[data['amount'] > 0].copy()
            purchase_data['date'] = pd.to_datetime(purchase_data['date'])
            
            # Monthly trends
            monthly_purchases = purchase_data.groupby(purchase_data['date'].dt.to_period('M')).agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            }).round(2)
            
            # Weekly trends
            weekly_purchases = purchase_data.groupby(purchase_data['date'].dt.to_period('W')).agg({
                'amount': ['sum', 'count'],
                'party_name': 'nunique'
            }).round(2)
            
            # Daily of week analysis
            purchase_data['day_of_week'] = purchase_data['date'].dt.day_name()
            daily_patterns = purchase_data.groupby('day_of_week')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            # Growth analysis
            monthly_spending = purchase_data.groupby(purchase_data['date'].dt.to_period('M'))['amount'].sum()
            monthly_growth = monthly_spending.pct_change().fillna(0) * 100
            
            return {
                "monthly_trends": {
                    "periods": len(monthly_purchases),
                    "avg_monthly_spending": round(monthly_purchases[('amount', 'sum')].mean(), 2),
                    "peak_month": monthly_purchases[('amount', 'sum')].idxmax().strftime('%Y-%m'),
                    "low_month": monthly_purchases[('amount', 'sum')].idxmin().strftime('%Y-%m')
                },
                "weekly_trends": {
                    "total_weeks": len(weekly_purchases),
                    "avg_weekly_spending": round(weekly_purchases[('amount', 'sum')].mean(), 2)
                },
                "daily_patterns": daily_patterns.to_dict('index'),
                "growth_analysis": {
                    "avg_monthly_growth": round(monthly_growth.mean(), 2),
                    "max_monthly_growth": round(monthly_growth.max(), 2),
                    "min_monthly_growth": round(monthly_growth.min(), 2)
                },
                "trend_direction": self.identify_trends(monthly_spending)
            }
            
        except Exception as e:
            return {"error": f"Purchase trends analysis failed: {str(e)}"}
    
    def analyze_cost_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost patterns and spending distribution"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for items with positive amounts
            cost_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(cost_data) == 0:
                return {"error": "No cost data available"}
            
            # Cost metrics by item
            cost_metrics = cost_data.groupby('item').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': ['sum', 'mean'],
                'rate': ['mean', 'min', 'max']
            }).round(2)
            
            # Flatten column names
            cost_metrics.columns = ['_'.join(col).strip() for col in cost_metrics.columns]
            
            # Top cost items
            top_items_by_cost = cost_metrics.nlargest(10, 'amount_sum')
            top_items_by_quantity = cost_metrics.nlargest(10, 'quantity_sum')
            top_items_by_frequency = cost_metrics.nlargest(10, 'amount_count')
            
            return {
                "total_items": len(cost_metrics),
                "cost_performance": {
                    "total_item_cost": round(cost_metrics['amount_sum'].sum(), 2),
                    "avg_item_cost": round(cost_metrics['amount_sum'].mean(), 2),
                    "highest_cost_item": cost_metrics['amount_sum'].idxmax(),
                    "most_purchased_item": cost_metrics['quantity_sum'].idxmax()
                },
                "top_items": {
                    "by_cost": top_items_by_cost[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index'),
                    "by_quantity": top_items_by_quantity[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index'),
                    "by_frequency": top_items_by_frequency[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index')
                },
                "cost_statistics": {
                    "avg_item_price": round(cost_metrics['rate_mean'].mean(), 2),
                    "price_range": {
                        "min": round(cost_metrics['rate_min'].min(), 2),
                        "max": round(cost_metrics['rate_max'].max(), 2)
                    }
                }
            }
            
        except Exception as e:
            return {"error": f"Cost analysis failed: {str(e)}"}
    
    def analyze_transaction_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction patterns and characteristics"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Transaction analysis
            transaction_data = data[data['amount'] > 0].copy()
            transaction_data['date'] = pd.to_datetime(transaction_data['date'])
            
            # Transaction size analysis
            transaction_sizes = transaction_data.groupby('voucher_number')['amount'].sum()
            
            # Time patterns
            transaction_data['hour'] = transaction_data['date'].dt.hour
            transaction_data['day_of_week'] = transaction_data['date'].dt.dayofweek
            transaction_data['month'] = transaction_data['date'].dt.month
            
            # Hourly patterns
            hourly_patterns = transaction_data.groupby('hour')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            # Monthly patterns
            monthly_patterns = transaction_data.groupby('month')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            return {
                "transaction_characteristics": {
                    "total_transactions": len(transaction_data),
                    "avg_transaction_size": round(transaction_sizes.mean(), 2),
                    "median_transaction_size": round(transaction_sizes.median(), 2),
                    "max_transaction_size": round(transaction_sizes.max(), 2),
                    "min_transaction_size": round(transaction_sizes.min(), 2)
                },
                "time_patterns": {
                    "peak_hour": hourly_patterns['sum'].idxmax(),
                    "peak_month": monthly_patterns['sum'].idxmax(),
                    "busiest_day_of_week": transaction_data['day_of_week'].value_counts().idxmax()
                },
                "transaction_distribution": {
                    "large_transactions": len(transaction_sizes[transaction_sizes > transaction_sizes.quantile(0.9)]),
                    "small_transactions": len(transaction_sizes[transaction_sizes < transaction_sizes.quantile(0.1)]),
                    "medium_transactions": len(transaction_sizes[
                        (transaction_sizes >= transaction_sizes.quantile(0.1)) & 
                        (transaction_sizes <= transaction_sizes.quantile(0.9))
                    ])
                }
            }
            
        except Exception as e:
            return {"error": f"Transaction patterns analysis failed: {str(e)}"}
    
    def analyze_voucher_types(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voucher types and their characteristics"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Voucher type analysis
            voucher_analysis = data.groupby('voucher_type').agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique',
                'voucher_number': 'nunique'
            }).round(2)
            
            # Flatten column names
            voucher_analysis.columns = ['_'.join(col).strip() for col in voucher_analysis.columns]
            
            return {
                "voucher_types": len(voucher_analysis),
                "voucher_analysis": voucher_analysis.to_dict('index'),
                "dominant_voucher_type": voucher_analysis['amount_sum'].idxmax(),
                "most_frequent_voucher_type": voucher_analysis['amount_count'].idxmax()
            }
            
        except Exception as e:
            return {"error": f"Voucher type analysis failed: {str(e)}"}
    
    def analyze_warehouse_distribution(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze warehouse distribution of purchases"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Warehouse analysis based on available data
            if 'godown' in data.columns:
                warehouse_analysis = data[data['amount'] > 0].groupby('godown').agg({
                    'amount': ['sum', 'count', 'mean'],
                    'party_name': 'nunique'
                }).round(2)
                
                # Flatten column names
                warehouse_analysis.columns = ['_'.join(col).strip() for col in warehouse_analysis.columns]
                
                return {
                    "warehouse_distribution": warehouse_analysis.to_dict('index'),
                    "top_warehouse": warehouse_analysis['amount_sum'].idxmax(),
                    "total_warehouses": len(warehouse_analysis)
                }
            else:
                return {"error": "No warehouse data available"}
                
        except Exception as e:
            return {"error": f"Warehouse analysis failed: {str(e)}"}
    
    def analyze_time_series_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time series patterns in purchase data"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Time series analysis
            purchase_data = data[data['amount'] > 0].copy()
            purchase_data['date'] = pd.to_datetime(purchase_data['date'])
            
            # Daily time series
            daily_purchases = purchase_data.groupby(purchase_data['date'].dt.date)['amount'].sum()
            
            # Calculate moving averages
            daily_purchases_series = pd.Series(daily_purchases.values, index=pd.to_datetime(daily_purchases.index))
            ma_7 = daily_purchases_series.rolling(window=7).mean()
            ma_30 = daily_purchases_series.rolling(window=30).mean()
            
            # Seasonality analysis
            purchase_data['quarter'] = purchase_data['date'].dt.quarter
            quarterly_purchases = purchase_data.groupby('quarter')['amount'].sum()
            
            return {
                "time_series_stats": {
                    "total_days": len(daily_purchases),
                    "avg_daily_purchases": round(daily_purchases.mean(), 2),
                    "max_daily_purchases": round(daily_purchases.max(), 2),
                    "min_daily_purchases": round(daily_purchases.min(), 2),
                    "purchase_volatility": round(daily_purchases.std(), 2)
                },
                "moving_averages": {
                    "ma_7_current": round(ma_7.iloc[-1], 2) if len(ma_7) > 0 else 0,
                    "ma_30_current": round(ma_30.iloc[-1], 2) if len(ma_30) > 0 else 0
                },
                "quarterly_patterns": quarterly_purchases.to_dict(),
                "peak_quarter": quarterly_purchases.idxmax(),
                "trend_direction": self.identify_trends(daily_purchases_series)
            }
            
        except Exception as e:
            return {"error": f"Time series analysis failed: {str(e)}"}
    
    def analyze_general_purchase_summary(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General purchase summary analysis"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General summary
            purchase_data = data[data['amount'] > 0]
            
            return {
                "total_records": len(data),
                "total_purchase_records": len(purchase_data),
                "total_spending": round(purchase_data['amount'].sum(), 2),
                "unique_suppliers": data['party_name'].nunique(),
                "unique_items": data['item'].nunique() if 'item' in data.columns else 0,
                "date_range": {
                    "start": data['date'].min(),
                    "end": data['date'].max()
                },
                "voucher_types": data['voucher_type'].nunique()
            }
            
        except Exception as e:
            return {"error": f"General purchase summary failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis based on query"""
        query_lower = query.lower()
        
        # More specific patterns to avoid conflicts
        if any(word in query_lower for word in ['procurement', 'spend', 'cost', 'money']):
            return 'procurement_analysis'
        elif any(word in query_lower for word in ['supplier', 'vendor', 'segment']):
            return 'supplier_segmentation'
        elif any(word in query_lower for word in ['trend', 'growth', 'pattern', 'time']):
            return 'purchase_trends'
        elif any(word in query_lower for word in ['cost', 'price', 'expense', 'item']):
            return 'cost_analysis'
        elif any(word in query_lower for word in ['transaction', 'size', 'frequency']):
            return 'transaction_patterns'
        elif any(word in query_lower for word in ['voucher', 'type', 'category']):
            return 'voucher_type_analysis'
        elif any(word in query_lower for word in ['warehouse', 'location', 'godown']):
            return 'warehouse_distribution'
        elif any(word in query_lower for word in ['time series', 'seasonal', 'quarterly']):
            return 'time_series_analysis'
        else:
            return 'general_summary'