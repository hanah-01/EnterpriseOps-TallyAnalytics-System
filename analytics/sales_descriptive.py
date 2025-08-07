"""
Sales Descriptive Analytics - "What happened?"
Historical sales data analysis and current state reporting
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import AnalyticsBase, AnalyticsResponse


class SalesDescriptiveAnalytics(AnalyticsBase):
    """Descriptive analytics for sales historical data analysis"""
    
    def __init__(self, agent_type: str = "sales"):
        super().__init__("SalesDescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'revenue_analysis',
            'customer_segmentation',
            'sales_trends',
            'product_performance',
            'transaction_patterns',
            'voucher_type_analysis',
            'geographic_analysis',
            'time_series_analysis'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform descriptive analytics on sales data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'revenue_analysis':
            results = self.analyze_revenue_patterns(data, params)
        elif analysis_type == 'customer_segmentation':
            results = self.analyze_customer_segments(data, params)
        elif analysis_type == 'sales_trends':
            results = self.analyze_sales_trends(data, params)
        elif analysis_type == 'product_performance':
            results = self.analyze_product_performance(data, params)
        elif analysis_type == 'transaction_patterns':
            results = self.analyze_transaction_patterns(data, params)
        elif analysis_type == 'voucher_type_analysis':
            results = self.analyze_voucher_types(data, params)
        elif analysis_type == 'geographic_analysis':
            results = self.analyze_geographic_distribution(data, params)
        elif analysis_type == 'time_series_analysis':
            results = self.analyze_time_series_patterns(data, params)
        else:
            results = self.analyze_general_sales_summary(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_revenue_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze revenue patterns and distribution"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for positive amounts (revenue)
            revenue_data = data[data['amount'] > 0]
            
            # Calculate revenue metrics
            total_revenue = revenue_data['amount'].sum()
            avg_transaction_value = revenue_data['amount'].mean()
            median_transaction_value = revenue_data['amount'].median()
            revenue_std = revenue_data['amount'].std()
            
            # Revenue distribution
            revenue_quartiles = revenue_data['amount'].quantile([0.25, 0.5, 0.75])
            
            # Daily revenue analysis
            revenue_data['date'] = pd.to_datetime(revenue_data['date'])
            daily_revenue = revenue_data.groupby(revenue_data['date'].dt.date)['amount'].sum()
            
            # Top revenue days
            top_revenue_days = daily_revenue.nlargest(10)
            
            return {
                "total_revenue": round(total_revenue, 2),
                "avg_transaction_value": round(avg_transaction_value, 2),
                "median_transaction_value": round(median_transaction_value, 2),
                "revenue_std": round(revenue_std, 2),
                "revenue_quartiles": {
                    "q1": round(revenue_quartiles[0.25], 2),
                    "q2": round(revenue_quartiles[0.5], 2),
                    "q3": round(revenue_quartiles[0.75], 2)
                },
                "daily_revenue_stats": {
                    "avg_daily_revenue": round(daily_revenue.mean(), 2),
                    "max_daily_revenue": round(daily_revenue.max(), 2),
                    "min_daily_revenue": round(daily_revenue.min(), 2)
                },
                "top_revenue_days": top_revenue_days.to_dict(),
                "total_revenue_transactions": len(revenue_data)
            }
            
        except Exception as e:
            return {"error": f"Revenue analysis failed: {str(e)}"}
    
    def analyze_customer_segments(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer segmentation and behavior"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for customers with positive amounts
            customer_data = data[data['amount'] > 0].copy()
            
            # Customer metrics
            customer_metrics = customer_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': 'sum',
                'date': ['min', 'max', 'nunique']
            }).round(2)
            
            # Flatten column names
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Customer segmentation using RFM-like analysis
            customer_metrics['total_revenue'] = customer_metrics['amount_sum']
            customer_metrics['transaction_count'] = customer_metrics['amount_count']
            customer_metrics['avg_transaction_value'] = customer_metrics['amount_mean']
            
            # Segmentation thresholds
            revenue_threshold = customer_metrics['total_revenue'].quantile(0.8)
            frequency_threshold = customer_metrics['transaction_count'].quantile(0.8)
            
            # Create segments
            customer_metrics['segment'] = 'Regular'
            customer_metrics.loc[
                (customer_metrics['total_revenue'] >= revenue_threshold) & 
                (customer_metrics['transaction_count'] >= frequency_threshold), 'segment'
            ] = 'VIP'
            customer_metrics.loc[
                (customer_metrics['total_revenue'] >= revenue_threshold) & 
                (customer_metrics['transaction_count'] < frequency_threshold), 'segment'
            ] = 'High Value'
            customer_metrics.loc[
                (customer_metrics['total_revenue'] < revenue_threshold) & 
                (customer_metrics['transaction_count'] >= frequency_threshold), 'segment'
            ] = 'Frequent'
            
            # Segment analysis
            segment_stats = customer_metrics.groupby('segment').agg({
                'total_revenue': ['sum', 'mean', 'count'],
                'transaction_count': 'mean',
                'avg_transaction_value': 'mean'
            }).round(2)
            
            return {
                "total_customers": len(customer_metrics),
                "customer_segments": {
                    "VIP": len(customer_metrics[customer_metrics['segment'] == 'VIP']),
                    "High_Value": len(customer_metrics[customer_metrics['segment'] == 'High Value']),
                    "Frequent": len(customer_metrics[customer_metrics['segment'] == 'Frequent']),
                    "Regular": len(customer_metrics[customer_metrics['segment'] == 'Regular'])
                },
                "segment_statistics": segment_stats.to_dict('index'),
                "top_customers": customer_metrics.nlargest(10, 'total_revenue')[['total_revenue', 'transaction_count', 'segment']].to_dict('index'),
                "segmentation_thresholds": {
                    "revenue_threshold": round(revenue_threshold, 2),
                    "frequency_threshold": frequency_threshold
                }
            }
            
        except Exception as e:
            return {"error": f"Customer segmentation failed: {str(e)}"}
    
    def analyze_sales_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sales trends over time"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for positive amounts
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Monthly trends
            monthly_sales = sales_data.groupby(sales_data['date'].dt.to_period('M')).agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            }).round(2)
            
            # Weekly trends
            weekly_sales = sales_data.groupby(sales_data['date'].dt.to_period('W')).agg({
                'amount': ['sum', 'count'],
                'party_name': 'nunique'
            }).round(2)
            
            # Daily of week analysis
            sales_data['day_of_week'] = sales_data['date'].dt.day_name()
            daily_patterns = sales_data.groupby('day_of_week')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            # Growth analysis
            monthly_revenue = sales_data.groupby(sales_data['date'].dt.to_period('M'))['amount'].sum()
            monthly_growth = monthly_revenue.pct_change().fillna(0) * 100
            
            return {
                "monthly_trends": {
                    "periods": len(monthly_sales),
                    "avg_monthly_revenue": round(monthly_sales[('amount', 'sum')].mean(), 2),
                    "peak_month": monthly_sales[('amount', 'sum')].idxmax().strftime('%Y-%m'),
                    "low_month": monthly_sales[('amount', 'sum')].idxmin().strftime('%Y-%m')
                },
                "weekly_trends": {
                    "total_weeks": len(weekly_sales),
                    "avg_weekly_revenue": round(weekly_sales[('amount', 'sum')].mean(), 2)
                },
                "daily_patterns": daily_patterns.to_dict('index'),
                "growth_analysis": {
                    "avg_monthly_growth": round(monthly_growth.mean(), 2),
                    "max_monthly_growth": round(monthly_growth.max(), 2),
                    "min_monthly_growth": round(monthly_growth.min(), 2)
                },
                "trend_direction": self.identify_trends(monthly_revenue)
            }
            
        except Exception as e:
            return {"error": f"Sales trends analysis failed: {str(e)}"}
    
    def analyze_product_performance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product/item performance"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter for items with positive amounts
            product_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(product_data) == 0:
                return {"error": "No product data available"}
            
            # Product metrics
            product_metrics = product_data.groupby('item').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': ['sum', 'mean'],
                'rate': ['mean', 'min', 'max']
            }).round(2)
            
            # Flatten column names
            product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
            
            # Top performing products
            top_products_by_revenue = product_metrics.nlargest(10, 'amount_sum')
            top_products_by_quantity = product_metrics.nlargest(10, 'quantity_sum')
            top_products_by_frequency = product_metrics.nlargest(10, 'amount_count')
            
            return {
                "total_products": len(product_metrics),
                "product_performance": {
                    "total_product_revenue": round(product_metrics['amount_sum'].sum(), 2),
                    "avg_product_revenue": round(product_metrics['amount_sum'].mean(), 2),
                    "top_revenue_product": product_metrics['amount_sum'].idxmax(),
                    "top_quantity_product": product_metrics['quantity_sum'].idxmax()
                },
                "top_products": {
                    "by_revenue": top_products_by_revenue[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index'),
                    "by_quantity": top_products_by_quantity[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index'),
                    "by_frequency": top_products_by_frequency[['amount_sum', 'quantity_sum', 'amount_count']].to_dict('index')
                },
                "product_statistics": {
                    "avg_product_price": round(product_metrics['rate_mean'].mean(), 2),
                    "price_range": {
                        "min": round(product_metrics['rate_min'].min(), 2),
                        "max": round(product_metrics['rate_max'].max(), 2)
                    }
                }
            }
            
        except Exception as e:
            return {"error": f"Product performance analysis failed: {str(e)}"}
    
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
    
    def analyze_geographic_distribution(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze geographic distribution of sales"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Geographic analysis based on available data
            if 'godown' in data.columns:
                geo_analysis = data[data['amount'] > 0].groupby('godown').agg({
                    'amount': ['sum', 'count', 'mean'],
                    'party_name': 'nunique'
                }).round(2)
                
                # Flatten column names
                geo_analysis.columns = ['_'.join(col).strip() for col in geo_analysis.columns]
                
                return {
                    "geographic_distribution": geo_analysis.to_dict('index'),
                    "top_location": geo_analysis['amount_sum'].idxmax(),
                    "total_locations": len(geo_analysis)
                }
            else:
                return {"error": "No geographic data available"}
                
        except Exception as e:
            return {"error": f"Geographic analysis failed: {str(e)}"}
    
    def analyze_time_series_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time series patterns in sales data"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Time series analysis
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Daily time series
            daily_sales = sales_data.groupby(sales_data['date'].dt.date)['amount'].sum()
            
            # Calculate moving averages
            daily_sales_series = pd.Series(daily_sales.values, index=pd.to_datetime(daily_sales.index))
            ma_7 = daily_sales_series.rolling(window=7).mean()
            ma_30 = daily_sales_series.rolling(window=30).mean()
            
            # Seasonality analysis
            sales_data['quarter'] = sales_data['date'].dt.quarter
            quarterly_sales = sales_data.groupby('quarter')['amount'].sum()
            
            return {
                "time_series_stats": {
                    "total_days": len(daily_sales),
                    "avg_daily_sales": round(daily_sales.mean(), 2),
                    "max_daily_sales": round(daily_sales.max(), 2),
                    "min_daily_sales": round(daily_sales.min(), 2),
                    "sales_volatility": round(daily_sales.std(), 2)
                },
                "moving_averages": {
                    "ma_7_current": round(ma_7.iloc[-1], 2) if len(ma_7) > 0 else 0,
                    "ma_30_current": round(ma_30.iloc[-1], 2) if len(ma_30) > 0 else 0
                },
                "quarterly_patterns": quarterly_sales.to_dict(),
                "peak_quarter": quarterly_sales.idxmax(),
                "trend_direction": self.identify_trends(daily_sales_series)
            }
            
        except Exception as e:
            return {"error": f"Time series analysis failed: {str(e)}"}
    
    def analyze_general_sales_summary(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General sales summary analysis"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General summary
            sales_data = data[data['amount'] > 0]
            
            return {
                "total_records": len(data),
                "total_sales_records": len(sales_data),
                "total_revenue": round(sales_data['amount'].sum(), 2),
                "unique_customers": data['party_name'].nunique(),
                "unique_products": data['item'].nunique() if 'item' in data.columns else 0,
                "date_range": {
                    "start": data['date'].min(),
                    "end": data['date'].max()
                },
                "voucher_types": data['voucher_type'].nunique()
            }
            
        except Exception as e:
            return {"error": f"General sales summary failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis based on query"""
        query_lower = query.lower()
        
        # More specific patterns to avoid conflicts
        if any(word in query_lower for word in ['revenue', 'income', 'earnings', 'money']):
            return 'revenue_analysis'
        elif any(word in query_lower for word in ['customer', 'client', 'segment', 'rfm']):
            return 'customer_segmentation'
        elif any(word in query_lower for word in ['trend', 'growth', 'pattern', 'time']):
            return 'sales_trends'
        elif any(word in query_lower for word in ['product', 'item', 'performance', 'top']):
            return 'product_performance'
        elif any(word in query_lower for word in ['transaction', 'size', 'frequency']):
            return 'transaction_patterns'
        elif any(word in query_lower for word in ['voucher', 'type', 'category']):
            return 'voucher_type_analysis'
        elif any(word in query_lower for word in ['geographic', 'location', 'region', 'godown']):
            return 'geographic_analysis'
        elif any(word in query_lower for word in ['time series', 'seasonal', 'quarterly']):
            return 'time_series_analysis'
        else:
            return 'general_summary'