"""
Inventory Descriptive Analytics - "What happened?"
Historical inventory data analysis and current state reporting
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import AnalyticsBase, AnalyticsResponse


class InventoryDescriptiveAnalytics(AnalyticsBase):
    """Descriptive analytics for inventory historical data analysis"""
    
    def __init__(self, agent_type: str = "inventory"):
        super().__init__("InventoryDescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'stock_level_analysis',
            'inventory_turnover',
            'abc_analysis',
            'stock_movement_summary',
            'godown_utilization',
            'item_performance',
            'seasonal_patterns',
            'stock_aging_analysis'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform descriptive analytics on inventory data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'stock_level_analysis':
            results = self.analyze_stock_levels(data, params)
        elif analysis_type == 'inventory_turnover':
            results = self.analyze_inventory_turnover(data, params)
        elif analysis_type == 'abc_analysis':
            results = self.analyze_abc_classification(data, params)
        elif analysis_type == 'stock_movement_summary':
            results = self.analyze_stock_movements(data, params)
        elif analysis_type == 'godown_utilization':
            results = self.analyze_godown_utilization(data, params)
        elif analysis_type == 'item_performance':
            results = self.analyze_item_performance(data, params)
        elif analysis_type == 'seasonal_patterns':
            results = self.analyze_seasonal_patterns(data, params)
        elif analysis_type == 'stock_aging_analysis':
            results = self.analyze_stock_aging(data, params)
        else:
            results = self.analyze_general_inventory_summary(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_stock_levels(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current stock levels and distribution"""
        try:
            # Convert to DataFrame if needed
            if 'item' not in data.columns:
                return {"error": "Invalid data format - missing 'item' column"}
            
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate stock metrics
            stock_summary = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'mean', 'std'],
                'amount': ['sum', 'mean'],
                'rate': ['mean', 'min', 'max']
            }).round(2)
            
            # Flatten column names
            stock_summary.columns = ['_'.join(col).strip() for col in stock_summary.columns]
            
            # Calculate additional metrics
            total_items = len(stock_summary)
            total_value = stock_summary['amount_sum'].sum()
            avg_value_per_item = total_value / total_items if total_items > 0 else 0
            
            # Identify high/low stock items
            high_stock_items = stock_summary.nlargest(10, 'quantity_sum')
            low_stock_items = stock_summary.nsmallest(10, 'quantity_sum')
            
            return {
                "total_unique_items": total_items,
                "total_inventory_value": round(total_value, 2),
                "average_value_per_item": round(avg_value_per_item, 2),
                "highest_stock_items": high_stock_items.to_dict('index'),
                "lowest_stock_items": low_stock_items.to_dict('index'),
                "stock_distribution": {
                    "mean_quantity": round(stock_summary['quantity_sum'].mean(), 2),
                    "median_quantity": round(stock_summary['quantity_sum'].median(), 2),
                    "std_quantity": round(stock_summary['quantity_sum'].std(), 2)
                }
            }
            
        except Exception as e:
            return {"error": f"Stock level analysis failed: {str(e)}"}
    
    def analyze_inventory_turnover(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate inventory turnover ratios"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Group by item and calculate turnover metrics
            item_turnover = data.groupby('item').agg({
                'quantity': 'sum',
                'amount': 'sum',
                'date': ['min', 'max', 'count']
            }).round(2)
            
            # Calculate days between first and last transaction
            item_turnover['days_active'] = (
                pd.to_datetime(item_turnover[('date', 'max')]) - 
                pd.to_datetime(item_turnover[('date', 'min')])
            ).dt.days
            
            # Calculate turnover rate (transactions per day)
            item_turnover['turnover_rate'] = (
                item_turnover[('date', 'count')] / 
                item_turnover['days_active'].replace(0, 1)
            ).round(4)
            
            # Classify turnover speed
            turnover_avg = item_turnover['turnover_rate'].mean()
            fast_movers = item_turnover[item_turnover['turnover_rate'] > turnover_avg * 1.5]
            slow_movers = item_turnover[item_turnover['turnover_rate'] < turnover_avg * 0.5]
            
            return {
                "average_turnover_rate": round(turnover_avg, 4),
                "fast_moving_items": len(fast_movers),
                "slow_moving_items": len(slow_movers),
                "top_fast_movers": fast_movers.nlargest(10, 'turnover_rate').to_dict('index'),
                "top_slow_movers": slow_movers.nsmallest(10, 'turnover_rate').to_dict('index'),
                "turnover_distribution": {
                    "min": round(item_turnover['turnover_rate'].min(), 4),
                    "max": round(item_turnover['turnover_rate'].max(), 4),
                    "median": round(item_turnover['turnover_rate'].median(), 4),
                    "std": round(item_turnover['turnover_rate'].std(), 4)
                }
            }
            
        except Exception as e:
            return {"error": f"Inventory turnover analysis failed: {str(e)}"}
    
    def analyze_abc_classification(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ABC analysis based on value contribution"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate total value per item
            item_values = data.groupby('item')['amount'].sum().sort_values(ascending=False)
            total_value = item_values.sum()
            
            # Calculate cumulative percentage
            cumulative_pct = (item_values.cumsum() / total_value * 100).round(2)
            
            # Classify items
            a_items = cumulative_pct[cumulative_pct <= 80].index.tolist()
            b_items = cumulative_pct[(cumulative_pct > 80) & (cumulative_pct <= 95)].index.tolist()
            c_items = cumulative_pct[cumulative_pct > 95].index.tolist()
            
            return {
                "total_items": len(item_values),
                "total_value": round(total_value, 2),
                "a_category": {
                    "count": len(a_items),
                    "percentage": round(len(a_items) / len(item_values) * 100, 2),
                    "value_contribution": round(item_values[a_items].sum() / total_value * 100, 2),
                    "top_items": a_items[:10]
                },
                "b_category": {
                    "count": len(b_items),
                    "percentage": round(len(b_items) / len(item_values) * 100, 2),
                    "value_contribution": round(item_values[b_items].sum() / total_value * 100, 2),
                    "sample_items": b_items[:10]
                },
                "c_category": {
                    "count": len(c_items),
                    "percentage": round(len(c_items) / len(item_values) * 100, 2),
                    "value_contribution": round(item_values[c_items].sum() / total_value * 100, 2),
                    "sample_items": c_items[:10]
                }
            }
            
        except Exception as e:
            return {"error": f"ABC analysis failed: {str(e)}"}
    
    def analyze_stock_movements(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stock movement patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Daily movement analysis
            daily_movements = data.groupby(data['date'].dt.date).agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            # Monthly movement analysis
            monthly_movements = data.groupby(data['date'].dt.to_period('M')).agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            # Movement direction analysis (positive vs negative quantities)
            inbound_movements = data[data['quantity'] > 0]
            outbound_movements = data[data['quantity'] < 0]
            
            return {
                "total_movements": len(data),
                "date_range": {
                    "start": data['date'].min().strftime('%Y-%m-%d'),
                    "end": data['date'].max().strftime('%Y-%m-%d')
                },
                "daily_averages": {
                    "avg_quantity": round(daily_movements[('quantity', 'sum')].mean(), 2),
                    "avg_transactions": round(daily_movements[('quantity', 'count')].mean(), 2),
                    "avg_value": round(daily_movements[('amount', 'sum')].mean(), 2)
                },
                "movement_direction": {
                    "inbound_count": len(inbound_movements),
                    "outbound_count": len(outbound_movements),
                    "inbound_value": round(inbound_movements['amount'].sum(), 2),
                    "outbound_value": round(abs(outbound_movements['amount'].sum()), 2)
                },
                "busiest_days": daily_movements.nlargest(5, ('quantity', 'count')).index.tolist(),
                "highest_value_days": daily_movements.nlargest(5, ('amount', 'sum')).index.tolist()
            }
            
        except Exception as e:
            return {"error": f"Stock movement analysis failed: {str(e)}"}
    
    def analyze_godown_utilization(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze godown/warehouse utilization"""
        try:
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            if 'godown' not in data.columns:
                return {"error": "Godown data not available"}
            
            # Godown-wise analysis
            godown_summary = data.groupby('godown').agg({
                'item': 'nunique',
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            # Calculate utilization metrics
            total_items = data['item'].nunique()
            total_value = data['amount'].sum()
            
            utilization_metrics = {}
            for godown in godown_summary.index:
                godown_items = godown_summary.loc[godown, ('item', 'nunique')]
                godown_value = godown_summary.loc[godown, ('amount', 'sum')]
                
                utilization_metrics[godown] = {
                    "unique_items": godown_items,
                    "item_percentage": round(godown_items / total_items * 100, 2),
                    "total_value": round(godown_value, 2),
                    "value_percentage": round(godown_value / total_value * 100, 2),
                    "total_transactions": godown_summary.loc[godown, ('quantity', 'count')]
                }
            
            return {
                "total_godowns": len(godown_summary),
                "godown_utilization": utilization_metrics,
                "most_utilized_godown": max(utilization_metrics.items(), 
                                           key=lambda x: x[1]['value_percentage'])[0],
                "least_utilized_godown": min(utilization_metrics.items(), 
                                            key=lambda x: x[1]['value_percentage'])[0]
            }
            
        except Exception as e:
            return {"error": f"Godown utilization analysis failed: {str(e)}"}
    
    def analyze_item_performance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual item performance metrics"""
        try:
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Item performance metrics
            item_metrics = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'mean', 'std'],
                'amount': ['sum', 'mean'],
                'rate': ['mean', 'min', 'max'],
                'date': ['min', 'max']
            }).round(2)
            
            # Calculate performance scores
            item_metrics['performance_score'] = (
                item_metrics[('amount', 'sum')] * 0.4 +  # Value weight
                item_metrics[('quantity', 'count')] * 0.3 +  # Frequency weight
                item_metrics[('quantity', 'sum')] * 0.3  # Volume weight
            )
            
            # Normalize performance scores
            max_score = item_metrics['performance_score'].max()
            item_metrics['performance_score_normalized'] = (
                item_metrics['performance_score'] / max_score * 100
            ).round(2)
            
            # Top and bottom performers
            top_performers = item_metrics.nlargest(10, 'performance_score_normalized')
            bottom_performers = item_metrics.nsmallest(10, 'performance_score_normalized')
            
            return {
                "performance_metrics": {
                    "average_score": round(item_metrics['performance_score_normalized'].mean(), 2),
                    "median_score": round(item_metrics['performance_score_normalized'].median(), 2),
                    "std_score": round(item_metrics['performance_score_normalized'].std(), 2)
                },
                "top_performers": top_performers.to_dict('index'),
                "bottom_performers": bottom_performers.to_dict('index'),
                "performance_distribution": {
                    "excellent": len(item_metrics[item_metrics['performance_score_normalized'] > 80]),
                    "good": len(item_metrics[(item_metrics['performance_score_normalized'] > 60) & 
                                           (item_metrics['performance_score_normalized'] <= 80)]),
                    "average": len(item_metrics[(item_metrics['performance_score_normalized'] > 40) & 
                                              (item_metrics['performance_score_normalized'] <= 60)]),
                    "poor": len(item_metrics[item_metrics['performance_score_normalized'] <= 40])
                }
            }
            
        except Exception as e:
            return {"error": f"Item performance analysis failed: {str(e)}"}
    
    def analyze_seasonal_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns in inventory movements"""
        try:
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['day_of_week'] = data['date'].dt.dayofweek
            
            # Monthly patterns
            monthly_patterns = data.groupby('month').agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            # Quarterly patterns
            quarterly_patterns = data.groupby('quarter').agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            # Day of week patterns
            dow_patterns = data.groupby('day_of_week').agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            }).round(2)
            
            return {
                "monthly_patterns": monthly_patterns.to_dict('index'),
                "quarterly_patterns": quarterly_patterns.to_dict('index'),
                "day_of_week_patterns": dow_patterns.to_dict('index'),
                "peak_month": monthly_patterns[('amount', 'sum')].idxmax(),
                "low_month": monthly_patterns[('amount', 'sum')].idxmin(),
                "peak_quarter": quarterly_patterns[('amount', 'sum')].idxmax(),
                "busiest_day_of_week": dow_patterns[('quantity', 'count')].idxmax()
            }
            
        except Exception as e:
            return {"error": f"Seasonal pattern analysis failed: {str(e)}"}
    
    def analyze_stock_aging(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stock aging patterns"""
        try:
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            data['date'] = pd.to_datetime(data['date'])
            current_date = datetime.now()
            
            # Calculate days since last transaction for each item
            last_transaction = data.groupby('item')['date'].max()
            days_since_last = (current_date - last_transaction).dt.days
            
            # Age categories
            age_categories = {
                "fresh": days_since_last[days_since_last <= 30],
                "moderate": days_since_last[(days_since_last > 30) & (days_since_last <= 90)],
                "old": days_since_last[(days_since_last > 90) & (days_since_last <= 365)],
                "stale": days_since_last[days_since_last > 365]
            }
            
            # Calculate values for each category
            category_values = {}
            for category, items in age_categories.items():
                if len(items) > 0:
                    category_data = data[data['item'].isin(items.index)]
                    category_values[category] = {
                        "item_count": len(items),
                        "total_value": round(category_data['amount'].sum(), 2),
                        "average_age": round(items.mean(), 2),
                        "oldest_item": items.idxmax(),
                        "oldest_age": items.max()
                    }
            
            return {
                "aging_analysis": category_values,
                "overall_statistics": {
                    "average_age": round(days_since_last.mean(), 2),
                    "median_age": round(days_since_last.median(), 2),
                    "oldest_item": days_since_last.idxmax(),
                    "oldest_age": days_since_last.max(),
                    "newest_item": days_since_last.idxmin(),
                    "newest_age": days_since_last.min()
                }
            }
            
        except Exception as e:
            return {"error": f"Stock aging analysis failed: {str(e)}"}
    
    def analyze_general_inventory_summary(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General inventory summary analysis"""
        try:
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            return {
                "total_records": len(data),
                "unique_items": data['item'].nunique(),
                "date_range": {
                    "start": data['date'].min(),
                    "end": data['date'].max()
                },
                "total_value": round(data['amount'].sum(), 2),
                "total_quantity": round(data['quantity'].sum(), 2),
                "average_rate": round(data['rate'].mean(), 2),
                "unique_godowns": data['godown'].nunique() if 'godown' in data.columns else 0
            }
            
        except Exception as e:
            return {"error": f"General inventory summary failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of analysis based on query"""
        query_lower = query.lower()
        
        # More specific patterns to avoid conflicts
        if any(word in query_lower for word in ['abc', 'classification', 'categorization']):
            return 'abc_analysis'
        elif any(word in query_lower for word in ['turnover', 'moving', 'fast', 'slow']):
            return 'inventory_turnover'
        elif any(word in query_lower for word in ['movement', 'transaction', 'activity']):
            return 'stock_movement_summary'
        elif any(word in query_lower for word in ['godown', 'warehouse', 'location', 'utilization']):
            return 'godown_utilization'
        elif any(word in query_lower for word in ['performance', 'metrics', 'score']):
            return 'item_performance'
        elif any(word in query_lower for word in ['seasonal', 'monthly', 'quarterly', 'pattern']):
            return 'seasonal_patterns'
        elif any(word in query_lower for word in ['aging', 'age', 'old', 'stale']):
            return 'stock_aging_analysis'
        elif any(word in query_lower for word in ['summary', 'general', 'overview']):
            return 'general_summary'
        elif any(word in query_lower for word in ['stock', 'level', 'current', 'inventory']):
            return 'stock_level_analysis'
        else:
            return 'stock_level_analysis'  # Default