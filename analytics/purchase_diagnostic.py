"""
Purchase Diagnostic Analytics - "Why did it happen?"
Root cause analysis for purchase patterns and performance issues
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from .base import AnalyticsBase, AnalyticsResponse


class PurchaseDiagnosticAnalytics(AnalyticsBase):
    """Diagnostic analytics for purchase root cause analysis"""
    
    def __init__(self, agent_type: str = "purchase"):
        super().__init__("PurchaseDiagnosticAnalytics", agent_type)
        self.supported_queries = [
            'cost_increase_analysis',
            'supplier_performance_issues',
            'procurement_efficiency_issues',
            'transaction_anomaly_detection',
            'item_cost_variance_analysis',
            'seasonal_variance_analysis',
            'supplier_reliability_issues',
            'voucher_type_performance_issues'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform diagnostic analytics on purchase data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'cost_increase_analysis':
            results = self.analyze_cost_increases(data, params)
        elif analysis_type == 'supplier_performance_issues':
            results = self.analyze_supplier_performance_issues(data, params)
        elif analysis_type == 'procurement_efficiency_issues':
            results = self.analyze_procurement_efficiency_issues(data, params)
        elif analysis_type == 'transaction_anomaly_detection':
            results = self.analyze_transaction_anomalies(data, params)
        elif analysis_type == 'item_cost_variance_analysis':
            results = self.analyze_item_cost_variance(data, params)
        elif analysis_type == 'seasonal_variance_analysis':
            results = self.analyze_seasonal_variance(data, params)
        elif analysis_type == 'supplier_reliability_issues':
            results = self.analyze_supplier_reliability_issues(data, params)
        elif analysis_type == 'voucher_type_performance_issues':
            results = self.analyze_voucher_performance_issues(data, params)
        else:
            results = self.analyze_general_purchase_diagnostics(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_cost_increases(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of cost increases"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter cost data
            cost_data = data[data['amount'] > 0].copy()
            cost_data['date'] = pd.to_datetime(cost_data['date'])
            
            # Monthly cost trends
            monthly_costs = cost_data.groupby(cost_data['date'].dt.to_period('M'))['amount'].sum()
            monthly_growth = monthly_costs.pct_change().fillna(0) * 100
            
            # Identify cost increase periods
            cost_increase_periods = monthly_growth[monthly_growth > 10].index.tolist()
            
            # Root cause analysis
            root_causes = {}
            
            # Price increase analysis
            if 'rate' in cost_data.columns:
                item_price_changes = cost_data.groupby(['item', cost_data['date'].dt.to_period('M')]).agg({
                    'rate': 'mean'
                }).reset_index()
                item_price_changes['price_change'] = item_price_changes.groupby('item')['rate'].pct_change() * 100
                high_price_increases = item_price_changes[item_price_changes['price_change'] > 20]
                root_causes['price_increase_items'] = len(high_price_increases)
            
            # Volume increase analysis
            monthly_quantities = cost_data.groupby(cost_data['date'].dt.to_period('M'))['quantity'].sum()
            quantity_increase = monthly_quantities.pct_change().fillna(0) * 100
            root_causes['volume_increase_periods'] = len(quantity_increase[quantity_increase > 15])
            
            # Supplier cost analysis
            supplier_cost_changes = cost_data.groupby(['party_name', cost_data['date'].dt.to_period('M')]).agg({
                'amount': 'sum'
            }).reset_index()
            supplier_cost_changes['cost_change'] = supplier_cost_changes.groupby('party_name')['amount'].pct_change() * 100
            high_cost_suppliers = supplier_cost_changes[supplier_cost_changes['cost_change'] > 25]
            root_causes['high_cost_increase_suppliers'] = len(high_cost_suppliers)
            
            return {
                "cost_increase_periods": len(cost_increase_periods),
                "total_cost_increase": round(monthly_growth.max(), 2),
                "root_causes": root_causes,
                "increase_periods": [str(period) for period in cost_increase_periods],
                "cost_analysis": {
                    "avg_monthly_increase": round(monthly_growth.mean(), 2),
                    "max_monthly_increase": round(monthly_growth.max(), 2),
                    "price_volatility": round(monthly_growth.std(), 2)
                },
                "recommendations": self._generate_cost_increase_recommendations(root_causes)
            }
            
        except Exception as e:
            return {"error": f"Cost increase analysis failed: {str(e)}"}
    
    def analyze_supplier_performance_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supplier performance issues and reliability"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier performance analysis
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Calculate supplier metrics
            supplier_metrics = supplier_data.groupby('party_name').agg({
                'date': ['min', 'max', 'count'],
                'amount': ['sum', 'mean', 'std'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            supplier_metrics.columns = ['_'.join(col).strip() for col in supplier_metrics.columns]
            
            # Calculate supplier reliability metrics
            current_date = supplier_data['date'].max()
            supplier_metrics['days_since_last'] = (current_date - pd.to_datetime(supplier_metrics['date_max'])).dt.days
            supplier_metrics['supplier_relationship_days'] = (
                pd.to_datetime(supplier_metrics['date_max']) - 
                pd.to_datetime(supplier_metrics['date_min'])
            ).dt.days
            
            # Identify problematic suppliers
            reliability_threshold = params.get('reliability_threshold', 60)
            inconsistent_suppliers = supplier_metrics[supplier_metrics['days_since_last'] > reliability_threshold]
            
            # Price volatility analysis
            high_volatility_suppliers = supplier_metrics[
                supplier_metrics['amount_std'] > supplier_metrics['amount_mean']
            ]
            
            # Performance issues
            performance_issues = {
                'inactive_suppliers': len(inconsistent_suppliers),
                'high_volatility_suppliers': len(high_volatility_suppliers),
                'one_time_suppliers': len(supplier_metrics[supplier_metrics['voucher_number_nunique'] == 1])
            }
            
            return {
                "total_suppliers": len(supplier_metrics),
                "suppliers_with_issues": len(inconsistent_suppliers) + len(high_volatility_suppliers),
                "performance_issues": performance_issues,
                "supplier_reliability": {
                    "avg_days_since_last": round(supplier_metrics['days_since_last'].mean(), 2),
                    "inactive_suppliers": len(inconsistent_suppliers),
                    "avg_relationship_duration": round(supplier_metrics['supplier_relationship_days'].mean(), 2)
                },
                "recommendations": self._generate_supplier_performance_recommendations(performance_issues)
            }
            
        except Exception as e:
            return {"error": f"Supplier performance analysis failed: {str(e)}"}
    
    def analyze_procurement_efficiency_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze procurement efficiency issues and bottlenecks"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Performance metrics
            procurement_data = data[data['amount'] > 0].copy()
            procurement_data['date'] = pd.to_datetime(procurement_data['date'])
            
            # Daily performance analysis
            daily_performance = procurement_data.groupby(procurement_data['date'].dt.date).agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            })
            
            # Identify efficiency issues
            daily_spending = daily_performance[('amount', 'sum')]
            daily_transactions = daily_performance[('amount', 'count')]
            
            # Statistical analysis
            spending_mean = daily_spending.mean()
            spending_std = daily_spending.std()
            
            # Low efficiency days
            low_efficiency_days = daily_spending[daily_spending < (spending_mean - spending_std)]
            
            # Analyze causes
            efficiency_issues = {
                'low_spending_days': len(low_efficiency_days),
                'avg_low_day_spending': round(low_efficiency_days.mean(), 2),
                'spending_volatility': round(spending_std / spending_mean * 100, 2)
            }
            
            # Transaction efficiency
            avg_transaction_value = daily_spending / daily_transactions
            low_value_days = avg_transaction_value[avg_transaction_value < avg_transaction_value.quantile(0.25)]
            efficiency_issues['low_value_transaction_days'] = len(low_value_days)
            
            return {
                "efficiency_issues": efficiency_issues,
                "low_efficiency_analysis": {
                    "total_low_days": len(low_efficiency_days),
                    "percentage_low_days": round(len(low_efficiency_days) / len(daily_spending) * 100, 2),
                    "efficiency_impact": round((spending_mean - low_efficiency_days.mean()) * len(low_efficiency_days), 2)
                },
                "recommendations": self._generate_efficiency_recommendations(efficiency_issues)
            }
            
        except Exception as e:
            return {"error": f"Procurement efficiency analysis failed: {str(e)}"}
    
    def analyze_transaction_anomalies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and analyze transaction anomalies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Transaction anomaly detection
            transaction_data = data[data['amount'] > 0].copy()
            
            # Statistical anomaly detection using Z-score
            z_scores = np.abs(stats.zscore(transaction_data['amount']))
            anomaly_threshold = params.get('anomaly_threshold', 3)
            
            anomalies = transaction_data[z_scores > anomaly_threshold]
            
            # Analyze anomaly patterns
            anomaly_analysis = {
                'high_value_anomalies': len(anomalies[anomalies['amount'] > transaction_data['amount'].quantile(0.99)]),
                'low_value_anomalies': len(anomalies[anomalies['amount'] < transaction_data['amount'].quantile(0.01)]),
                'supplier_anomalies': anomalies['party_name'].nunique()
            }
            
            # Quantity-based anomalies
            if 'quantity' in transaction_data.columns:
                quantity_z_scores = np.abs(stats.zscore(transaction_data['quantity']))
                quantity_anomalies = transaction_data[quantity_z_scores > anomaly_threshold]
                anomaly_analysis['quantity_anomalies'] = len(quantity_anomalies)
            
            return {
                "total_transactions": len(transaction_data),
                "anomalous_transactions": len(anomalies),
                "anomaly_rate": round(len(anomalies) / len(transaction_data) * 100, 2),
                "anomaly_analysis": anomaly_analysis,
                "anomaly_impact": {
                    "total_anomaly_value": round(anomalies['amount'].sum(), 2),
                    "avg_anomaly_value": round(anomalies['amount'].mean(), 2),
                    "max_anomaly_value": round(anomalies['amount'].max(), 2)
                },
                "recommendations": self._generate_anomaly_recommendations(anomaly_analysis)
            }
            
        except Exception as e:
            return {"error": f"Transaction anomaly analysis failed: {str(e)}"}
    
    def analyze_item_cost_variance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze item cost variance and pricing issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Item cost variance analysis
            item_data = data[(data['amount'] > 0) & (data['item'].notna()) & (data['rate'].notna())].copy()
            
            if len(item_data) == 0:
                return {"error": "No item cost data available"}
            
            # Item price metrics
            item_metrics = item_data.groupby('item').agg({
                'rate': ['mean', 'std', 'min', 'max', 'count'],
                'amount': ['sum', 'count']
            }).round(2)
            
            # Flatten column names
            item_metrics.columns = ['_'.join(col).strip() for col in item_metrics.columns]
            
            # Calculate price variance
            item_metrics['price_variance'] = (item_metrics['rate_std'] / item_metrics['rate_mean']) * 100
            item_metrics['price_range'] = item_metrics['rate_max'] - item_metrics['rate_min']
            
            # Identify high variance items
            high_variance_threshold = params.get('variance_threshold', 30)
            high_variance_items = item_metrics[item_metrics['price_variance'] > high_variance_threshold]
            
            # Analyze variance causes
            variance_analysis = {
                'high_variance_items': len(high_variance_items),
                'avg_price_variance': round(item_metrics['price_variance'].mean(), 2),
                'items_with_wide_range': len(item_metrics[item_metrics['price_range'] > item_metrics['rate_mean']])
            }
            
            return {
                "total_items": len(item_metrics),
                "high_variance_items": len(high_variance_items),
                "variance_rate": round(len(high_variance_items) / len(item_metrics) * 100, 2),
                "variance_analysis": variance_analysis,
                "cost_impact": {
                    "avg_variance": round(item_metrics['price_variance'].mean(), 2),
                    "max_variance": round(item_metrics['price_variance'].max(), 2),
                    "variance_cost_impact": round(high_variance_items['amount_sum'].sum(), 2)
                },
                "recommendations": self._generate_variance_recommendations(variance_analysis)
            }
            
        except Exception as e:
            return {"error": f"Item cost variance analysis failed: {str(e)}"}
    
    def analyze_seasonal_variance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal variances and their causes"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Seasonal analysis
            seasonal_data = data[data['amount'] > 0].copy()
            seasonal_data['date'] = pd.to_datetime(seasonal_data['date'])
            seasonal_data['month'] = seasonal_data['date'].dt.month
            seasonal_data['quarter'] = seasonal_data['date'].dt.quarter
            
            # Monthly variance
            monthly_spending = seasonal_data.groupby('month')['amount'].sum()
            monthly_variance = monthly_spending.std() / monthly_spending.mean() * 100
            
            # Quarterly variance
            quarterly_spending = seasonal_data.groupby('quarter')['amount'].sum()
            quarterly_variance = quarterly_spending.std() / quarterly_spending.mean() * 100
            
            # Identify seasonal patterns
            peak_month = monthly_spending.idxmax()
            low_month = monthly_spending.idxmin()
            
            variance_analysis = {
                'high_seasonal_variance': monthly_variance > 30,
                'peak_to_low_ratio': monthly_spending.max() / monthly_spending.min(),
                'consistent_quarters': len(quarterly_spending[quarterly_spending > quarterly_spending.mean() * 0.8])
            }
            
            return {
                "seasonal_variance": {
                    "monthly_variance": round(monthly_variance, 2),
                    "quarterly_variance": round(quarterly_variance, 2),
                    "peak_month": peak_month,
                    "low_month": low_month
                },
                "variance_analysis": variance_analysis,
                "seasonal_patterns": {
                    "monthly_spending": monthly_spending.to_dict(),
                    "quarterly_spending": quarterly_spending.to_dict()
                },
                "recommendations": self._generate_seasonal_recommendations(variance_analysis)
            }
            
        except Exception as e:
            return {"error": f"Seasonal variance analysis failed: {str(e)}"}
    
    def analyze_supplier_reliability_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supplier reliability and consistency issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier reliability analysis
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Calculate supplier consistency metrics
            supplier_consistency = supplier_data.groupby('party_name').agg({
                'amount': ['count', 'std', 'mean'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            supplier_consistency.columns = ['_'.join(col).strip() for col in supplier_consistency.columns]
            
            # Calculate reliability metrics
            supplier_consistency['transaction_frequency'] = supplier_consistency['amount_count'] / (
                (pd.to_datetime(supplier_consistency['date_max']) - pd.to_datetime(supplier_consistency['date_min'])).dt.days + 1
            )
            supplier_consistency['price_consistency'] = supplier_consistency['amount_std'] / supplier_consistency['amount_mean']
            
            # Identify reliability issues
            reliability_issues = {
                'inconsistent_suppliers': len(supplier_consistency[supplier_consistency['price_consistency'] > 0.5]),
                'low_frequency_suppliers': len(supplier_consistency[supplier_consistency['transaction_frequency'] < 0.1]),
                'one_time_suppliers': len(supplier_consistency[supplier_consistency['voucher_number_nunique'] == 1])
            }
            
            return {
                "total_suppliers": len(supplier_consistency),
                "reliability_issues": reliability_issues,
                "consistency_metrics": {
                    "avg_price_consistency": round(supplier_consistency['price_consistency'].mean(), 2),
                    "avg_transaction_frequency": round(supplier_consistency['transaction_frequency'].mean(), 2),
                    "suppliers_with_issues": len(supplier_consistency[
                        (supplier_consistency['price_consistency'] > 0.5) | 
                        (supplier_consistency['transaction_frequency'] < 0.1)
                    ])
                },
                "recommendations": self._generate_reliability_recommendations(reliability_issues)
            }
            
        except Exception as e:
            return {"error": f"Supplier reliability analysis failed: {str(e)}"}
    
    def analyze_voucher_performance_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voucher type performance issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Voucher performance analysis
            voucher_metrics = data[data['amount'] > 0].groupby('voucher_type').agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            }).round(2)
            
            # Flatten column names
            voucher_metrics.columns = ['_'.join(col).strip() for col in voucher_metrics.columns]
            
            # Identify underperforming voucher types
            avg_spending = voucher_metrics['amount_sum'].mean()
            avg_frequency = voucher_metrics['amount_count'].mean()
            
            underperforming_vouchers = voucher_metrics[
                (voucher_metrics['amount_sum'] < avg_spending * 0.5) |
                (voucher_metrics['amount_count'] < avg_frequency * 0.5)
            ]
            
            performance_issues = {
                'low_spending_vouchers': len(underperforming_vouchers[underperforming_vouchers['amount_sum'] < avg_spending * 0.5]),
                'low_frequency_vouchers': len(underperforming_vouchers[underperforming_vouchers['amount_count'] < avg_frequency * 0.5]),
                'total_underperforming': len(underperforming_vouchers)
            }
            
            return {
                "voucher_performance_issues": performance_issues,
                "underperforming_vouchers": underperforming_vouchers.to_dict('index'),
                "impact_analysis": {
                    "lost_efficiency": round((avg_spending - underperforming_vouchers['amount_sum'].mean()) * len(underperforming_vouchers), 2),
                    "process_efficiency_loss": round((avg_frequency - underperforming_vouchers['amount_count'].mean()) / avg_frequency * 100, 2)
                },
                "recommendations": self._generate_voucher_recommendations(performance_issues)
            }
            
        except Exception as e:
            return {"error": f"Voucher performance analysis failed: {str(e)}"}
    
    def analyze_general_purchase_diagnostics(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General purchase diagnostic analysis"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General diagnostics
            purchase_data = data[data['amount'] > 0]
            
            # Data quality issues
            quality_issues = {
                'missing_supplier_names': data['party_name'].isna().sum(),
                'zero_amounts': len(data[data['amount'] == 0]),
                'negative_amounts': len(data[data['amount'] < 0]),
                'missing_dates': data['date'].isna().sum()
            }
            
            return {
                "total_records": len(data),
                "purchase_records": len(purchase_data),
                "data_quality_issues": quality_issues,
                "general_patterns": {
                    "avg_transaction_value": round(purchase_data['amount'].mean(), 2),
                    "transaction_volatility": round(purchase_data['amount'].std() / purchase_data['amount'].mean() * 100, 2),
                    "supplier_concentration": round(purchase_data['party_name'].nunique() / len(purchase_data) * 100, 2)
                }
            }
            
        except Exception as e:
            return {"error": f"General purchase diagnostics failed: {str(e)}"}
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic analysis results"""
        if 'recommendations' in analysis_results:
            return analysis_results['recommendations']
        return []
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of diagnostic analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cost', 'increase', 'expense', 'rise']):
            return 'cost_increase_analysis'
        elif any(word in query_lower for word in ['supplier', 'performance', 'vendor', 'reliability']):
            return 'supplier_performance_issues'
        elif any(word in query_lower for word in ['efficiency', 'procurement', 'bottleneck']):
            return 'procurement_efficiency_issues'
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'outlier', 'irregular']):
            return 'transaction_anomaly_detection'
        elif any(word in query_lower for word in ['variance', 'price', 'cost', 'item']):
            return 'item_cost_variance_analysis'
        elif any(word in query_lower for word in ['seasonal', 'variance', 'fluctuation']):
            return 'seasonal_variance_analysis'
        elif any(word in query_lower for word in ['reliability', 'consistency', 'supplier']):
            return 'supplier_reliability_issues'
        elif any(word in query_lower for word in ['voucher', 'type', 'performance']):
            return 'voucher_type_performance_issues'
        else:
            return 'general_diagnostics'
    
    # Recommendation generators
    def _generate_cost_increase_recommendations(self, root_causes: Dict[str, int]) -> List[str]:
        """Generate recommendations for cost increases"""
        recommendations = []
        
        if root_causes.get('price_increase_items', 0) > 0:
            recommendations.append("Negotiate better pricing with suppliers for high-increase items")
        
        if root_causes.get('volume_increase_periods', 0) > 0:
            recommendations.append("Implement demand planning to optimize purchase volumes")
        
        if root_causes.get('high_cost_increase_suppliers', 0) > 0:
            recommendations.append("Review supplier contracts and explore alternative suppliers")
        
        recommendations.append("Implement cost monitoring and early warning systems")
        
        return recommendations
    
    def _generate_supplier_performance_recommendations(self, issues: Dict[str, int]) -> List[str]:
        """Generate recommendations for supplier performance"""
        recommendations = []
        
        if issues.get('inactive_suppliers', 0) > 0:
            recommendations.append("Re-engage inactive suppliers or find replacements")
        
        if issues.get('high_volatility_suppliers', 0) > 0:
            recommendations.append("Negotiate price stability agreements with volatile suppliers")
        
        if issues.get('one_time_suppliers', 0) > 0:
            recommendations.append("Develop long-term relationships with reliable suppliers")
        
        recommendations.append("Implement supplier performance scorecards")
        
        return recommendations
    
    def _generate_efficiency_recommendations(self, issues: Dict[str, Any]) -> List[str]:
        """Generate recommendations for efficiency issues"""
        recommendations = []
        
        if issues.get('low_spending_days', 0) > 0:
            recommendations.append("Analyze and address factors causing low-efficiency days")
        
        if issues.get('spending_volatility', 0) > 30:
            recommendations.append("Implement procurement planning to reduce spending volatility")
        
        recommendations.append("Establish procurement performance monitoring")
        recommendations.append("Optimize procurement processes and workflows")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_analysis: Dict[str, int]) -> List[str]:
        """Generate recommendations for transaction anomalies"""
        recommendations = []
        
        if anomaly_analysis.get('high_value_anomalies', 0) > 0:
            recommendations.append("Implement approval workflows for high-value transactions")
        
        if anomaly_analysis.get('supplier_anomalies', 0) > 0:
            recommendations.append("Review and validate unusual supplier transactions")
        
        recommendations.append("Establish transaction monitoring and alerts")
        recommendations.append("Create procurement policies for anomalous transactions")
        
        return recommendations
    
    def _generate_variance_recommendations(self, variance_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for cost variance"""
        recommendations = []
        
        if variance_analysis.get('high_variance_items', 0) > 0:
            recommendations.append("Negotiate fixed pricing agreements for high-variance items")
        
        if variance_analysis.get('items_with_wide_range', 0) > 0:
            recommendations.append("Standardize procurement processes for consistent pricing")
        
        recommendations.append("Implement price benchmarking and monitoring")
        recommendations.append("Develop preferred supplier programs")
        
        return recommendations
    
    def _generate_seasonal_recommendations(self, variance: Dict[str, Any]) -> List[str]:
        """Generate recommendations for seasonal variance"""
        recommendations = []
        
        if variance.get('high_seasonal_variance', False):
            recommendations.append("Develop seasonal procurement planning strategies")
        
        recommendations.append("Implement inventory management for seasonal patterns")
        recommendations.append("Create targeted supplier agreements for seasonal fluctuations")
        
        return recommendations
    
    def _generate_reliability_recommendations(self, issues: Dict[str, int]) -> List[str]:
        """Generate recommendations for reliability issues"""
        recommendations = []
        
        if issues.get('inconsistent_suppliers', 0) > 0:
            recommendations.append("Implement supplier consistency monitoring")
        
        if issues.get('low_frequency_suppliers', 0) > 0:
            recommendations.append("Develop strategic partnerships with frequent suppliers")
        
        recommendations.append("Create supplier reliability scorecards")
        recommendations.append("Implement supplier development programs")
        
        return recommendations
    
    def _generate_voucher_recommendations(self, issues: Dict[str, int]) -> List[str]:
        """Generate recommendations for voucher performance"""
        recommendations = []
        
        if issues.get('low_spending_vouchers', 0) > 0:
            recommendations.append("Optimize or consolidate underperforming voucher types")
        
        if issues.get('low_frequency_vouchers', 0) > 0:
            recommendations.append("Streamline processes for underutilized voucher types")
        
        recommendations.append("Review voucher type effectiveness and workflows")
        recommendations.append("Implement voucher performance tracking")
        
        return recommendations